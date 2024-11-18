import os
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN, KMeans
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.preprocess import preprocess_pdfs
from src.utils import perform_ocr

HOME = os.environ["HOME"]


def ocr_image(image, output_path):
    if output_path.exists():
        return pd.read_csv(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.array(image)
    layout = perform_ocr(image)
    layout["x_center_100x"] = layout.x_center * 100
    layout.to_csv(output_path, index=False)

    return layout


def plot_layout(layout: pd.DataFrame):
    plt.figure()
    hue = layout.page.astype(str) + layout.column.astype(str)
    ax = sns.scatterplot(
        data=layout, x="x_center", y=-layout.y_center, hue=hue, palette="tab10"
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_ylabel("")

    return ax


def compute_perplexity(text, model, tokenizer, batch_size=32):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    num_batches = (input_ids.size(1) + batch_size - 1) // batch_size
    nlls = []
    for i in range(num_batches):
        batch_input_ids = input_ids[:, i * batch_size : (i + 1) * batch_size]
        batch_input_ids = batch_input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, labels=batch_input_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).mean()).item()


def reordered_clusters(
    layout: pd.DataFrame, clusters: np.ndarray
) -> np.ndarray:
    # Returns clusters layout["cluster"] reordered by increasing mean x_center
    cluster_centers = layout.groupby(clusters)["x_center"].mean().sort_values()
    reordered_clusters = clusters.copy()
    for i, cluster in enumerate(cluster_centers.index, 1):
        reordered_clusters[clusters == cluster] = i

    return reordered_clusters


def find_pages(layout: pd.DataFrame) -> pd.Series:
    x_span = layout.x_center_100x.max() - layout.x_center_100x.min()
    cluster_selection_epsilon = x_span / 4
    hdbscan = HDBSCAN(
        metric="chebyshev",
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=2,
    )
    pages = hdbscan.fit_predict(layout[["x_center_100x", "y_center"]])
    pages[pages == -1] = 0

    return reordered_clusters(layout, pages)


def ocr_and_cluster(issues: list[str], output_path: str = "results"):
    issues_images = preprocess_pdfs(issues, output_path=output_path)
    output_path = Path(output_path)
    issues_text = {}
    for issue in list(issues_images.keys()):
        issue_path = output_path / issue / "ocr.txt"
        if issue_path.exists():
            with open(issue_path, "r") as f:
                issues_text[issue] = f.read()
            issues_images.pop(issue)

    n_images = sum(len(images) for images in issues_images.values())
    if n_images == 0:
        return issues_text

    issues_layouts = defaultdict(list)
    n_layouts = 0
    with joblib_progress(description="OCRing", total=n_images):
        for issue, layout in Parallel(n_jobs=-2, return_as="generator")(
            delayed(
                lambda issue, image_path, output_path: (
                    issue,
                    ocr_image(image_path, output_path),
                )
            )(issue, image_path, output_path / issue / "ocr" / f"{i + 1}.csv")
            for issue, images in issues_images.items()
            for i, image_path in enumerate(images)
        ):
            issues_layouts[issue].append(layout)
            n_layouts += 1

    # Initialize GPT2 to compute perplexity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device", device)
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilgpt2", clean_up_tokenization_spaces=True
    )
    tokenizer.model_max_length = 2**16
    with tqdm(total=n_layouts, desc="Clustering OCR outputs") as pbar:
        for issue, layouts in issues_layouts.items():
            try:
                n_pages = 0
                df = []
                text = ""
                for scan, layout in enumerate(layouts, 1):
                    layout["scan"] = scan
                    if len(layout) < 4:
                        continue

                    # Find pages
                    image_width = layout.x_center.max() - layout.x_center.min()
                    image_height = layout.y_center.max() - layout.y_center.min()
                    if image_width > image_height:  # 2 pages
                        layout["page"] = find_pages(layout)
                    else:
                        layout["page"] = 1

                    # Find columns
                    layout["column"] = 1
                    layout["description"] = layout.description.astype(str)
                    for page in layout.page.unique():
                        slice = layout.page == page
                        layout_page = layout[slice]
                        if len(layout_page) < 4:
                            continue
                        res = {}
                        for n_cols in [2, 3]:
                            kmeans = KMeans(n_clusters=n_cols, random_state=0)
                            cols = kmeans.fit_predict(
                                layout_page[["x_center_100x", "y_center"]]
                            )
                            cols = reordered_clusters(layout_page, cols)
                            t = "\n".join(
                                layout_page.groupby(cols)
                                .description.apply(" ".join)
                                .values
                            )
                            perplexity = compute_perplexity(t, model, tokenizer)
                            res[n_cols] = {
                                "cols": cols,
                                "perplexity": perplexity,
                            }
                        # The best number of columns is the one which results in the lowest GPT2 perplexity on the reordered text
                        best_n_cols = min(
                            res, key=lambda x: res[x]["perplexity"]
                        )
                        layout.loc[slice, "column"] = res[best_n_cols]["cols"]

                    layout_path = output_path / issue / "ocr" / f"{scan}.csv"
                    layout.to_csv(layout_path, index=False)
                    ax = plot_layout(layout)
                    ax.get_figure().savefig(layout_path.with_suffix(".jpeg"))
                    plt.close()

                    text += f"# Scan {scan}\n"
                    pbar.update()

                    layout["page"] = layout.page + n_pages
                    for page, layout_page in layout.groupby("page"):
                        text += f"## Page {page}\n"
                        for col, layout_col in layout_page.groupby("column"):
                            text += f"### Column {col}\n\n"
                            text += " ".join(layout_col.description.astype(str))
                            text += "\n\n\n"
                        text += "\n"
                    text += "\n"
                    df.append(layout)
                    n_pages += layout.page.nunique()
                df = pd.concat(df)
                issue_path = output_path / issue
                issue_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(issue_path / "ocr.csv", index=False)
                issues_text[issue] = text
                with open(issue_path / "ocr.txt", "w") as f:
                    f.write(text)
            except:
                print(
                    f"##### OCR and clustering failed for {issue}:\n{traceback.format_exc()}\n#####"
                )

    return issues_text
