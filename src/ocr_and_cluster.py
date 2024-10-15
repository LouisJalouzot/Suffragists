import os
from collections import defaultdict
from pathlib import Path
from typing import List

import layoutparser as lp
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.cluster import KMeans

from src.preprocess import preprocess_pdfs

HOME = os.environ["HOME"]


def ocr_and_cluster_image(image, output_path):
    if output_path.exists():
        return pd.read_csv(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.array(image)
    ocr_agent = lp.GCVAgent.with_credential(
        f"{HOME}/.config/gcloud/application_default_credentials.json",
        languages=["en"],
    )
    ocr = ocr_agent.detect(image, return_response=True)
    layout = ocr_agent.gather_full_text_annotation(
        ocr, agg_level=lp.GCVFeatureType.WORD
    ).to_dataframe()

    if len(layout) == 0:
        return
    elif len(layout) < 4:
        layout[["page", "column"]] = 1
        return layout

    layout["x_center"] = layout.points.apply(lambda x: np.mean(x[::2]))
    layout["y_center"] = layout.points.apply(lambda x: np.mean(x[1::2]))
    # 4x x_center feature to help with clustering
    layout["x_center_4x"] = layout.x_center * 4

    kmeans = KMeans(n_clusters=2, random_state=0)
    if image.shape[0] < image.shape[1]:  # 2 pages
        layout["page"] = kmeans.fit_predict(layout[["x_center_4x", "y_center"]])
        # Renumber pages based on increasing mean x_center
        page_centers = layout.groupby("page")["x_center"].mean().sort_values()
        page_mapping = {
            old: new for new, (old, _) in enumerate(page_centers.items(), 1)
        }
        layout["page"] = layout["page"].map(page_mapping)
    else:
        layout["page"] = 1

    layout["column"] = 1
    for page in layout.page.unique():
        slice = layout.page == page
        kmeans.n_clusters = 2
        col_2 = kmeans.fit_predict(
            layout.loc[slice, ["x_center_4x", "y_center"]]
        )
        inertia_2 = kmeans.inertia_
        kmeans.n_clusters = 3
        col_3 = kmeans.fit_predict(
            layout.loc[slice, ["x_center_4x", "y_center"]]
        )
        inertia_3 = kmeans.inertia_
        if inertia_2 * 2 < inertia_3 * 3:
            col = col_2
        else:
            col = col_3
        # Renumber columns based on increasing mean x_center
        col_centers = (
            layout.loc[slice].groupby(col)["x_center"].mean().sort_values()
        )
        col_mapping = {
            old: new for new, (old, _) in enumerate(col_centers.items(), 1)
        }
        col = [col_mapping[c] for c in col]
        layout.loc[slice, "column"] = col

    layout.to_csv(output_path, index=False)

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
    ax.get_figure().savefig(output_path.with_suffix(".jpeg"))

    return layout


def ocr_and_cluster(issues: List[str], output_path: str = "results"):
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
    with joblib_progress(description="OCRing", total=n_images):
        for issue, layout in Parallel(n_jobs=-2, return_as="generator")(
            delayed(
                lambda issue, image, output_path: (
                    issue,
                    ocr_and_cluster_image(image, output_path),
                )
            )(issue, image, output_path / issue / "ocr" / f"{i + 1}.csv")
            for issue, images in issues_images.items()
            for i, image in enumerate(images)
        ):
            issues_layouts[issue].append(layout)

    for issue, layouts in issues_layouts.items():
        n_pages = 0
        df = []
        text = ""
        for scan, layout in enumerate(layouts):
            if layout is None:
                continue
            layout["scan"] = scan + 1
            text += f"# Scan {scan + 1}\n"
            layout["page"] = layout.page + n_pages
            for page, layout_page in layout.groupby("page"):
                text += f"## Page {page}\n"
                for col, layout_col in layout_page.groupby("column"):
                    text += f"### Column {col}\n\n"
                    text += " ".join(layout_col.text.astype(str))
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

    return issues_text
