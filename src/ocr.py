import os
from collections import defaultdict
from pathlib import Path
from typing import List

import layoutparser as lp
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.cluster import KMeans

from src.preprocess import preprocess_pdfs

HOME = os.environ["HOME"]


def ocr_image(image, output_path):
    if output_path.parent.with_suffix(".txt").exists():
        with open(output_path.parent.with_suffix(".txt"), "r") as f:
            return f.read()
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
    layout["x_center"] = layout.points.apply(lambda x: np.mean(x[::2]))
    layout["y_center"] = layout.points.apply(lambda x: np.mean(x[1::2]))
    # 3x x_center feature to help with clustering
    layout["x_center_3x"] = layout.x_center * 3

    kmeans = KMeans(n_clusters=2, random_state=0)
    if image.shape[0] < image.shape[1]:  # 2 pages
        layout["page"] = kmeans.fit_predict(layout[["x_center_3x", "y_center"]])
    else:
        layout["page"] = 0

    for page in layout.page.unique():
        slice = layout.page == page
        col = (
            kmeans.fit_predict(layout.loc[slice, ["x_center_3x", "y_center"]])
            + page * 2
        )
        layout.loc[slice, "column"] = col
    layout.to_csv(output_path.with_suffix(".csv"), index=False)

    ax = sns.scatterplot(
        data=layout, x="x_center", y="y_center", hue="column", palette="tab10"
    )
    ax.legend_.remove()
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.figure.savefig(output_path.with_suffix(".jpeg"))

    text = layout.groupby("column").text.apply(" ".join)
    text = "\n\n\n".join(text)
    with open(output_path.parent.with_suffix(".txt"), "w") as f:
        f.write(text)

    return text


def ocr_and_cluster(issues: List[str], output_path: str = "results"):
    issues_images = preprocess_pdfs(issues, output_path=output_path)
    output_path = Path(output_path)
    issues_text = defaultdict(lambda: "")
    for issue in list(issues_images.keys()):
        issue_path = output_path / issue / "ocr.txt"
        if issue_path.exists():
            with open(issue_path, "r") as f:
                issues_text[issue] = f.read()
            issues_images.pop(issue)

    n_images = sum(len(images) for images in issues_images.values())
    if n_images == 0:
        return issues_text

    with joblib_progress(description="OCRing", total=n_images):
        for issue, text in Parallel(n_jobs=-2, return_as="generator")(
            delayed(
                lambda issue, image, output_path: (
                    issue,
                    ocr_image(image, output_path),
                )
            )(issue, image, output_path / issue / "ocr" / str(i + 1))
            for issue, images in issues_images.items()
            for i, image in enumerate(images)
        ):
            issues_text[issue] += text + "\n\n\n\n\n"

    for issue in issues_images:
        (output_path / issue).mkdir(parents=True, exist_ok=True)
        with open(output_path / issue / "ocr.txt", "w") as f:
            f.write(issues_text[issue])

    return issues_text