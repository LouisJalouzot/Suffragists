from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from PIL import Image, Jpeg2KImagePlugin

from src.utils import extract_images_from_pdf


def preprocess_image(
    image: Jpeg2KImagePlugin.Jpeg2KImageFile,
) -> Image.Image:
    gray = image.convert("L")
    gray = np.array(gray)
    blur = cv2.GaussianBlur(gray, (49, 49), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    dilate = cv2.dilate(blur, kernel, iterations=2)
    thresh = cv2.threshold(dilate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print(f"No contours found in the image")
        return

    contour = max(contours, key=cv2.contourArea)
    # Get the minimum area rectangle for the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Get the rotation matrix
    center = tuple(map(int, rect[0]))
    angle = rect[2]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the entire image
    rotated_img = cv2.warpAffine(
        gray,
        rotation_matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC,
    )

    # Rotate the bounding box points
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]

    # Get the new bounding rectangle after rotation
    x, y, w, h = cv2.boundingRect(rotated_box)

    # Ensure the bounding rectangle is within the image
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    h_image, w_image = rotated_img.shape
    if x + w > w_image:
        w = w_image - x
    if y + h > h_image:
        h = h_image - y

    # Crop the rotated image
    cropped_img = rotated_img[y : y + h, x : x + w]

    return Image.fromarray(cropped_img)


def preprocess_pdfs(
    issues: list[str], output_path: str = "results", n_jobs: int = -2
) -> dict[str, list[Image.Image]]:
    pdf_paths = {}
    output_path = Path(output_path)
    issues_images = defaultdict(list)
    for issue in issues:
        issue_path = output_path / issue / "preprocessed"
        if issue_path.exists():
            for image_path in sorted(issue_path.glob("*.jpeg")):
                issues_images[issue].append(image_path)
            continue
        else:
            issue_name = issue.rsplit("_", 1)[0]
            pdf_paths[issue] = f"data/{issue_name}/{issue}.pdf"

    n_paths = len(pdf_paths)
    if n_paths == 0:
        return issues_images

    n_images = 0
    issues_images_to_preprocess = {}
    with joblib_progress("Extracting images", total=n_paths):
        for issue, images in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(
                lambda issue: (issue, extract_images_from_pdf(pdf_paths[issue]))
            )(issue)
            for issue in pdf_paths
        ):
            issues_images_to_preprocess[issue] = images
            n_images += len(images)

    with joblib_progress("Preprocessing images", total=n_images):
        for issue, i, preprocessed_image in Parallel(
            n_jobs=n_jobs, return_as="generator"
        )(
            delayed(
                lambda issue, i, image: (issue, i, preprocess_image(image))
            )(issue, i, image)
            for issue, images in issues_images_to_preprocess.items()
            for i, image in enumerate(images)
        ):
            issue_path = output_path / issue / "preprocessed"
            issue_path.mkdir(parents=True, exist_ok=True)
            image_path = issue_path / f"{i+1}.jpeg"
            preprocessed_image.save(image_path)
            issues_images[issue].append(image_path)

    return issues_images
