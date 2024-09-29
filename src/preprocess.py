from glob import glob
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from PIL import Image

from src.utils import extract_images_from_pdf


def preprocess_image(image, output_path):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the largest contour which should be the document
    if len(contours) == 0:
        print(f"No contours found in the image")
        return

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle for the largest contour (this is the rotated bounding box)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    # Get rotation matrix to align the document
    center = (int(rect[0][0]), int(rect[0][1]))  # Center of the rectangle
    angle = 90 - rect[2]  # The angle of rotation

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the entire image
    rotated_img = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
    )

    # Crop the rotated image to the bounding box
    # Get coordinates of the rotated bounding box
    x, y, w, h = cv2.boundingRect(box)
    cropped_img = rotated_img[y : y + h, x : x + w]

    # Optionally, convert to grayscale after cropping
    grayscale_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Save the result
    image = Image.fromarray(grayscale_cropped)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def preprocess_pdfs(pdf_path, output_path, n_jobs=-1):
    pdf_paths = glob(pdf_path)
    with joblib_progress("Extracting images", total=len(pdf_paths)):
        images = sum(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(extract_images_from_pdf)(pdf_path)
                for pdf_path in pdf_paths
            ),
            [],
        )

    with joblib_progress("Preprocessing images", total=len(images)):
        Parallel(n_jobs=n_jobs)(
            delayed(preprocess_image)(image, Path(output_path) / f"{i}.jpeg")
            for i, image in enumerate(images)
        )

    return images
