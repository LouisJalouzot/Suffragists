from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from PIL import Image, Jpeg2KImagePlugin

from src.utils import extract_images_from_pdf


def preprocess_image(
    image: Jpeg2KImagePlugin.Jpeg2KImageFile,
    output_path: Path,
    upscale_factor: float = 2,
) -> None:
    try:
        gray = image.convert("L")
        gray = np.array(gray)

        # Invert the grayscale image
        inverted_gray = 255 - gray

        # Apply threshold to create binary image
        blur = cv2.GaussianBlur(inverted_gray, (5, 5), 0)
        thresh = cv2.threshold(
            blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

        # Upscale the image
        upscaled_img = cv2.resize(
            cropped_img,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        # Save the result
        output_img = Image.fromarray(upscaled_img)
        output_path.unlink(missing_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path)

    except Exception as e:
        print(f"Error processing {output_path}: {e}")


def preprocess_pdfs(pdf_path: str, output_path: str, n_jobs: int = -1) -> None:
    pdf_paths = sorted(glob(pdf_path))
    images_dict = defaultdict(list)
    n_images = 0
    with joblib_progress("Extracting images", total=len(pdf_paths)):
        for issue, images in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(lambda p: (Path(p).stem, extract_images_from_pdf(p)))(
                pdf_path
            )
            for pdf_path in pdf_paths
        ):
            images_dict[issue].extend(images)
            n_images += len(images)

    with joblib_progress("Preprocessing images", total=n_images):
        Parallel(n_jobs=n_jobs)(
            delayed(preprocess_image)(
                image, Path(output_path) / f"{issue}_{i+1}.jpeg"
            )
            for issue, images in images_dict.items()
            for i, image in enumerate(images)
        )
