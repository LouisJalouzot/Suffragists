import io

import pandas as pd
import pymupdf
from google.cloud import vision_v1
from PIL import Image


def extract_images_from_pdf(pdf_path):
    images = []
    with pymupdf.open(pdf_path) as pdf_document:
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert image bytes to a Pillow image
                image = Image.open(io.BytesIO(image_bytes))

                # If the image is a JPEG 2000 image, convert it to a Pillow image
                if image_ext == "jp2":
                    image = image.convert("RGB")

                images.append(image)

    return images


def perform_ocr(image_path):
    client = vision_v1.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision_v1.types.Image(content=content)
    response = client.text_detection(image=image)

    df = []
    for annot in response.text_annotations[1:]:
        row = [annot.description]
        x_center, y_center = 0, 0
        for vertex in annot.bounding_poly.vertices:
            row.append(vertex.x)
            row.append(vertex.y)
            x_center += vertex.x
            y_center += vertex.y
        x_center /= len(annot.bounding_poly.vertices)
        y_center /= len(annot.bounding_poly.vertices)
        row.append(x_center)
        row.append(y_center)
        df.append(row)

    df = pd.DataFrame(
        df,
        columns=[
            "description",
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "x4",
            "y4",
            "x_center",
            "y_center",
        ],
    )
    return df
