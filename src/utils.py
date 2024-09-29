import io

import pymupdf
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

progress = Progress(
    SpinnerColumn(),
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
)


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
