import os
import traceback
from pathlib import Path
from typing import Dict, List

import layoutparser as lp
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.preprocess import preprocess_pdfs

HOME = os.environ["HOME"]


def detect_and_ocr_tables(
    issues: List[str],
    output_path: str = "results",
) -> Dict[str, str]:
    # model = lp.Detectron2LayoutModel(
    #     config_path="lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
    # )
    ocr_agent = lp.GCVAgent.with_credential(
        f"{HOME}/.config/gcloud/application_default_credentials.json",
        languages=["en"],
    )

    issues_images = preprocess_pdfs(issues, output_path=output_path)
    output_path = Path(output_path)
    issues_text = {}

    for issue, images in tqdm(
        list(issues_images.items()), desc="Detecting and OCRing"
    ):
        issue_path = output_path / issue / "ocr.txt"
        if issue_path.exists():
            with open(issue_path) as f:
                issues_text[issue] = f.read()
            continue
        try:
            issue_text = []
            for i, image in enumerate(images):
                # layout = model.detect(image)

                # if layout:
                #     for box in layout:
                #         image = lp.draw_box(
                #             image, [box], box_width=int(box.score * 40 + 10)
                #         )
                #     image_box_path = (
                #         output_path / issue / "ocr" / f"box_{i+1}.jpeg"
                #     )
                #     image_box_path.parent.mkdir(parents=True, exist_ok=True)
                #     subsampled_image = image.resize(
                #         (image.width // 3, image.height // 3), Image.NEAREST
                #     )
                #     subsampled_image.save(image_box_path)

                #     w, h = image.width, image.height
                #     w_pad = int(0.05 * w)
                #     h_pad = int(0.05 * h)
                #     for i, block in enumerate(layout):
                #         segmented_image = block.pad(
                #             left=w_pad, right=w_pad, top=h_pad, bottom=h_pad
                #         ).crop_image(np.array(image))
                #         ocr = ocr_agent.detect(
                #             segmented_image, return_response=True
                #         )
                #         text_layout = ocr_agent.gather_full_text_annotation(
                #             ocr, agg_level=lp.GCVFeatureType.PARA
                #         )
                #         text = "\n".join(
                #             text_layout.to_dataframe().text.tolist()
                #         )
                #         issue_text.append(text)
                ocr = ocr_agent.detect(np.array(image), return_response=True)
                text_layout = ocr_agent.gather_full_text_annotation(
                    ocr, agg_level=lp.GCVFeatureType.PARA
                )
                text = "\n".join(text_layout.to_dataframe().text.tolist())
                issue_text.append(text)

            issues_text[issue] = "\n".join(issue_text)
            with open(issue_path, "w") as f:
                f.write(issues_text[issue])
        except:
            with open(issue_path.parent / "ocr_error.log", "w") as f:
                f.write(traceback.format_exc())

    return issues_text
