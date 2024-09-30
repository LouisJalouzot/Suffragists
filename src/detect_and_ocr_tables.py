import os
from pathlib import Path
from typing import List

import cv2
import layoutparser as lp
import numpy as np
from tqdm.auto import tqdm

HOME = os.environ["HOME"]


def detect_and_ocr_tables(
    image_paths: List[Path],
    output_path: Path,
    save_box: Path = None,
):
    model = lp.Detectron2LayoutModel(
        config_path="lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
    )
    ocr_agent = lp.GCVAgent.with_credential(
        f"{HOME}/.config/gcloud/application_default_credentials.json",
        languages=["en"],
    )
    # ocr_agent = lp.TesseractAgent(languages="eng")

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        layout = model.detect(image)

        if save_box is not None:
            for box in layout:
                image = np.array(
                    lp.draw_box(
                        image, [box], box_width=int(box.score * 40 + 10)
                    )
                )
                cv2.putText(
                    image,
                    f"{box.score:.3g}",
                    (int(box.block.x_1), int(box.block.y_1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 0),
                    5,
                )
            image_box_path = save_box / image_path.name
            image_box_path.unlink(missing_ok=True)
            image_box_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_box_path), image[::5, ::5])

        for i, block in enumerate(layout):
            segmented_image = block.pad(
                left=50, right=50, top=50, bottom=50
            ).crop_image(image)
            # text = ocr_agent.detect(segmented_image)
            ocr = ocr_agent.detect(segmented_image, return_response=True)
            text_layout = ocr_agent.gather_full_text_annotation(
                ocr, agg_level=lp.GCVFeatureType.PARA
            )
            text = "\n".join(text_layout.to_dataframe().text.tolist())
            text_path = output_path / f"{image_path.stem}_{i}.txt"
            text_path.unlink(missing_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            with open(text_path, "w") as f:
                f.write(text)
