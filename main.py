from pathlib import Path

from src.ocr_and_cluster import ocr_and_cluster
from src.prompt_gemini import prompt_gemini

issues = sorted(
    [p.stem for p in Path("data/common_cause").glob("*.pdf")],
    key=lambda x: int(x.split("_")[-1]),
)
issues = [
    f"common_cause_{i}" for i in [1, 3, 4, 5, 15, 11, 29, 159, 165, 198, 200]
]
prompt_gemini(issues)
