from pathlib import Path

from src.prompt_gemini import prompt_gemini

issues = sorted(
    [p.stem for p in Path("data/common_cause").glob("*.pdf")],
    key=lambda x: int(x.split("_")[-1]),
)
issues = [
    f"common_cause_{i}"
    for i in [
        1,
        3,
        4,
        5,
        11,
        15,
        29,
        59,
        100,
        115,
        130,
        132,
        159,
        165,
        184,
        198,
        200,
        224,
        252,
    ]
]
prompt_gemini(issues)
