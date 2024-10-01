from pathlib import Path

from src.prompt_gemini import prompt_gemini

issues = sorted([p.stem for p in Path("data/common_cause").glob("*.pdf")])
for i in range(0, len(issues), 10):
    prompt_gemini(issues[i : i + 10])
