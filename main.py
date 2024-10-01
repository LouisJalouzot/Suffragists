from pathlib import Path

from src.prompt_gemini import prompt_gemini

prompt_gemini([p.stem for p in Path("data/common_cause").glob("*.pdf")])
