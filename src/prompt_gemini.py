import json
import os
import traceback
from pathlib import Path
from typing import List

import google.generativeai as genai
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from src.ocr import ocr_and_cluster

prompt = """
Here is the OCR output of an issue of the journal 'The Common Cause'. There is a table 'Forthcoming Meetings', please extract as much data about it as you can. Beware that the data may be scattered. Also extract the date of the issue.
Output your results in JSON format.
OCR output:
"""
# Each entry/meeting should have a date, a time and a location. Then they can also have a description, a host and some speakers.
# Be aware that data may be misaligned. In particular information about time, host and speaker might come in separate chunks after the others but still in the right order.
# """


def prompt_gemini(issues: List[str], output_path: str = "results") -> None:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
    )

    issues_text = ocr_and_cluster(issues, output_path=output_path)
    output_path = Path(output_path)

    def process_issue(issue: str, text: str) -> None:
        issue_path = output_path / issue / "gemini"
        if issue_path.exists():
            return
        try:
            chat_session = model.start_chat(history=[])
            message = prompt + text
            response = chat_session.send_message(message).text
            issue_path.mkdir(parents=True, exist_ok=True)
            with open(issue_path / "response.txt", "w") as f:
                f.write(response)
            if response:
                response = json.loads(response)
                json_path = issue_path / "response.json"
                with open(json_path, "w") as f:
                    json.dump(response, f, indent=4)
                meetings = response.get("Forthcoming Meetings", None)
                if meetings:
                    df = pd.DataFrame(meetings)
                    df["Issue date"] = response.get("Date", None)
                    df.to_csv(issue_path.parent / f"{issue}.csv", index=False)
        except:
            with open(issue_path.parent / "gemini_error.log", "w") as f:
                f.write(traceback.format_exc())

    with joblib_progress(
        description="Prompting Gemini", total=len(issues_text)
    ):
        Parallel(n_jobs=-2)(
            delayed(process_issue)(issue, text)
            for issue, text in issues_text.items()
        )
