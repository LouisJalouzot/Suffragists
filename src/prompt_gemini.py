import json
import os
import traceback
from pathlib import Path
from typing import List

import google.generativeai as genai
import pandas as pd
from tqdm.auto import tqdm

from src.detect_and_ocr_tables import detect_and_ocr_tables

prompt = """
I'm working with pdf scans of old journals about women voting rights in the UK.
Here it's the journal 'The Common Cause'. There are tables with information about upcoming political meetings and I want to retrieve them.
Here is the result from some OCR of an automatically and possibly mistakenly detected table.
If there is indeed information about upcoming meetings in this data please extract as much information as you can and put it in JSON format. Otherwise please output nothing.
Each entry/meeting should have a date, a time and a location. Then they can also have a description, a host and some speakers.
Be aware that data may be misaligned. In particular information about time, host and speaker might come in separate chunks after the others but still in the right order.
OCR output:
"""


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
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

    issues_text = detect_and_ocr_tables(issues, output_path=output_path)
    output_path = Path(output_path)

    for issue, text in tqdm(list(issues_text.items()), desc="Prompting Gemini"):
        issue_path = output_path / issue / "gemini"
        if issue_path.exists():
            continue
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
                meetings = response.get("meetings", None)
                if meetings:
                    pd.DataFrame(meetings).to_csv(
                        issue_path.parent / f"{issue}.csv", index=False
                    )
        except:
            with open(issue_path.parent / "gemini_error.log", "w") as f:
                f.write(traceback.format_exc())
