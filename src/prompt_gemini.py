import json
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from src.ocr_and_cluster import ocr_and_cluster

response_schema = {
    "type": "object",
    "properties": {
        "IssueDate": {
            "type": "string",
            "description": "Publication date of this journal issue",
        },
        "Meetings": {
            "type": "array",
            "description": "Information about each meeting in the tables of meetings in this journal issue. Meetings are uniquely defined by a date and a location and should have a similar formatting to the ones just below the title of the table.",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {
                        "type": "string",
                        "description": "Date of the meeting",
                    },
                    "Location": {
                        "type": "string",
                        "description": "Location of the meeting",
                    },
                    "Raw": {
                        "type": "string",
                        "description": "Verbatim text snippet from the document describing this meeting",
                    },
                    "Speakers": {
                        "type": "array",
                        "description": "List of speakers at the meeting",
                        "items": {"type": "string"},
                    },
                    "Hosts": {
                        "type": "array",
                        "description": "List of hosts for the meeting",
                        "items": {"type": "string"},
                    },
                    "Description": {
                        "type": "string",
                        "description": "Extra details about the meeting",
                    },
                },
                "required": ["Date", "Location", "Raw"],
            },
        },
    },
    "required": ["IssueDate"],
}


def prompt_gemini(issues: list[str], output_path: str = "results") -> None:
    genai.configure()
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    issues_text = ocr_and_cluster(issues, output_path=output_path)
    output_path = Path(output_path)

    def process_issue(issue: str, text: str) -> None:
        try:
            issue_path = output_path / issue
            if (issue_path / "meetings.csv").exists():
                return
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(
                text,
                generation_config={
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            ).text
            try:
                json.loads(response)
            except:
                response += chat_session.send_message(
                    "Complete your previous answer from where you stopped without repeating what you said.",
                    generation_config={"max_output_tokens": 8192},
                ).text
            issue_path.mkdir(parents=True, exist_ok=True)
            with open(issue_path / "response.txt", "w") as f:
                f.write(response)
            response = json.loads(response)
            issue_path.mkdir(parents=True, exist_ok=True)
            with open(issue_path / "response.json", "w") as f:
                f.write(json.dumps(response, indent=4))
            df = pd.DataFrame(response["Meetings"])
            df["Issue date"] = response.get("IssueDate", None)
            df.to_csv(issue_path / "meetings.csv", index=False)
        except Exception as e:
            print(f"Gemini prompt script failed for {issue}: {e}")

    with joblib_progress(
        description="Prompting Gemini", total=len(issues_text)
    ):
        Parallel(n_jobs=-2, prefer="threads")(
            delayed(process_issue)(issue, text)
            for issue, text in issues_text.items()
        )
