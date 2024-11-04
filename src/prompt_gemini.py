import json
import traceback
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
            "description": "Information about all the meetings in the tables of meetings in this journal issue, which can be scattered into multiple chunks. The meetings should have similar formatting.",
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
    if isinstance(issues, str):
        issues = [issues]
    genai.configure()
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

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
            with open(issue_path / "response.txt", "w") as f:
                f.write(response)
            try:
                response = json.loads(response)
            except:
                response_2 = chat_session.send_message(
                    "Write the end of the previous output.",
                    generation_config={
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                ).text
                with open(issue_path / "response_2.txt", "w") as f:
                    f.write(response_2)
                response = response.rsplit(", {", maxsplit=1)[0] + "]}"
                response = json.loads(response)
                response_2 = json.loads(response_2)
                response["Meetings"].extend(response_2["Meetings"])
            issue_path.mkdir(parents=True, exist_ok=True)
            with open(issue_path / "response.json", "w") as f:
                f.write(json.dumps(response, indent=4))
            df = pd.DataFrame(response["Meetings"])
            df["Issue date"] = response.get("IssueDate", None)
            df.to_csv(issue_path / "meetings.csv", index=False)
        except Exception as e:
            print(f"Gemini prompt script failed for {issue}: {e}")
            traceback.print_exc()

    with joblib_progress(
        description="Prompting Gemini", total=len(issues_text)
    ):
        Parallel(n_jobs=-2, prefer="threads")(
            delayed(process_issue)(issue, text)
            for issue, text in issues_text.items()
        )
