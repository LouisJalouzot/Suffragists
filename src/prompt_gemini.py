import json
import re
import traceback
from pathlib import Path
import shutil
import os
import glob

import google.generativeai as genai
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from src.ocr_and_cluster import ocr_and_cluster

response_schema = {
    "type": "object",
    "properties": {
        "Meetings": {
            "type": "array",
            "description": "List of upcoming meetings",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {
                        "type": "string",
                        "description": "Date of the meeting",
                    },
                    "Location": {
                        "type": "string",
                        "description": "Meaningful location of the meeting, city first",
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
                    "Time": {
                        "type": "string",
                        "description": "Time of the meeting",
                    },
                    "Raw": {
                        "type": "string",
                        "description": "Verbatim text snippet from the document describing this meeting",
                    },
                },
                "required": ["Date", "Location", "Raw"],
            },
        },
    },
    "required": ["Meetings"],
}
prompt = """
<INSTRUCTIONS>
You are a data scientist working for a political journal.
Here is the text from a journal issue.
Your task is to extract the publication date of the issue and information about upcoming political meetings from a journal publication.
Make sure to include all the meetings from all the cities.
Some of them might be within the text and others in tables.
Infer the full date and try to complete the location with the city when necessary.
For each meeting, also extract the lists of speakers and hosts and additional information when relevant.
Put the meetings in order of appearance.
Your answer should have the following JSON format:
"""
prompt += json.dumps(response_schema, indent=4)
prompt += "\n</INSTRUCTIONS>\n"
prompt_follow_up = """
<INSTRUCTIONS>
If the previous outputs are complete, answer with an empty "Meetings" list.
Otherwise complete it.
Your answer should have the same JSON format
</INSTRUCTIONS>
"""


def parse_gemini_response(response: str) -> dict:
    try:
        response = json.loads(response)
    except:
        # Incomplete JSON
        match = re.search(r",\s*{", response)
        if match:
            split_index = match.start()
            response = response[:split_index] + "]}"
        else:
            raise Exception("Incomplete JSON but no match found in response")
        response = json.loads(response)

    return response


def prompt_gemini(
    issues: list[str],
    output_path: str = "results",
    model_name: str = "gemini-2.0-flash-exp",
    temperature: float = 1,
    top_p: float = 0.95,
) -> None:
    if isinstance(issues, str):
        issues = [issues]
    genai.configure()
    model = genai.GenerativeModel(model_name=model_name)

    issues_text = ocr_and_cluster(issues, output_path=output_path)
    output_path = Path(output_path)

    def process_issue(issue: str, text: str) -> None:
        try:
            issue_path = output_path / issue
            if (issue_path / "meetings.csv").exists():
                return

            # Remove existing files before continuing
            for pattern in [
                "response*",
                "message*",
                "parameters.json",
                "meetings.csv",
            ]:
                files = glob.glob(str(issue_path / pattern))
                for file_path in files:
                    Path(file_path).unlink(missing_ok=True)

            # Save parameters before generation
            issue_path.mkdir(parents=True, exist_ok=True)
            params = {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
            }
            with open(issue_path / "parameters.json", "w") as f:
                json.dump(params, f, indent=4)

            chat_session = model.start_chat(history=[])
            message = "<TEXT>\n" + text + "\n</TEXT>\n" + prompt
            with open(issue_path / "message.txt", "w") as f:
                f.write(message)
            response = chat_session.send_message(
                message,
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": temperature,
                    "top_p": top_p,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            ).text
            with open(issue_path / "response.txt", "w") as f:
                f.write(response)
            response = parse_gemini_response(response)
            for i in range(1, 10):
                response_follow_up = chat_session.send_message(
                    prompt_follow_up,
                    generation_config={
                        "max_output_tokens": 8192,
                        "temperature": temperature,
                        "top_p": top_p,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                ).text
                with open(issue_path / f"response_follow_up_{i}.txt", "w") as f:
                    f.write(response_follow_up)
                response_follow_up = parse_gemini_response(response_follow_up)
                if not response_follow_up["Meetings"]:
                    break
                response["Meetings"].extend(response_follow_up["Meetings"])

            issue_path.mkdir(parents=True, exist_ok=True)
            with open(issue_path / "response.json", "w") as f:
                f.write(json.dumps(response, indent=4))
            df = pd.DataFrame(response["Meetings"])
            df["Issue date"] = response.get("IssueDate", None)
            df.to_csv(issue_path / "meetings.csv", index=False)

        except:
            print(
                f"##### Gemini prompt script failed for {issue}:\n{traceback.format_exc()}\n#####"
            )

    with joblib_progress(
        description="Prompting Gemini", total=len(issues_text)
    ):
        Parallel(n_jobs=1, prefer="threads")(
            delayed(process_issue)(issue, text)
            for issue, text in issues_text.items()
        )
