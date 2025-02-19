import json
import re
import traceback
from pathlib import Path
import glob
import time

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
Polling days should not be included and consider only upcoming meetings, those which will happen after the publication date.
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
        last_match = list(re.finditer(r",\s*{", response))[-1]
        if last_match:
            split_index = last_match.start()
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
    top_k: int = 64,
) -> None:
    if isinstance(issues, str):
        issues = [issues]
    genai.configure()
    model = genai.GenerativeModel(model_name=model_name)

    issues_text = ocr_and_cluster(issues, output_path=output_path)
    output_path = Path(output_path)

    def process_scan(issue: str, scan_id: int, text: str) -> dict:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                scan_path = output_path / issue / "gemini" / str(scan_id)
                scan_path.mkdir(parents=True, exist_ok=True)

                response_json_path = scan_path / "response.json"
                if response_json_path.exists():
                    with open(response_json_path, "r") as f:
                        response = json.load(f)
                    return response

                # Remove existing files before continuing
                for pattern in [
                    "response*",
                    "message*",
                    "parameters.json",
                ]:
                    files = glob.glob(str(scan_path / pattern))
                    for file_path in files:
                        Path(file_path).unlink(missing_ok=True)

                # Save parameters before generation
                params = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
                with open(scan_path / "parameters.json", "w") as f:
                    json.dump(params, f, indent=4)

                chat_session = model.start_chat(history=[])
                message = prompt + "<TEXT>\n" + text + "\n</TEXT>"
                with open(scan_path / "message.txt", "w") as f:
                    f.write(message)
                response = chat_session.send_message(
                    message,
                    generation_config={
                        "max_output_tokens": 8192,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                ).text
                with open(scan_path / "response.txt", "w") as f:
                    f.write(response)
                response = parse_gemini_response(response)
                if len(response["Meetings"]) > 0:
                    # Maybe Gemini hit the 8192 token limit, continue prompting
                    for i in range(1, 10):
                        response_follow_up = chat_session.send_message(
                            prompt_follow_up,
                            generation_config={
                                "max_output_tokens": 8192,
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                                "response_mime_type": "application/json",
                                "response_schema": response_schema,
                            },
                        ).text
                        with open(
                            scan_path / f"response_follow_up_{i}.txt", "w"
                        ) as f:
                            f.write(response_follow_up)
                        response_follow_up = parse_gemini_response(
                            response_follow_up
                        )
                        if not response_follow_up["Meetings"]:
                            break
                        response["Meetings"].extend(
                            response_follow_up["Meetings"]
                        )

                with open(scan_path / "response.json", "w") as f:
                    f.write(json.dumps(response, indent=4))
                return response

            except Exception as e:
                if (
                    "resource" in str(e).lower()
                    and "exhausted" in str(e).lower()
                ):  # Rate limit error
                    print(f"Rate limit exceeded. Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    raise Exception(
                        f"##### Gemini prompt script failed for {issue} scan {scan_id}:\n{traceback.format_exc()}\n#####"
                    )

        raise Exception("Max retries exceeded for {issue} scan {scan_id}")

    def get_most_common(lst: list):
        valid_dates = [date for date in lst if date and date != ""]
        if not valid_dates:
            return None  # or a default value if appropriate
        return max(set(valid_dates), key=valid_dates.count)

    def process_issue(issue: str, scans_text: dict) -> None:
        try:
            issue_path = output_path / issue
            if (issue_path / "meetings.csv").exists():
                return

            all_meetings = []
            issue_dates = []
            for scan_id, text in scans_text.items():
                response = process_scan(issue, scan_id, text)
                all_meetings.extend(response["Meetings"])
                issue_dates.append(response.get("IssueDate", None))

            # Save combined meetings
            most_common_date = get_most_common(issue_dates)
            response = {"Meetings": all_meetings, "IssueDate": most_common_date}
            with open(issue_path / "response.json", "w") as f:
                f.write(json.dumps(response, indent=4))
            df = pd.DataFrame(response["Meetings"])
            df["Issue date"] = most_common_date
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
