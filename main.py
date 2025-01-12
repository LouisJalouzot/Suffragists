import json
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

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
response_schema_follow_up = deepcopy(response_schema)
response_schema["properties"]["IssueDate"] = {
    "type": "string",
    "description": "Publication date of this journal issue",
}
response_schema["required"].append("IssueDate")
prompt = "Here is a scan of a journal issue, each scan having 1 or 2 pages and each page can have from a 1 to 4 column layout. Extract the publication date of the issue. Also extract information about all upcoming political meetings. Make sure to include all the meetings from all the cities, some of them might be within and others in tables. Infer the full date and complete the location with the city when necessary. For each meeting, also extract the lists of speakers and hosts and additional information when relevant. Your answer should have the following JSON format:\n"
prompt += json.dumps(response_schema, indent=4)
prompt_follow_up = 'If the previous outputs are complete, answer with an empty "Meetings" list. Otherwise complete it. Your answer should have the following JSON format:\n'
prompt_follow_up += json.dumps(response_schema_follow_up, indent=4)


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)

    return file


def wait_for_file_active(file):
    """Waits for the given file to be active.

    The file uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    name = file.name
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
        time.sleep(10)
        file = genai.get_file(name)
    if file.state.name != "ACTIVE":
        raise Exception(f"File {file.name} failed to process")


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
    issues: list[str] | str,
    output_path: str = "results",
    temperature: float = 1,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 8192,
) -> None:
    if isinstance(issues, str):
        issues = [issues]
    output_path = Path(output_path)
    issues = [
        issue
        for issue in issues
        if not (output_path / issue / "meetings.csv").exists()
    ]
    genai.configure()
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    def process_issue(issue: str) -> None:
        try:
            issue_name = issue.rsplit("_", 1)[0]
            pdf_path = f"data/{issue_name}/{issue}.pdf"
            file = upload_to_gemini(pdf_path, mime_type="application/pdf")
            wait_for_file_active(file)
            # Save parameters before generation
            issue_path = output_path / issue
            issue_path.mkdir(parents=True, exist_ok=True)
            params = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
            with open(issue_path / "parameters.json", "w") as f:
                json.dump(params, f, indent=4)

            chat_session = model.start_chat(
                history=[
                    {"role": "user", "parts": [file]},
                ]
            )
            response = chat_session.send_message(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    **params,
                },
            ).text
            with open(issue_path / "response.txt", "w") as f:
                f.write(response)
            response = parse_gemini_response(response)
            for i in range(1, 10):
                response_follow_up = chat_session.send_message(
                    prompt_follow_up,
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema_follow_up,
                        **params,
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

    for _ in tqdm(
        Parallel(n_jobs=-2, prefer="threads", return_as="generator")(
            delayed(process_issue)(issue) for issue in issues
        ),
        total=len(issues),
        desc="Prompting Gemini",
    ):
        pass


if __name__ == "__main__":
    prompt_gemini(
        [
            "votes_for_wmn_1",
            "votes_for_wmn_26",
            "votes_for_wmn_39",
            "votes_for_wmn_54",
            "votes_for_wmn_97",
            "votes_for_wmn_98",
            "votes_for_wmn_99",
            "votes_for_wmn_100",
            "votes_for_wmn_101",
            "votes_for_wmn_102",
            "votes_for_wmn_103",
            "votes_for_wmn_108",
            "votes_for_wmn_112",
            "votes_for_wmn_120",
            "votes_for_wmn_140",
            "votes_for_wmn_165",
            "votes_for_wmn_192",
            "votes_for_wmn_217",
            "votes_for_wmn_223",
        ]
    )
