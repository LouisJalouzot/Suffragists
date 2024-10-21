import json
import os
from pathlib import Path
from typing import List

import google.generativeai as genai
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from src.ocr_and_cluster import ocr_and_cluster

prompt = """
##### Beginning of OCR output ######

###OCR###

##### End of OCR output ######

Above is the OCR output of an issue of the journal 'The Common Cause'. Your response should be in JSON format.

Extract the date of the issue (response JSON key: "IssueDate").

There is a the table titled 'Forthcoming Meetings'. Because of ads, sometimes the table is split into multiple parts in the OCR output but the events are not scattered individually in the text. Only the first part has the title.
Extract the scans, pages, and columns on which it spans (response JSON key: "TableSpan" as a list of dictionaries with keys "Scan", "Page", and "Column").

Finally, extract as much information as you can from the table for each meeting. Format it into a list of dictionaries with the following keys:
    - "Date": date of the meeting
    - "Location": all the information about the location of the meeting
    - "Description": additional information about the meeting, if any
    - "Hosts": list of hosts of the meeting, if any
    - "Speakers": list of speakers for the meeting, if any
    - "Time": time of the meeting if specified
    - "Raw": all the extracted text corresponding to this meeting, unformatted
If a meeting is missing address, location, description, hosts, speakers, or time, skip the corresponding field.
Events are not necessarily presented in chronological order.
The date might not appear for each meeting, in this case forward fill it. This can also be the case for big cities and countries, however a specific location is bound to one meeting and should not be forward filled.
Correct punctuation and word breaks.

Example OCR snippet: "# Scan 1 ## Page 1 ### Column 1 THE COMMON CAUSE, JUNE 10, 1909"
Expected JSON for snippet:
```json
{
    "IssueDate": "...",
    "TableSpan": [
        {"Scan": ..., "Page": ..., "Column": ...},
        ...
    ],
    "Meetings": [
        {
            "Date": "...",
            "Location": "...",
            "Description": "...",
            "Hosts": ["..."],
            "Speakers": ["..."],
            "Time": "...",
            "Raw": "..."   
        },
        ...
    ],
}
```
"""
response_schema = {
    "type": "object",
    "properties": {
        "IssueDate": {"type": "string", "nullable": True},
        "TableSpan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Scan": {"type": "integer"},
                    "Page": {"type": "integer"},
                    "Column": {"type": "integer"},
                },
                "required": ["Scan", "Page", "Column"],
            },
        },
        "Meetings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {"type": "string"},
                    "Location": {"type": "string"},
                    "Description": {"type": "string", "nullable": True},
                    "Hosts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "nullable": True,
                    },
                    "Speakers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "nullable": True,
                    },
                    "Time": {"type": "string", "nullable": True},
                    "Raw": {"type": "string"},
                },
                "required": ["Date", "Location", "Raw"],
            },
        },
    },
    "required": ["IssueDate", "TableSpan", "Meetings"],
}


def prompt_gemini(issues: List[str], output_path: str = "results") -> None:
    genai.configure()
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    issues_text = ocr_and_cluster(issues, output_path=output_path)
    output_path = Path(output_path)

    def process_issue(issue: str, text: str) -> None:
        issue_path = output_path / issue
        if (issue_path / "meetings.csv").exists():
            return
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(
            prompt.replace("###OCR###", text),
            generation_config={
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        ).text
        with open(issue_path / "response.txt", "w") as f:
            f.write(response)
        response = json.loads(response)
        issue_path.mkdir(parents=True, exist_ok=True)
        with open(issue_path / "response.json", "w") as f:
            f.write(json.dumps(response, indent=4))
        df = pd.DataFrame(response["Meetings"])
        df["Issue date"] = response.get("IssueDate", None)
        df.to_csv(issue_path / "meetings.csv", index=False)

    with joblib_progress(
        description="Prompting Gemini", total=len(issues_text)
    ):
        Parallel(n_jobs=-2)(
            delayed(process_issue)(issue, text)
            for issue, text in issues_text.items()
        )
