import argparse
from pathlib import Path

import numpy as np

from src.prompt_gemini import prompt_gemini

# Add argument parser
parser = argparse.ArgumentParser(
    description="Process journal issues with Gemini"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.2,
    help="Temperature parameter for Gemini (default: 0.2)",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.6,
    help="Top p parameter for Gemini (default: 0.6)",
)
args = parser.parse_args()

np.random.seed(0)

cc_sel = [
    1,
    3,
    4,
    5,
    11,
    15,
    29,
    59,
    100,
    115,
    130,
    132,
    159,
    165,
    184,
    198,
    200,
    224,
    252,
]
issues = {
    journal: [
        int(p.stem.split("_")[-1])
        for p in Path(f"data/{journal}").glob("*.pdf")
    ]
    for journal in ["common_cause", "suffragette", "votes_for_wmn"]
}
issues["common_cause"] = [i for i in issues["common_cause"] if i < 258]
issues["suffragette"] = [i for i in issues["suffragette"] if i < 98]
issues["votes_for_wmn"] = [i for i in issues["votes_for_wmn"] if i < 244]

batches = {}
batches["common_cause"] = issues["common_cause"]
batches["suffragette"] = issues["suffragette"]
batches["votes_for_wmn"] = list(np.random.choice(issues["votes_for_wmn"], 7))
batches["votes_for_wmn"] = set(batches["votes_for_wmn"] + [100, 120, 140])

for k, v in batches.items():
    print(f"{k}: {v}")
    batches[k] = [k + "_" + str(i) for i in v]
    # for i in range(0, len(batches[k]), 20):
    #     prompt_gemini(
    #         batches[k][i : i + 20],
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #     )

    prompt_gemini(
        batches[k],
        temperature=args.temperature,
        top_p=args.top_p,
    )
