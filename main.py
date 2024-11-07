from pathlib import Path

import numpy as np

from src.prompt_gemini import prompt_gemini

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
batches["common_cause"] = list(np.random.choice(issues["common_cause"], 40))
batches["common_cause"] = set(batches["common_cause"] + cc_sel)
batches["suffragette"] = np.random.choice(issues["suffragette"], 10)
batches["votes_for_wmn"] = list(np.random.choice(issues["votes_for_wmn"], 7))
batches["votes_for_wmn"] = set(batches["votes_for_wmn"] + [100, 120, 140])

for k, v in batches.items():
    print(f"{k}: {v}")
    batches[k] = [k + "_" + str(i) for i in v]

prompt_gemini(sum(batches.values(), []))
