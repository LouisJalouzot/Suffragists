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
        p.stem
        for p in Path(f"data/{journal}").glob("*.pdf")
        if journal != "common_cause" or int(p.stem.split("_")[-1] not in cc_sel)
    ]
    for journal in ["common_cause", "suffragette", "votes_for_wmn"]
}

batch_cc = np.random.choice(issues["common_cause"], 40)
batch_suffragettes = np.random.choice(issues["suffragette"], 10)
batch_vfw = np.random.choice(issues["suffragette"], 7)
batch_vfw = set(
    list(batch_vfw) + [f"votes_for_wmn_{i}" for i in [100, 120, 140]]
)

batch = list(batch_cc) + list(batch_suffragettes) + list(batch_vfw)
print(batch)
prompt_gemini(batch)
