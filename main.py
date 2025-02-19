import argparse
import numpy as np

from src.prompt_gemini import prompt_gemini

# Add argument parser
parser = argparse.ArgumentParser(
    description="Process journal issues with Gemini"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="gemini-2.0-flash-exp",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1,
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
)
parser.add_argument(
    "--top_k",
    type=int,
    default=64,
)
args = parser.parse_args()

np.random.seed(0)

prompt_gemini(
    [
        f"votes_for_wmn_{i}"
        for i in [
            26,
            39,
            54,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            120,
            140,
            164,
            192,
            223,
        ]
    ],
    **vars(args),
)
