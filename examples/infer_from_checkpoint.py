"""Minimal checkpoint-based inference example.

Replace the paths and observations with values for your PEtab problem.
"""

import argparse

import numpy as np

from iaapi.api import InferenceModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("petab_yaml")
    args = parser.parse_args()

    model = InferenceModel.from_checkpoint(args.checkpoint)
    result = model.infer(
        petab_problem=args.petab_yaml,
        observations={
            "time": np.array([0.0, 1.0, 2.0]),
            "values": np.array([[1.0, 0.8, 0.6]]),
            "mask": np.ones((1, 3)),
        },
    )
    print(result.get_posterior_stats())


if __name__ == "__main__":
    main()
