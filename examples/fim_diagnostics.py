"""Compute scale-invariant descriptors for a small Fisher information matrix."""

import numpy as np

from iaapi.evaluation.fim_descriptors import compute_fim_descriptors


def main() -> None:
    fim = np.diag([100.0, 10.0, 0.1, 0.001])
    result = compute_fim_descriptors(fim)
    print("features:", dict(zip(result["names"], result["descriptors"])))
    print("quality flags:", result["quality_flags"])


if __name__ == "__main__":
    main()
