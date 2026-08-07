"""
Fisher Information Matrix (FIM) Computation Module.

This module computes the Fisher Information Matrix and its eigen-decomposition,
which provides information about parameter identifiability and subspace structure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class FIMComputer:
    """
    Compute Fisher Information Matrix and eigen-decomposition.

    The FIM quantifies how much information the data contains about each
    parameter. Its eigen-decomposition reveals the parameter subspace structure:
    - Large eigenvalues: well-constrained (identifiable) directions
    - Small eigenvalues: poorly constrained (sloppy) directions
    """

    def __init__(self, noise_level: float = 0.1):
        """
        Initialize FIM computer.

        Args:
            noise_level: Default standard deviation of measurement noise.
        """
        self.noise_level = noise_level

    def compute_fim(
        self,
        sensitivities: np.ndarray,
        sigma: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Fisher Information Matrix from sensitivities.

        For each observable k with noise sigma_k:
        F_{ij} = sum_t sum_k (1/sigma_k^2) * (ds_k/dtheta_i)(t) * (ds_k/dtheta_j)(t)

        Args:
            sensitivities: Sensitivities shaped (n_obs, n_time, n_param) or a
                flattened Jacobian shaped (n_data, n_param).
            sigma: Observation noise. Supported shapes: scalar, (n_obs,),
                (n_obs, n_time), or (n_data,) for flattened Jacobians.

        Returns:
            FIM matrix (n_param, n_param).
        """
        sensitivities = np.asarray(sensitivities, dtype=float)
        if sensitivities.ndim == 3:
            n_obs, n_time, n_param = sensitivities.shape
            jacobian = sensitivities.reshape(n_obs * n_time, n_param)
            weights = self._weights_for_3d_sigma(sigma, n_obs, n_time)
        elif sensitivities.ndim == 2:
            n_data, n_param = sensitivities.shape
            jacobian = sensitivities
            weights = self._weights_for_flat_sigma(sigma, n_data)
        else:
            raise ValueError(
                "sensitivities must have shape (n_obs, n_time, n_param) "
                f"or (n_data, n_param), got {sensitivities.shape}"
            )

        jacobian = np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
        fim = jacobian.T @ (weights[:, None] * jacobian)
        fim = 0.5 * (fim + fim.T)
        return fim

    def _weights_for_3d_sigma(
        self,
        sigma: Optional[np.ndarray],
        n_obs: int,
        n_time: int,
    ) -> np.ndarray:
        """Return flattened weights for 3D sensitivity tensors."""
        if sigma is None:
            return np.full(n_obs * n_time, 1.0 / (self.noise_level**2), dtype=float)

        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_full = np.full((n_obs, n_time), float(sigma_arr), dtype=float)
        elif sigma_arr.shape == (n_obs,):
            sigma_full = np.repeat(sigma_arr[:, None], n_time, axis=1)
        elif sigma_arr.shape == (n_obs, n_time):
            sigma_full = sigma_arr
        elif sigma_arr.shape == (n_obs * n_time,):
            sigma_full = sigma_arr.reshape(n_obs, n_time)
        else:
            raise ValueError(
                f"sigma shape {sigma_arr.shape} is incompatible with sensitivities "
                f"shape ({n_obs}, {n_time}, n_param)"
            )
        return self._sigma_to_weights(sigma_full.reshape(-1))

    def _weights_for_flat_sigma(self, sigma: Optional[np.ndarray], n_data: int) -> np.ndarray:
        """Return flattened weights for 2D Jacobians."""
        if sigma is None:
            return np.full(n_data, 1.0 / (self.noise_level**2), dtype=float)

        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_flat = np.full(n_data, float(sigma_arr), dtype=float)
        else:
            sigma_flat = sigma_arr.reshape(-1)
            if sigma_flat.shape != (n_data,):
                raise ValueError(
                    f"sigma shape {sigma_arr.shape} is incompatible with flattened "
                    f"sensitivities shape ({n_data}, n_param)"
                )
        return self._sigma_to_weights(sigma_flat)

    def _sigma_to_weights(self, sigma: np.ndarray) -> np.ndarray:
        """Convert sigma values to inverse-variance weights."""
        sigma = np.asarray(sigma, dtype=float)
        if np.any(sigma <= 0) or np.any(~np.isfinite(sigma)):
            raise ValueError("sigma values must be positive and finite")
        return 1.0 / (sigma**2)

    def eigendecompose(
        self,
        fim: np.ndarray,
        relative_threshold: float = 1e-6,
        negative_tolerance: float = 1e-10,
        rank_method: str = "spectral_gap",
        spectral_gap_min_depth: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Perform eigen-decomposition of FIM.

        Args:
            fim: Fisher Information Matrix.
            relative_threshold: Eigenvalues above this fraction of the largest
                eigenvalue are counted as identifiable/effective-rank directions.
                Used only when ``rank_method == "threshold"``.
            negative_tolerance: Tiny negative numerical eigenvalues above
                ``-negative_tolerance`` are clipped to zero.
            rank_method: Method to determine the effective rank separating the
                identifiable subspace from the sloppy subspace. One of:

                * ``"spectral_gap"`` (default): find the largest log10 gap between
                  adjacent sorted eigenvalues; if that gap spans at least
                  ``spectral_gap_min_depth`` orders of magnitude, split there.
                  Otherwise the spectrum is treated as full-rank. This mirrors
                  the gap-based criterion used in profile-likelihood analysis
                  and is robust to absolute scaling of the FIM.
                * ``"threshold"``: legacy relative-threshold rule (kept for
                  backward compatibility). Tends to over-report rank because a
                  1e-6 relative cutoff rarely flags borderline directions.
                * ``"entropy"``: round the entropy effective rank.
            spectral_gap_min_depth: Minimum gap depth (in log10 orders of
                magnitude) required to declare a sloppy subspace when using the
                ``"spectral_gap"`` method.

        Returns:
            Dictionary containing eigenvalues/eigenvectors and subspace metadata.
        """
        fim = np.asarray(fim, dtype=float)
        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            raise ValueError(f"fim must be a square matrix, got shape {fim.shape}")

        fim = np.nan_to_num(fim, nan=0.0, posinf=0.0, neginf=0.0)
        fim = 0.5 * (fim + fim.T)
        eigenvalues, eigenvectors = np.linalg.eigh(fim)

        eigenvalues = np.where(
            (eigenvalues < 0) & (eigenvalues >= -negative_tolerance),
            0.0,
            eigenvalues,
        )
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        effective_rank, spectral_gap_depth = self._compute_effective_rank_dispatch(
            eigenvalues,
            rank_method=rank_method,
            relative_threshold=relative_threshold,
            spectral_gap_min_depth=spectral_gap_min_depth,
        )
        entropy_effective_rank = self._compute_entropy_effective_rank(eigenvalues)
        identifiable_subspace = eigenvectors[:, :effective_rank]
        sloppy_subspace = eigenvectors[:, effective_rank:]

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "effective_rank": effective_rank,
            "entropy_effective_rank": entropy_effective_rank,
            "rank_method": rank_method,
            "spectral_gap_depth": spectral_gap_depth,
            "identifiable_subspace": identifiable_subspace,
            "sloppy_subspace": sloppy_subspace,
            "condition_number": self.compute_condition_number_from_eigenvalues(eigenvalues),
        }

    def _compute_effective_rank_dispatch(
        self,
        eigenvalues: np.ndarray,
        rank_method: str = "spectral_gap",
        relative_threshold: float = 1e-6,
        spectral_gap_min_depth: float = 2.0,
    ) -> tuple[int, float]:
        """Dispatch effective-rank computation and report the gap depth.

        Returns a ``(effective_rank, gap_depth)`` tuple where ``gap_depth`` is
        the spectral-gap depth (orders of magnitude) for the chosen split; it is
        ``0.0`` for methods that do not produce a gap estimate.
        """
        if rank_method == "threshold":
            return self._compute_effective_rank(eigenvalues, relative_threshold), 0.0
        if rank_method == "entropy":
            rank = int(round(self._compute_entropy_effective_rank(eigenvalues)))
            return max(0, rank), 0.0
        # default: spectral_gap
        return self._compute_spectral_gap_rank(eigenvalues, spectral_gap_min_depth)

    def _compute_spectral_gap_rank(
        self,
        eigenvalues: np.ndarray,
        min_gap_depth: float = 2.0,
    ) -> tuple[int, float]:
        """Effective rank via the largest log10 spectral gap.

        The eigenvalue spectrum is inspected in descending order. The largest
        drop between adjacent eigenvalues (measured in log10 orders of
        magnitude) identifies the boundary between the identifiable and sloppy
        subspaces. If no drop reaches ``min_gap_depth`` orders of magnitude, the
        spectrum is treated as full-rank (no sloppy directions). This is the
        discrete analogue of the gap criterion used in profile-likelihood
        identifiability analysis and is robust to absolute FIM scaling.

        Returns ``(rank, gap_depth)`` where ``rank`` counts the eigenvalues
        before the largest qualifying gap and ``gap_depth`` is its magnitude.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if eigenvalues.size == 0:
            return 0, 0.0
        max_eigenvalue = np.max(eigenvalues)
        if max_eigenvalue <= 0:
            return 0, 0.0
        if eigenvalues.size == 1:
            return 1, 0.0
        log_ev = np.log10(np.maximum(eigenvalues, 1e-300))
        gaps = -np.diff(log_ev)  # positive values = descending drops
        if gaps.size == 0:
            return int(eigenvalues.size), 0.0
        gap_idx = int(np.argmax(gaps))
        gap_depth = float(gaps[gap_idx])
        if gap_depth >= min_gap_depth:
            return gap_idx + 1, gap_depth
        return int(eigenvalues.size), gap_depth

    def _compute_effective_rank(
        self,
        eigenvalues: np.ndarray,
        relative_threshold: float = 1e-6,
    ) -> int:
        """Compute threshold-based effective rank."""
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if eigenvalues.size == 0:
            return 0
        max_eigenvalue = np.max(eigenvalues)
        if max_eigenvalue <= 0:
            return 0
        return int(np.sum(eigenvalues > relative_threshold * max_eigenvalue))

    def _compute_entropy_effective_rank(self, eigenvalues: np.ndarray) -> float:
        """Compute entropy effective rank as a supplementary diagnostic."""
        eig_pos = np.asarray(eigenvalues, dtype=float)
        eig_pos = eig_pos[eig_pos > 0]
        if eig_pos.size == 0:
            return 0.0
        p = eig_pos / np.sum(eig_pos)
        entropy = -np.sum(p * np.log(p + 1e-300))
        return float(np.exp(entropy))

    def classify_sloppy(
        self,
        eigenvalues: np.ndarray,
        prior_precision: float = 1.0,
        likelihood_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Classify directions as identifiable, prior-dominated, or likelihood-tail.

        Args:
            eigenvalues: FIM eigenvalues, preferably sorted descending.
            prior_precision: Prior precision threshold for prior-dominated sloppy
                directions.
            likelihood_threshold: Relative eigenvalue threshold for identifiable
                directions.

        Returns:
            Dictionary with classification labels and indices.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if eigenvalues.size == 0 or np.max(eigenvalues) <= 0:
            classifications = ["prior_dominated"] * len(eigenvalues)
        else:
            max_eigenvalue = np.max(eigenvalues)
            classifications = []
            for eig_val in eigenvalues:
                if eig_val >= likelihood_threshold * max_eigenvalue:
                    classifications.append("identifiable")
                elif eig_val < prior_precision:
                    classifications.append("prior_dominated")
                else:
                    classifications.append("likelihood_tail")

        identifiable_indices = [i for i, label in enumerate(classifications) if label == "identifiable"]
        prior_indices = [i for i, label in enumerate(classifications) if label == "prior_dominated"]
        tail_indices = [i for i, label in enumerate(classifications) if label == "likelihood_tail"]
        return {
            "classifications": classifications,
            "identifiable_indices": identifiable_indices,
            "prior_dominated_indices": prior_indices,
            "likelihood_tail_indices": tail_indices,
            "n_identifiable": len(identifiable_indices),
            "n_prior_dominated": len(prior_indices),
            "n_likelihood_tail": len(tail_indices),
        }

    def compute_condition_number(self, fim: np.ndarray) -> float:
        """Compute condition number of FIM from positive eigenvalues."""
        eigenvalues = np.linalg.eigvalsh(0.5 * (fim + fim.T))
        return self.compute_condition_number_from_eigenvalues(eigenvalues)

    def compute_condition_number_from_eigenvalues(self, eigenvalues: np.ndarray) -> float:
        """Compute condition number from an eigenvalue vector."""
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if eigenvalues.size == 0:
            return float("inf")
        return float(np.max(eigenvalues) / np.min(eigenvalues))


def grassmannian_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute Grassmannian distance between two subspaces.

    The Grassmannian distance is based on the principal angles between subspaces.
    """
    S = np.linalg.svd(U1.T @ U2, compute_uv=False)
    principal_angles = np.arccos(np.clip(S, -1.0, 1.0))
    return float(np.linalg.norm(principal_angles))


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Compute subspace overlap (mean cosine of principal angles)."""
    S = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return float(np.mean(S))
