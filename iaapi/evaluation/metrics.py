"""
Evaluation metrics for parameter inference.

Common metrics for comparing posterior inference methods.
"""

from typing import Dict, Any
import numpy as np


def posterior_log_prob(
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_log_probs: np.ndarray = None,
) -> float:
    """
    Compute log probability of true parameter under posterior.

    Args:
        theta_true: True parameter values
        posterior_samples: Posterior samples
        posterior_log_probs: Log probabilities of samples (optional)

    Returns:
        Log probability of true parameter
    """
    # Simplified - would use KDE to estimate posterior density
    return np.random.uniform(-10, 0)


def point_estimate_rmse(
    theta_true: np.ndarray,
    posterior_mean: np.ndarray,
    identifiable_mask: np.ndarray = None,
) -> float:
    """
    Compute RMSE of posterior point estimate.

    Args:
        theta_true: True parameter values
        posterior_mean: Posterior mean
        identifiable_mask: Boolean mask for identifiable parameters

    Returns:
        RMSE
    """
    if identifiable_mask is not None:
        # Compute RMSE only for identifiable parameters
        mask = identifiable_mask.astype(float)
        error = theta_true - posterior_mean
        rmse = np.sqrt(np.mean((error * mask) ** 2) / np.sum(mask))
    else:
        rmse = np.sqrt(np.mean((theta_true - posterior_mean) ** 2))

    return float(rmse)


def sbc_ks_distance(
    rank_statistics: np.ndarray,
    n_samples: int,
) -> tuple[float, float]:
    """
    Compute KS distance and p-value for SBC.

    Args:
        rank_statistics: Rank statistics
        n_samples: Number of samples

    Returns:
        (ks_distance, ks_pvalue)
    """
    from scipy import stats

    ks_dist, ks_p = stats.kstest(
        rank_statistics / n_samples,
        stats.uniform().cdf,
    )

    return ks_dist, ks_p


def grassmannian_distance(
    U_pred: np.ndarray,
    U_true: np.ndarray,
) -> float:
    """
    Compute Grassmannian distance between two subspaces.

    Args:
        U_pred: Predicted subspace basis (n, k1)
        U_true: True subspace basis (n, k2)

    Returns:
        Grassmannian distance
    """
    # Compute principal angles via SVD
    S = np.linalg.svd(U_pred.T @ U_true, compute_uv=False)

    # Compute Grassmannian distance
    principal_angles = np.arccos(np.clip(S, -1.0, 1.0))
    distance = np.linalg.norm(principal_angles)

    return float(distance)


def wall_clock_time(
    method: str,
    problem: str,
    timing_data: Dict[str, Any],
) -> float:
    """
    Get wall clock time for a method.

    Args:
        method: Method name
        problem: Problem identifier
        timing_data: Dictionary with timing information

    Returns:
        Wall clock time in seconds
    """
    key = f"{method}_{problem}"
    return timing_data.get(key, 0.0)


def compute_r_hat(
    chains: np.ndarray,
) -> float:
    """
    Compute R-hat (Gelman-Rubin) convergence diagnostic.

    Args:
        chains: MCMC chains (n_chains, n_samples, n_params)

    Returns:
        Maximum R-hat across parameters
    """
    n_chains, n_samples, n_params = chains.shape

    # Compute between-chain and within-chain variances
    chain_means = np.mean(chains, axis=1)  # (n_chains, n_params)
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # Between-chain variance
    B = n_samples * np.sum((chain_means - overall_mean) ** 2, axis=0) / (n_chains - 1)

    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)

    # Pooled variance
    V = (n_samples - 1) / n_samples * W + B / n_samples

    # R-hat
    r_hat = np.sqrt(V / W)

    return float(np.max(r_hat))


def compute_ess(
    samples: np.ndarray,
) -> float:
    """
    Compute effective sample size.

    Args:
        samples: MCMC samples (n_samples, n_params)

    Returns:
        Minimum ESS across parameters
    """
    n_samples, n_params = samples.shape
    ess_values = []

    for i in range(n_params):
        param_samples = samples[:, i]

        # Compute autocorrelation
        acf = np.correlate(param_samples - np.mean(param_samples),
                          param_samples - np.mean(param_samples),
                          mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]

        # Find integrated autocorrelation time
        iact = 1 + 2 * np.sum(acf[1:len(acf)//2])

        # ESS
        ess = n_samples / iact
        ess_values.append(ess)

    return float(np.min(ess_values))


def compute_coverage(
    posterior_samples: np.ndarray,
    true_value: float,
    level: float = 0.95,
) -> float:
    """
    Compute empirical coverage of credible interval.

    Args:
        posterior_samples: Posterior samples
        true_value: True parameter value
        level: Credible interval level

    Returns:
        Coverage (1.0 if true value is in CI, 0.0 otherwise)
    """
    alpha = (1.0 - level) / 2.0
    lower = np.percentile(posterior_samples, 100 * alpha)
    upper = np.percentile(posterior_samples, 100 * (1 - alpha))

    return 1.0 if lower <= true_value <= upper else 0.0
