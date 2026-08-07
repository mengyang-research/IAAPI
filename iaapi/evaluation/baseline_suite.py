"""Baseline suite (NCS plan task P4-01).

A unified protocol so EVERY baseline shares the same train/dev/sealed model
splits, priors, test observations and metric definitions. Lightweight baselines
(prior, local-FIM threshold, frozen-FIM Ridge probe, Laplace, PL) are implemented
here; the neural baselines (per-model NPE, BayFlow, pooled NPE, ISP ablations)
are STANDARDIZED as configs/train-eval hooks (full sbi training is a separate
heavy run).

Discipline: each baseline declares ``source`` ("prior"/"FIM"/"PL"/"MCMC"/"neural")
and ``independent`` (FIM-derived predictions are NOT independent of FIM features).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

SCHEMA_VERSION = "1.0"

# Shared metric DEFINITIONS (P4-02 implements the computation; P4-01 fixes the
# definitions so all baselines are scored identically).
METRIC_DEFINITIONS = {
    "normalized_rmse": "RMSE(theta_pred, theta_true) / RMSE(prior_mean, theta_true)",
    "mean_correlation": "Pearson r between predicted and true theta (per model, aggregated)",
    "coverage_90": "fraction of true theta inside the 90% posterior credible region",
    "coverage_95": "fraction of true theta inside the 95% posterior credible region",
    "sbc": "Simulation-Based Calibration rank statistic mean (0=perfect)",
    "posterior_contraction": "1 - (det post / det prior)^{1/d}",
}


@dataclass
class BaselineProtocol:
    """Shared protocol: splits + priors + test observations + metrics + seed."""
    splits: Dict[str, List[str]]            # {"train":[...],"dev":[...],"sealed":[...]}
    prior_mean: Dict[str, np.ndarray]       # model_id -> prior mean (theta, whitened)
    prior_cov: Dict[str, np.ndarray]        # model_id -> prior covariance (whitened)
    test_observations: Dict[str, Any]       # model_id -> test observation tensor/dict
    fim: Dict[str, np.ndarray]              # model_id -> FIM (for FIM-based baselines)
    seed: int = 0
    metric_definitions: Dict[str, str] = field(default_factory=lambda: dict(METRIC_DEFINITIONS))

    def models_in_split(self, split: str) -> List[str]:
        return list(self.splits.get(split, []))

    def all_models(self) -> List[str]:
        seen: List[str] = []
        for ms in self.splits.values():
            for m in ms:
                if m not in seen:
                    seen.append(m)
        return seen


class BaselineResult:
    """A baseline's prediction for one (model, test-obs)."""
    def __init__(self, mean: Optional[np.ndarray], cov: Optional[np.ndarray],
                 samples: Optional[np.ndarray], k_pred: Optional[int],
                 source: str, independent: bool, meta: Optional[dict] = None):
        self.mean = mean
        self.cov = cov
        self.samples = samples
        self.k_pred = k_pred
        self.source = source
        self.independent = independent
        self.meta = meta or {}


class Baseline(Protocol):
    name: str
    source: str
    independent: bool

    def fit(self, protocol: BaselineProtocol, train_data: Dict[str, Any]) -> "Baseline":
        ...

    def predict(self, protocol: BaselineProtocol, model_id: str) -> BaselineResult:
        ...


# --------------------------------------------------------------------------- #
# Lightweight baselines
# --------------------------------------------------------------------------- #
class PriorBaseline:
    """Predict from the prior (mean=prior mean, cov=prior cov). Source=prior."""
    name = "prior"
    source = "prior"
    independent = True

    def fit(self, protocol, train_data=None):
        return self

    def predict(self, protocol, model_id):
        rng = np.random.default_rng(protocol.seed + hash(model_id) % 2**31)
        mean = protocol.prior_mean[model_id]
        cov = protocol.prior_cov[model_id]
        samples = rng.multivariate_normal(mean, cov, size=200)
        return BaselineResult(mean=mean, cov=cov, samples=samples, k_pred=None,
                               source=self.source, independent=self.independent)


class LocalFIMThreshold:
    """Discrete k via local FIM eigenvalue threshold (relative). Source=FIM (NOT independent)."""
    name = "local_fim_threshold"
    source = "FIM"
    independent = False

    def __init__(self, tau_rel: float = 1e-2):
        self.tau_rel = tau_rel

    def fit(self, protocol, train_data=None):
        return self

    def predict(self, protocol, model_id):
        fim = protocol.fim[model_id]
        eig = np.linalg.eigvalsh(fim)
        eig = np.clip(eig, 0, None)
        k = int(np.sum(eig > self.tau_rel * eig.max())) if eig.max() > 0 else 0
        return BaselineResult(mean=None, cov=None, samples=None, k_pred=k,
                               source=self.source, independent=self.independent,
                               meta={"tau_rel": self.tau_rel, "eigenvalues": eig.tolist()})


class FrozenFIMRidgeProbe:
    """Ridge regression on FROZEN FIM spectral features -> target. Source=FIM.

    Predicting an FIM-derived target is CIRCULAR; predicting an independent
    (PL/MCMC) target is independent. ``independent`` reflects the target source.
    """
    name = "frozen_fim_ridge"
    source = "FIM"
    independent = False  # default: predicting FIM-derived k is circular

    def __init__(self, alpha: float = 1.0, independent_target: bool = False):
        self.alpha = alpha
        if independent_target:
            self.independent = True
            self.source = "PL"  # target comes from PL/MCMC labels

    def fit(self, protocol, train_data):
        # train_data: {"X": (n_train, n_feat), "y": (n_train,)}
        X = np.asarray(train_data["X"], float)
        y = np.asarray(train_data["y"], float)
        Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
        d = Xs.shape[1]
        self._beta = np.linalg.solve(Xs.T @ Xs + self.alpha * np.eye(d), Xs.T @ y)
        self._xmean, self._xstd = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
        return self

    def predict(self, protocol, model_id):
        X = np.asarray(protocol.test_observations.get(model_id), float).reshape(1, -1)
        Xs = (X - self._xmean) / self._xstd
        pred = float((Xs @ self._beta).reshape(-1)[0])
        return BaselineResult(mean=None, cov=None, samples=None, k_pred=None,
                               source=self.source, independent=self.independent,
                               meta={"ridge_prediction": pred})


class LaplaceApproximation:
    """Posterior ~ N(theta_MAP, FIM^-1) at the nominal/MAP. Source=FIM (Laplace uses FIM)."""
    name = "laplace"
    source = "FIM"
    independent = False

    def fit(self, protocol, train_data=None):
        return self

    def predict(self, protocol, model_id):
        fim = protocol.fim[model_id]
        mean = protocol.prior_mean[model_id]  # MAP ~ nominal in whitened coords
        # cov = FIM^-1 (add jitter for PD)
        try:
            cov = np.linalg.inv(fim + 1e-6 * np.eye(fim.shape[0]))
            cov = 0.5 * (cov + cov.T)
        except np.linalg.LinAlgError:
            cov = np.eye(fim.shape[0])
        rng = np.random.default_rng(protocol.seed + hash(model_id) % 2**31)
        samples = rng.multivariate_normal(mean, cov, size=200)
        return BaselineResult(mean=mean, cov=cov, samples=samples, k_pred=None,
                               source=self.source, independent=self.independent)


class PLBaseline:
    """Wraps P2-02 profile-likelihood per-parameter CIs. Source=PL (independent)."""
    name = "profile_likelihood"
    source = "PL"
    independent = True

    def fit(self, protocol, train_data=None):
        return self

    def predict(self, protocol, model_id):
        pl = protocol.test_observations.get(model_id, {})
        if not isinstance(pl, dict):
            pl = {}
        cis = pl.get("param_cis", [])
        k_identifiable = 0
        for ci in cis:
            lo, hi = ci
            if lo is None or hi is None:
                continue
            if np.isfinite(lo) and np.isfinite(hi):
                k_identifiable += 1
        return BaselineResult(mean=None, cov=None, samples=None, k_pred=k_identifiable,
                               source=self.source, independent=self.independent,
                               meta={"param_cis": cis, "codes": pl.get("codes", [])})


# --------------------------------------------------------------------------- #
# Standardized neural baseline specs (full training is a separate heavy run)
# --------------------------------------------------------------------------- #
@dataclass
class NeuralBaselineSpec:
    """Config/hook for a neural baseline; shares the protocol's splits/priors/obs/metrics."""
    name: str
    kind: str            # "per_model_npe" | "pooled_npe" | "bayesflow" | "isp_ablation"
    independent: bool = True
    ablation: Optional[str] = None  # for ISP ablations: "no_graph"/"no_shape"/"no_scale"/"no_geometry_loss"
    train_epochs: int = 50
    seed: int = 0
    notes: str = ""

    def requires_heavy_training(self) -> bool:
        return True


def default_neural_specs() -> List[NeuralBaselineSpec]:
    return [
        NeuralBaselineSpec("per_model_npe", "per_model_npe", notes="sbi NPE trained per model"),
        NeuralBaselineSpec("pooled_npe", "pooled_npe", notes="NPE pooled across models with model-ID token"),
        NeuralBaselineSpec("bayesflow", "bayesflow", notes="strong conditional-flow baseline"),
        NeuralBaselineSpec("isp_no_graph", "isp_ablation", ablation="no_graph"),
        NeuralBaselineSpec("isp_no_shape", "isp_ablation", ablation="no_shape"),
        NeuralBaselineSpec("isp_no_scale", "isp_ablation", ablation="no_scale"),
        NeuralBaselineSpec("isp_no_geometry_loss", "isp_ablation", ablation="no_geometry_loss"),
    ]


def all_baselines() -> Dict[str, Any]:
    """Registry of all baselines (implemented + standardized)."""
    return {
        "prior": PriorBaseline(),
        "local_fim_threshold": LocalFIMThreshold(),
        "frozen_fim_ridge": FrozenFIMRidgeProbe(),
        "laplace": LaplaceApproximation(),
        "profile_likelihood": PLBaseline(),
        "neural_specs": default_neural_specs(),
    }
