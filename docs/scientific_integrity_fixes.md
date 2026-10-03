# Scientific integrity fixes

These changes repair defects in the public research preview. They do not
recompute manuscript CSV files, validate historical experiments, reconstruct
the author's private experiment code, or establish that past runs used this
public implementation.

## MCMC summaries (schema 2.0)

- Posterior mean and covariance use all post-burn-in walker states, not
  averages of walkers. The retained-draw count includes walkers.
- Independent ensembles receive deterministic sampler and initialization
  seeds.
- Rank normalization pools ranks across independent chains. Split R-hat is
  the maximum of location and folded-scale diagnostics. Tail ESS measures
  quantile indicators, not the second half of a chain.
- For each fixed walker index, diagnose its chains across independently
  seeded ensembles. Report worst R-hat/tau and minimum ESS across indices.
  ESS is deliberately not multiplied by the dependent walker count.
  The conservative diagnostic method is recorded in each summary.
- Versioned runs save chains.npz in (ensemble, walker, step, parameter)
  order, along with parameter IDs and the schema. Old summary-only results
  and results from another configuration cannot silently be resumed.

If historical posterior covariances used walker averages, recalculate them
from the original retained states and recheck convergence. Do not multiply
old covariance matrices by a guessed walker correction factor. Do not
automatically retrain a neural model before determining which labels or
metrics were affected.

## Frozen holdout and inner CV (schema 3.0)

Fit the scaler, intercept and Ridge coefficients on training data only:

~~~python
from iaapi.evaluation.stat_protocol import fit_ridge, SealedHoldout

fitted = fit_ridge(
    X_train, y_train, alpha=alpha_selected_on_training_cv,
    primary_descriptor_idx=(0, 1, 2),
    training_model_ids=train_ids,
)
holdout = SealedHoldout(
    X_test, y_test, predictor=fitted, model_ids=test_ids,
)
result = holdout.evaluate_once()
~~~

The previous evaluate_once(alpha=...) interface is intentionally removed:
it trained on holdout labels. The new interface only calls the frozen
predictor. Provide both ID lists to enable train/test model overlap checking;
without them, disjointness is explicitly reported as unchecked. Independently
enforce parent/perturbation family separation using the split manifest.

Inner LOO now fits preprocessing inside each training fold. Cached prediction
weights depend only on each fold's X and alpha, never on targets or a scaler
fitted across the validation row. This keeps nested permutation tests fast
while reproducing actual fold-by-fold refits.

## FIM descriptor versions

Version 2.0 is the default and is recorded in each descriptor result:

- Feature 7: top_3_information_fraction, taking all eigenvalues when n < 3.
- Feature 8: decay_slope, minus the least-squares slope of log10 eigenvalues
  against indices 1 through n over the entire descending spectrum.
- Information scaling uses scale^(1/2) FIM scale^(1/2). Covariance scaling
  uses the inverse transformation; these two transformations are different.
  The API's prior_cov name is retained, but a bound-derived scale is not
  necessarily the empirical-prior covariance.
- Zero eigenvalues in version 2.0 use a relative floor of 1e-300 for logs.
  Persist this numerical convention as well as the version and ordered
  feature names with every feature matrix and fitted Ridge.

For explicitly identified old models, feature_version="1.0" retains the old
inverse-scale convention, top-ceil(n/3) fraction and threshold fraction.
It is an explicit compatibility mode, not a valid replacement for the
manuscript's new definitions. Do not feed default version-2 features to an
old Ridge solely because both vectors have eight columns. Verify actual
historical feature names and scaling before deciding whether the Ridge
must be refit. This change does not demonstrate that neural training
must be repeated.

## No random scientific result fallback

CoverageEvaluator requires a posterior_provider(model, problem, n_samples)
that returns actual posterior draws and known generating parameters in the
same coordinates. It returns coordinate hit indicators, denominators and
equal-tailed coverage. Averaging coordinates of one problem is not the same
as independent replicated-data coverage or its confidence interval.

OEDModule requires a validated information_gain_fn(basis, theta, candidate,
simulator), with higher values better. For expected target posterior variance,
return its negative. Preserve a common target basis across candidate designs
and record selection costs separately from independent reference evaluation.
Without a scorer, design ranking raises NotImplementedError.

Unimplemented pyPESTO/SBI/BayesFlow convenience adapters now raise
NotImplementedError rather than returning random parameters, diagnostics or
timings. Use the actual validated backend or supply an implementation.
Problem-level biological prior sampling similarly refuses an unspecified
parameter-to-type mapping; explicit single-type distribution sampling remains
available. No empirical database or paper result is fabricated.

## Author implementation still required

The public TrainingDataGenerator and full-model batch path still lack the
author-confirmed separation between MAP-based input FIM and true-parameter
label FIM. The convenience simulate_with_noise path also retains relative
trajectory-scale noise; it has not been reconstructed into the actual
PEtab observation/noise likelihood used by the manuscript.

Do not use these legacy training entry points to claim reproduction of the
MAP-input or sampled-noise manuscript protocol. Integrate the author's actual
optimizer, parameter/observable transforms, noise formulas, input/label fields,
dataset loaders and checkpoints first. Their settings cannot be inferred
reliably from the manuscript's aggregate CSV files.

Original figure values, rank-scan definitions, confidence intervals, design
replicate results and historical checkpoints remain unvalidated by this
code-only repair.

## Regression validation

Run the focused tests with NumPy, SciPy, pytest and emcee installed:

~~~bash
python -m pytest -q tests/test_review_regressions.py tests/test_mcmc.py \
  tests/test_fim_descriptors.py tests/test_stat_protocol.py
~~~

Checks include posterior covariance recovery on an analytic Gaussian,
location/scale disagreement between chains, raw-chain reproducibility of
summaries, holdout-label independence of predictions, true fold-wise
standardization, versioned descriptor definitions, invariance to parameter
units, actual coverage hits, and rejection of random-result adapters.
External pilot-shard tests skip when the corresponding data is unavailable.
This test set does not run AMICI integrations, neural training or manuscript
experiments.
