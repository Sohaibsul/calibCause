"""Threshold-refit diagnostic (reviewer comment 4).

Reviewer's point: holding EyePACS out for LOEO removes ~2/3 of the training
images, so the LOEO-heldout-EyePACS QWK drop conflates two different things
-- (a) a genuine loss of transferable features, and (b) a grading-standard
shift between datasets (different graders / different severity thresholds
between EyePACS, DDR and APTOS) that a fixed argmax decision rule can't
absorb. The cheap diagnostic they suggest: fit the ordinal decision
thresholds on a *small* EyePACS calibration sample (a couple hundred images,
drawn from the validation split -- never the test split, to avoid
double-dipping) and see whether QWK on the held-out EyePACS test set jumps.
If it jumps substantially, much of the failure is (b), not (a).

What "fit the thresholds" means here: instead of predicting
argmax_k softmax(logits)_k, take the model's expected grade
E[grade] = sum_k k * softmax(logits)_k (a continuous score in [0, 4]) and
learn 4 cut points t1 < t2 < t3 < t4 that turn that continuous score into a
discrete grade, choosing the cut points to maximise QWK on the calibration
sample. This changes nothing about the model or its features -- only where
the decision boundaries between adjacent grades sit -- which is exactly the
"grading-standard shift" the reviewer is asking about.

Repeats the whole thing --n-draws times with different random calibration
subsets (default 30) so the result is a mean +/- std, not one lucky draw.
Also runs the identical procedure on an in-domain run (--control-preds /
--control-calib) as a sanity check: in-domain there's no real distribution
shift to correct for, so the control's refit QWK should barely move, while
the LOEO-EyePACS refit QWK is the one expected to move if the reviewer's
hypothesis is right.

Usage:
    python scripts/threshold_refit.py \
        --test-preds runs/stage2/preds_loeo_heldout_eyepacs_test.npz \
        --calib-source runs/stage2/preds_loeo_heldout_eyepacs_val.npz \
        --n-calib 200 --n-draws 30 \
        --control-preds runs/stage2/preds_teacher_test.npz \
        --control-calib runs/stage2/preds_teacher_val.npz

Needs preds_*_val.npz files, which evaluate.py only writes when called with
--calibrate-temperature. If a run's *_val.npz is missing, re-run:
    python -m src.engine.evaluate --config <cfg> --ckpt <ckpt> \
        --calibrate-temperature --split val --envs eyepacs --tag <tag>
(no retraining needed -- it's just a forward pass over the val split.)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _fast_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Vectorised QWK, used only inside the inner optimisation loop where it's
    called tens of thousands of times per calibration draw -- sklearn's
    cohen_kappa_score recomputes a lot of Python-side bookkeeping per call
    that adds up at that volume. Same formula, just without the overhead."""
    # confusion matrix via bincount (faster than sklearn.confusion_matrix here)
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    O = np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.float64)
    w = (np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]) ** 2
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / max(O.sum(), 1.0)
    num = (w * O).sum()
    den = (w * E).sum()
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1.0 - num / den)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_grade(logits: np.ndarray) -> np.ndarray:
    p = softmax(logits)
    grades = np.arange(p.shape[1])
    return (p * grades[None, :]).sum(axis=1)


def _thresholds_from_params(params: np.ndarray) -> np.ndarray:
    """params -> 4 strictly-increasing thresholds, via a positive-delta
    reparameterisation so unconstrained Nelder-Mead can't produce an invalid
    (non-monotonic) set of cut points."""
    t0 = params[0]
    deltas = np.exp(np.clip(params[1:], -10, 10))
    return np.cumsum(np.concatenate([[t0], deltas]))


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, thresholds)


def default_thresholds() -> np.ndarray:
    """Thresholds equivalent to plain rounding of the expected grade -- the
    fair "before" baseline for this diagnostic (same decision-rule family as
    the refit thresholds, just not fit to any calibration data)."""
    return np.array([0.5, 1.5, 2.5, 3.5])


def fit_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Global derivative-free search over 4 cut points maximising QWK on
    (scores, labels).

    QWK as a function of threshold position is piecewise-constant (it only
    changes when a threshold crosses a data point) -- locally flat almost
    everywhere, with occasional jumps. A local method like Nelder-Mead,
    started exactly at the default thresholds, can get trapped on that first
    flat plateau and report "converged" after barely moving (this is exactly
    what happened in the buggy first version of this function -- verified by
    the suspiciously tiny across-draw std). differential_evolution is a
    population-based global search that explores the whole threshold range
    regardless of local flatness, so it isn't fooled by the plateau.
    """
    lo, hi = float(scores.min()), float(scores.max())
    span = max(hi - lo, 1e-3)
    # bounds: t0 anywhere in [lo, hi]; each log-delta wide enough that the
    # cumulative thresholds can span the full observed score range
    bounds = [(lo, hi)] + [(-4.0, np.log(span))] * 3

    labels_int = labels.astype(np.int64)

    def neg_qwk(params):
        t = _thresholds_from_params(params)
        pred = apply_thresholds(scores, t)
        return -_fast_qwk(labels_int, pred)

    res = differential_evolution(neg_qwk, bounds, seed=0, maxiter=60, popsize=15,
                                 tol=1e-4, mutation=(0.5, 1.5), recombination=0.7,
                                 polish=False, init="sobol")
    fitted = _thresholds_from_params(res.x)

    # a global search can still occasionally land worse than the untouched
    # default (e.g. on a small/unlucky calibration draw) -- never let the
    # "fit" make things worse than not fitting at all.
    default_qwk = -neg_qwk(np.array([0.5, 0.0, 0.0, 0.0]))
    if -res.fun < default_qwk:
        return default_thresholds()
    return fitted


def run_diagnostic(test_npz: str, calib_npz: str, n_calib: int, n_draws: int,
                    seed0: int = 0) -> dict:
    test = np.load(test_npz)
    calib_pool = np.load(calib_npz)

    test_scores = expected_grade(test["logits"])
    test_labels = test["labels"]
    test_argmax_pred = test["logits"].argmax(axis=1)
    raw_argmax_qwk = qwk(test_labels, test_argmax_pred)
    # fair "before" baseline: SAME decision-rule family as the refit
    # thresholds (expected-grade + cut points), just not fit to any
    # calibration data. Comparing refit against this, not against argmax,
    # isolates the effect of moving the thresholds -- argmax over the full
    # categorical softmax is a different (and usually stronger) decision
    # rule than thresholding a single scalar summary of it, so a refit-vs-
    # argmax delta would be confounded by that switch, not just by the
    # calibration shift the reviewer is asking about.
    default_pred = apply_thresholds(test_scores, default_thresholds())
    default_qwk = qwk(test_labels, default_pred)

    calib_scores_pool = expected_grade(calib_pool["logits"])
    calib_labels_pool = calib_pool["labels"]
    n_pool = len(calib_labels_pool)
    n_use = min(n_calib, n_pool)
    if n_use < n_calib:
        print(f"  [warn] calibration pool only has {n_pool} images (< requested {n_calib}); "
              f"using all {n_use}")

    refit_qwks = []
    for draw in range(n_draws):
        rng = np.random.default_rng(seed0 + draw)
        idx = rng.choice(n_pool, size=n_use, replace=False)
        thresholds = fit_thresholds(calib_scores_pool[idx], calib_labels_pool[idx])
        refit_pred = apply_thresholds(test_scores, thresholds)
        refit_qwks.append(qwk(test_labels, refit_pred))
    refit_qwks = np.array(refit_qwks)

    return {
        "n_test": int(len(test_labels)),
        "n_calib_pool": int(n_pool),
        "n_calib_used": int(n_use),
        "n_draws": int(n_draws),
        "raw_argmax_qwk": raw_argmax_qwk,
        "default_threshold_qwk": default_qwk,
        "refit_qwk_mean": float(refit_qwks.mean()),
        "refit_qwk_std": float(refit_qwks.std()),
        "refit_qwk_min": float(refit_qwks.min()),
        "refit_qwk_max": float(refit_qwks.max()),
        # Secondary/context only. In principle this isolates "moving the
        # thresholds away from default", but empirically default_threshold_qwk
        # (plain rounding of E[grade]) can be a weak baseline for reasons that
        # have nothing to do with domain shift -- the softmax mean isn't
        # generally the softmax mode, so rounding can lag argmax by a lot even
        # with zero distribution shift (confirmed on the in-domain control:
        # this delta came out large there too). Kept for transparency, but
        # don't use it alone to argue the reviewer's point -- use
        # delta_vs_argmax together with a control run instead (see main()).
        "delta_vs_default": float(refit_qwks.mean() - default_qwk),
        # refit vs the run's OWN raw argmax QWK. This still carries a small
        # generic decision-rule-switch effect (argmax over the full
        # categorical distribution vs. thresholding a scalar summary of it),
        # but that generic effect is roughly constant across domains for the
        # same model/backbone -- so it cancels out when this number is
        # compared against the SAME quantity computed on an in-domain control
        # run. That cross-run comparison (done in main() when --control-preds
        # is given) is the actual answer to the reviewer's question, not
        # either delta on its own.
        "delta_vs_argmax": float(refit_qwks.mean() - raw_argmax_qwk),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-preds", required=True,
                    help="preds_*_test.npz for the run under diagnosis (e.g. LOEO-heldout-EyePACS)")
    ap.add_argument("--calib-source", required=True,
                    help="preds_*_val.npz to draw the small calibration sample from (never test)")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-draws", type=int, default=30)
    ap.add_argument("--control-preds", default=None,
                    help="optional in-domain preds_*_test.npz, run through the same procedure as a sanity check")
    ap.add_argument("--control-calib", default=None)
    ap.add_argument("--out", default="reports/threshold_refit_results.json")
    args = ap.parse_args()

    print(f"[threshold_refit] target run: {args.test_preds}")
    target = run_diagnostic(args.test_preds, args.calib_source, args.n_calib, args.n_draws)
    print(f"  raw argmax QWK          = {target['raw_argmax_qwk']:.4f}")
    print(f"  threshold-refit QWK     = {target['refit_qwk_mean']:.4f} "
         f"+/- {target['refit_qwk_std']:.4f}  "
         f"(range {target['refit_qwk_min']:.4f}-{target['refit_qwk_max']:.4f}, "
         f"n={target['n_draws']} draws, {target['n_calib_used']} calib imgs/draw)")
    print(f"  delta vs argmax         = {target['delta_vs_argmax']:+.4f}  "
         f"(interpret together with the control run below, not alone)")
    print(f"  delta vs default        = {target['delta_vs_default']:+.4f}  "
         f"(context only -- empirically NOT clean of domain-shift-unrelated confounds, see control)")

    results = {"target": target}

    if args.control_preds and args.control_calib:
        print(f"\n[threshold_refit] control (in-domain) run: {args.control_preds}")
        control = run_diagnostic(args.control_preds, args.control_calib, args.n_calib, args.n_draws)
        print(f"  raw argmax QWK          = {control['raw_argmax_qwk']:.4f}")
        print(f"  threshold-refit QWK     = {control['refit_qwk_mean']:.4f} "
             f"+/- {control['refit_qwk_std']:.4f}")
        print(f"  delta vs argmax         = {control['delta_vs_argmax']:+.4f}")
        print(f"  delta vs default        = {control['delta_vs_default']:+.4f}")

        # THE number: how much more does threshold-refit recover under the
        # LOEO domain shift than it does with no shift at all? This
        # subtraction cancels out the generic argmax-vs-threshold decision-
        # rule effect (present in both runs, same model family), isolating
        # whatever part of it is specific to the target run's distribution
        # shift -- which is exactly what the reviewer asked to see measured.
        cross = target["delta_vs_argmax"] - control["delta_vs_argmax"]
        print(f"\n  domain-shift-specific recovery (target delta_vs_argmax - control delta_vs_argmax)")
        print(f"  = {target['delta_vs_argmax']:+.4f} - ({control['delta_vs_argmax']:+.4f}) = {cross:+.4f}  "
             f"<-- report THIS number, not either delta alone")
        results["control"] = control
        results["domain_shift_specific_recovery"] = float(cross)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
