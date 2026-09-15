import numpy as np
import scipy as sc

from evals.metrics.base import unlearning_metric


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    values = [result["agg_value"] for _, result in kwargs["pre_compute"].items()]
    return {"agg_value": sc.stats.hmean(values)}


@unlearning_metric(name="composite_forgetting_index")
def composite_forgetting_index(model, **kwargs):
    """Compute CFI for TOFU using direction-corrected forget metrics.

    CFI = HM(1 - forget_Q_A_Prob, 1 - forget_Q_A_ROUGE, forget_truth_ratio)
    """
    eps = kwargs.get("eps", 1e-8)

    forget_prob = kwargs["pre_compute"]["forget_prob"]["agg_value"]
    forget_rouge = kwargs["pre_compute"]["forget_rouge"]["agg_value"]
    forget_truth_ratio = kwargs["pre_compute"]["forget_truth_ratio"]["agg_value"]

    values = np.array(
        [
            1 - float(forget_prob),
            1 - float(forget_rouge),
            float(forget_truth_ratio),
        ],
        dtype=np.float64,
    )
    values = np.clip(values, eps, 1.0)
    return {"agg_value": sc.stats.hmean(values)}


@unlearning_metric(name="balanced_unlearning_score")
def balanced_unlearning_score(model, **kwargs):
    """Compute the Balanced Unlearning Score (BUS), a parameter-free metric
    that combines forgetting quality (CFI) and model utility into a single scalar.

    BUS = 2 * CFI * Utility / (CFI + Utility)

    This is the harmonic mean of CFI and Utility (equivalent to the F1-score).
    The harmonic mean is chosen over arithmetic or geometric mean because it
    is strictly dominated by the smaller of the two values — a method that
    achieves strong forgetting by collapsing model utility (or vice versa)
    will receive a low BUS. This naturally enforces the core trade-off in LLM
    unlearning without any tunable hyperparameters or ad-hoc thresholds.

    Args:
        model: The model being evaluated (unused; required by the framework).
        **kwargs: Must contain pre-computed CFI and Utility metrics under
                  pre_compute["cfi"]["agg_value"] and
                  pre_compute["utility"]["agg_value"].

    Returns:
        dict with "agg_value" containing the BUS scalar in [0, 1].
    """
    cfi = float(kwargs["pre_compute"]["cfi"]["agg_value"])
    utility = float(kwargs["pre_compute"]["utility"]["agg_value"])

    # Harmonic mean: when either term is zero the result is zero
    eps = kwargs.get("eps", 1e-10)
    bus = 2.0 * cfi * utility / (cfi + utility + eps)

    return {"agg_value": float(bus)}
