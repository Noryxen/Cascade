from typing import Any, Dict

from omegaconf import DictConfig

from evals.muse import MUSEEvaluator
from evals.tofu import TOFUEvaluator

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def _maybe_register_lm_eval():
    """Lazy-import LMEvalEvaluator only when needed (avoids hard lm_eval dependency)."""
    if "LMEvalEvaluator" not in EVALUATOR_REGISTRY:
        try:
            from evals.lm_eval import LMEvalEvaluator

            _register_evaluator(LMEvalEvaluator)
        except ImportError:
            pass  # lm_eval not installed; LMEvalEvaluator will be unavailable


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    if evaluator_handler_name is None:
        return None
    _maybe_register_lm_eval()
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        raise NotImplementedError(f"{evaluator_handler_name} not implemented or not registered")
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluator = get_evaluator(eval_name, eval_cfg, **kwargs)
        if evaluator is not None:
            evaluators[eval_name] = evaluator
    return evaluators


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
# LMEvalEvaluator is registered lazily in _maybe_register_lm_eval()
