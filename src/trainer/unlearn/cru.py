from __future__ import annotations

import hashlib
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


def _module_output_to_tensor(module_output):
    if torch.is_tensor(module_output):
        return module_output
    if isinstance(module_output, (tuple, list)) and len(module_output) > 0:
        if torch.is_tensor(module_output[0]):
            return module_output[0]
    return None


class Cascade(GradDiff):
    """Cascade: hierarchical recoverability control for LLM unlearning.

    This implementation follows a practical, train-time variant with three stages:
    1) Route discovery via contrastive activation statistics (priv vs safe when available)
    2) Route attribution via module importance ranking
    3) Route suppression with an activation penalty on selected modules

    It is intentionally lightweight so it can plug into the existing unlearning stack.
    """

    def __init__(
        self,
        lambda_forget: float = 2.0,
        alpha: float = 1.0,
        gamma: float = 0.0,
        retain_loss_type: str = "NLL",
        topk_modules: int = 24,
        route_update_interval: int = 20,
        route_ema_decay: float = 0.85,
        threshold_tau: float = 0.0,
        dynamic_threshold: bool = True,
        threshold_power: float = 2.0,
        contrastive_forget: bool = True,
        contrastive_margin: float = 0.0,
        normalize_route_loss: bool = True,
        lambda_warmup_steps: int = 40,
        lambda_forget_boost_start_step: int = -1,
        lambda_forget_boost_factor: float = 1.0,
        candidate_patterns: Optional[List[str]] = None,
        cas_curvature: float = 1.0,
        cas_hierarchy_margin: float = 0.1,
        cas_hierarchy_weight: float = 0.0,
        cas_hierarchy_use_retain: bool = False,
        cas_locality_weight: float = 0.0,
        cas_locality_tau: float = 0.0,
        cas_compress_weight: float = 0.0,
        cas_compress_tau: float = 0.0,
        cas_eps: float = 1e-6,
        cas_projection_enable: bool = True,
        cas_projection_hidden_dim: int = 0,
        cas_projection_out_dim: int = 0,
        cas_projection_dropout: float = 0.0,
        cas_gate_enable: bool = True,
        cas_gate_a: float = 0.0,
        cas_gate_b: float = 2.0,
        cas_sensitive_top_ratio: float = 0.5,
        cas_route_bank_enable: bool = False,
        cas_route_bank_max_size: int = 50000,
        cas_route_discovery_mode: str = "activation",
        cas_patching_topm: int = 8,
        cas_patching_score_weight: float = 1.0,
        cas_direct_forget_weight: float = 0.0,
        cas_direct_safe_weight: float = 1.0,
        cas_direct_pair_weight: float = 1.0,
        cas_direct_margin: float = 0.0,
        cas_direct_use_softplus: bool = True,
        cas_direct_warmup_steps: int = 0,
        cas_direct_safe_decay_start_step: int = -1,
        cas_direct_safe_decay_duration: int = 0,
        cas_direct_safe_min_weight: Optional[float] = None,
        cas_freeze_projection_head: bool = False,
        cru_forget_use_softplus: bool = False,
        cru_forget_softplus_beta: float = 1.0,
        cas_repr_mode: str = "hyperbolic",
        route_selection_mode: str = "cascade",
        route_random_seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            retain_loss_type=retain_loss_type,
            *args,
            **kwargs,
        )
        self.lambda_forget = lambda_forget
        self.topk_modules = topk_modules
        self.route_update_interval = route_update_interval
        self.route_ema_decay = route_ema_decay
        self.threshold_tau = threshold_tau
        self.dynamic_threshold = dynamic_threshold
        self.threshold_power = threshold_power
        self.contrastive_forget = contrastive_forget
        self.contrastive_margin = contrastive_margin
        self.normalize_route_loss = normalize_route_loss
        self.lambda_warmup_steps = lambda_warmup_steps
        self.lambda_forget_boost_start_step = int(lambda_forget_boost_start_step)
        self.lambda_forget_boost_factor = max(0.0, float(lambda_forget_boost_factor))
        self.cas_curvature = max(float(cas_curvature), 1e-8)
        self.cas_hierarchy_margin = cas_hierarchy_margin
        self.cas_hierarchy_weight = max(0.0, float(cas_hierarchy_weight))
        self.cas_hierarchy_use_retain = cas_hierarchy_use_retain
        self.cas_locality_weight = max(0.0, float(cas_locality_weight))
        self.cas_locality_tau = cas_locality_tau
        self.cas_compress_weight = max(0.0, float(cas_compress_weight))
        self.cas_compress_tau = cas_compress_tau
        self.cas_eps = cas_eps
        self.cas_projection_enable = cas_projection_enable
        self.cas_projection_hidden_dim = max(0, int(cas_projection_hidden_dim))
        self.cas_projection_out_dim = max(0, int(cas_projection_out_dim))
        self.cas_projection_dropout = max(0.0, float(cas_projection_dropout))
        self.cas_gate_enable = cas_gate_enable
        self.cas_gate_a = float(cas_gate_a)
        self.cas_gate_b = float(cas_gate_b)
        self.cas_sensitive_top_ratio = min(1.0, max(0.0, float(cas_sensitive_top_ratio)))
        self.cas_route_bank_enable = cas_route_bank_enable
        self.cas_route_bank_max_size = max(0, int(cas_route_bank_max_size))
        self.cas_route_discovery_mode = str(cas_route_discovery_mode)
        self.cas_patching_topm = max(1, int(cas_patching_topm))
        self.cas_patching_score_weight = max(0.0, float(cas_patching_score_weight))
        self.cas_direct_forget_weight = max(0.0, float(cas_direct_forget_weight))
        self.cas_direct_safe_weight = max(0.0, float(cas_direct_safe_weight))
        self.cas_direct_pair_weight = max(0.0, float(cas_direct_pair_weight))
        self.cas_direct_margin = float(cas_direct_margin)
        self.cas_direct_use_softplus = bool(cas_direct_use_softplus)
        self.cas_direct_warmup_steps = max(0, int(cas_direct_warmup_steps))
        self.cas_direct_safe_decay_start_step = int(cas_direct_safe_decay_start_step)
        self.cas_direct_safe_decay_duration = max(0, int(cas_direct_safe_decay_duration))
        self.cas_direct_safe_min_weight = (
            max(0.0, float(cas_direct_safe_min_weight)) if cas_direct_safe_min_weight is not None else None
        )
        self.cas_freeze_projection_head = bool(cas_freeze_projection_head)
        self.cru_forget_use_softplus = bool(cru_forget_use_softplus)
        self.cru_forget_softplus_beta = max(0.1, float(cru_forget_softplus_beta))
        self.cas_repr_mode = str(cas_repr_mode)
        self.route_selection_mode = str(route_selection_mode)
        self.route_random_seed = int(route_random_seed)

        import logging as _logging

        _logging.getLogger("trainer").info(
            f"[Cascade] route_selection_mode={self.route_selection_mode}, route_random_seed={self.route_random_seed}"
        )

        self.candidate_patterns = candidate_patterns or [
            "self_attn.o_proj",
            "mlp.down_proj",
        ]

        self._module_name_to_module = {}
        self._sync_module_map(self.model)
        self.candidate_module_names = self._build_candidate_module_names()
        self.route_module_names: List[str] = []
        self.module_importance_ema: Dict[str, float] = defaultdict(float)
        self.module_priv_ema: Dict[str, float] = defaultdict(float)
        self._route_random = __import__("random").Random(self.route_random_seed)
        self._cas_projection_head_name = "cas_projection_head"
        self.route_bank: Dict[str, List[str]] = {}
        self._last_route_bank_hit = False
        self._maybe_init_projection_head(self.model)

    def _infer_model_hidden_size(self, model) -> Optional[int]:
        base_model = self._unwrap_model(model)
        config = getattr(base_model, "config", None)
        if config is None:
            return None

        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(config, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def _maybe_init_projection_head(self, model):
        if not self.cas_projection_enable:
            return
        if self._projection_head_exists(model):
            return
        hidden_size = self._infer_model_hidden_size(model)
        if hidden_size is None:
            return
        self._get_or_create_projection_head(model, in_dim=hidden_size)

    def _projection_head_exists(self, model) -> bool:
        base_model = self._unwrap_model(model)
        return hasattr(base_model, self._cas_projection_head_name)

    def _get_or_create_projection_head(self, model, in_dim: int):
        if not self.cas_projection_enable:
            return None

        base_model = self._unwrap_model(model)
        existing = getattr(base_model, self._cas_projection_head_name, None)
        if existing is not None:
            self._set_projection_head_trainable(existing)
            return existing

        out_dim = self.cas_projection_out_dim if self.cas_projection_out_dim > 0 else in_dim
        if self.cas_projection_hidden_dim > 0:
            head = nn.Sequential(
                nn.Linear(in_dim, self.cas_projection_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.cas_projection_dropout),
                nn.Linear(self.cas_projection_hidden_dim, out_dim),
            )
        else:
            head = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )
        head = head.to(self.accelerator.device)
        self._set_projection_head_trainable(head)
        setattr(base_model, self._cas_projection_head_name, head)
        return head

    def _set_projection_head_trainable(self, head):
        trainable = not self.cas_freeze_projection_head
        for param in head.parameters():
            param.requires_grad = trainable

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _sync_module_map(self, model):
        base_model = self._unwrap_model(model)
        self._module_name_to_module = dict(base_model.named_modules())

    def _build_candidate_module_names(self) -> List[str]:
        module_names = []
        for name, module in self._module_name_to_module.items():
            if not any(pattern in name for pattern in self.candidate_patterns):
                continue
            if not hasattr(module, "forward"):
                continue
            module_names.append(name)
        return module_names

    def _sample_keys_from_batch(self, sample: Dict[str, torch.Tensor]) -> List[str]:
        if "index" in sample:
            idx = sample["index"]
            if torch.is_tensor(idx):
                return [f"idx:{int(v)}" for v in idx.detach().flatten().tolist()]
            if isinstance(idx, list):
                return [f"idx:{int(v)}" for v in idx]
            return [f"idx:{int(idx)}"]

        # Fallback: deterministic hash of prompt prefix when index is unavailable.
        if "input_ids" not in sample:
            return []
        input_ids = sample["input_ids"]
        if not torch.is_tensor(input_ids):
            return []
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        keys = []
        rows = input_ids.detach().cpu().tolist()
        for row in rows:
            prefix = row[:16]
            key_str = "|".join(str(int(v)) for v in prefix)
            key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
            keys.append(f"hash:{key_hash}")
        return keys

    def _lookup_route_from_bank(self, sample: Dict[str, torch.Tensor]) -> List[str]:
        if not self.cas_route_bank_enable:
            self._last_route_bank_hit = False
            return []

        keys = self._sample_keys_from_batch(sample)
        if not keys:
            self._last_route_bank_hit = False
            return []

        freq = defaultdict(int)
        for key in keys:
            modules = self.route_bank.get(key, [])
            for module_name in modules:
                freq[module_name] += 1

        if not freq:
            self._last_route_bank_hit = False
            return []

        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        self._last_route_bank_hit = True
        return [name for name, _ in ranked[: self.topk_modules]]

    def _update_route_bank(self, sample: Dict[str, torch.Tensor]):
        if not self.cas_route_bank_enable or not self.route_module_names:
            return
        keys = self._sample_keys_from_batch(sample)
        if not keys:
            return

        for key in keys:
            self.route_bank[key] = list(self.route_module_names)

        if self.cas_route_bank_max_size > 0:
            overflow = len(self.route_bank) - self.cas_route_bank_max_size
            if overflow > 0:
                # Remove oldest inserted keys to cap memory growth.
                for old_key in list(self.route_bank.keys())[:overflow]:
                    self.route_bank.pop(old_key, None)

    def _to_model_inputs(self, sample: Dict[str, torch.Tensor], with_labels: bool):
        inputs = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
        }
        if with_labels and "labels" in sample:
            inputs["labels"] = sample["labels"]
        return inputs

    def _extract_priv_safe_pair(
        self, forget_batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if "original" in forget_batch and "alternate" in forget_batch:
            return forget_batch["original"], forget_batch["alternate"]
        return forget_batch, None

    def _replace_module_output(self, original_output, replacement_tensor: torch.Tensor):
        if torch.is_tensor(original_output):
            return replacement_tensor
        if isinstance(original_output, tuple) and len(original_output) > 0:
            return (replacement_tensor, *original_output[1:])
        if isinstance(original_output, list) and len(original_output) > 0:
            return [replacement_tensor, *original_output[1:]]
        return original_output

    @contextmanager
    def _capture_module_outputs(self, module_names: Iterable[str]):
        captured = {}
        hooks = []

        for name in module_names:
            module = self._module_name_to_module.get(name)
            if module is None:
                continue

            def _hook(_, __, output, module_name=name):
                tensor = _module_output_to_tensor(output)
                if tensor is not None:
                    captured[module_name] = tensor

            hooks.append(module.register_forward_hook(_hook))

        try:
            yield captured
        finally:
            for hook in hooks:
                hook.remove()

    def _mean_activation_norm(self, output_tensor: torch.Tensor) -> torch.Tensor:
        # Mean squared activation over all dimensions except batch.
        dim = tuple(range(1, output_tensor.ndim))
        return output_tensor.float().pow(2).mean(dim=dim).mean()

    def _mean_hidden_vector(self, output_tensor: torch.Tensor) -> torch.Tensor:
        hidden = output_tensor.float()
        if hidden.ndim <= 1:
            return hidden.reshape(1, -1)
        if hidden.ndim == 2:
            return hidden
        # Keep batch dimension and average all non-feature dimensions.
        reduce_dims = tuple(range(1, hidden.ndim - 1))
        if len(reduce_dims) > 0:
            hidden = hidden.mean(dim=reduce_dims)
        return hidden

    def _project_hidden_vector(self, model, hidden_vec: torch.Tensor) -> torch.Tensor:
        if not self.cas_projection_enable:
            return hidden_vec

        if hidden_vec.ndim == 1:
            hidden_vec = hidden_vec.unsqueeze(0)
        in_dim = int(hidden_vec.shape[-1])
        head = self._get_or_create_projection_head(model, in_dim=in_dim)
        if head is None:
            return hidden_vec
        # Keep projection input dtype/device aligned with head params, especially
        # when DeepSpeed runs bf16 while intermediate stats are accumulated in fp32.
        head_param = next(head.parameters(), None)
        if head_param is not None:
            hidden_vec = hidden_vec.to(
                device=head_param.device,
                dtype=head_param.dtype,
            )
        return head(hidden_vec)

    def _route_importance_tensor(self, module_names: List[str], device) -> torch.Tensor:
        if len(module_names) == 0:
            return torch.empty(0, device=device)

        importances = []
        for name in module_names:
            importances.append(float(self.module_importance_ema.get(name, 0.0)))
        imp = torch.tensor(importances, device=device)
        if imp.numel() == 0:
            return imp
        imp = imp - imp.mean()
        imp = imp / (imp.std(unbiased=False) + self.cas_eps)
        return imp

    def _route_importance_gate(self, module_names: List[str], device) -> torch.Tensor:
        if len(module_names) == 0:
            return torch.empty(0, device=device)
        if not self.cas_gate_enable:
            return torch.ones(len(module_names), device=device)

        imp = self._route_importance_tensor(module_names=module_names, device=device)
        gate = torch.sigmoid(self.cas_gate_a + self.cas_gate_b * imp)
        return gate

    def _hyperbolic_distance_to_root(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        eps = self.cas_eps
        c = hidden_vec.new_tensor(self.cas_curvature)
        sqrt_c = torch.sqrt(c)

        vec_norm = torch.linalg.norm(hidden_vec, dim=-1, keepdim=True).clamp_min(eps)
        scaled = torch.tanh(sqrt_c * vec_norm) * hidden_vec / (sqrt_c * vec_norm)

        max_radius = (1.0 - eps) / sqrt_c
        radius = torch.linalg.norm(scaled, dim=-1).clamp(max=max_radius)
        arg = (sqrt_c * radius).clamp(max=1.0 - eps)
        return (2.0 / sqrt_c) * torch.atanh(arg)

    def _euclidean_distance_to_root(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean (L2) distance from origin for each vector."""
        return torch.linalg.norm(hidden_vec, dim=-1)

    def _repr_distance(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        """Compute representation distance based on repr_mode."""
        if self.cas_repr_mode == "euclidean_norm":
            return self._euclidean_distance_to_root(hidden_vec)
        elif self.cas_repr_mode == "euclidean_contrast":
            return self._euclidean_distance_to_root(hidden_vec)
        else:
            # Default: hyperbolic
            return self._hyperbolic_distance_to_root(hidden_vec)

    def _route_hyperbolic_radius(self, model, route_cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not route_cache:
            return torch.tensor(0.0, device=self.accelerator.device)

        module_names = list(route_cache.keys())
        gates = self._route_importance_gate(module_names, device=self.accelerator.device)
        distances = []
        for out in route_cache.values():
            pooled = self._mean_hidden_vector(out)
            projected = self._project_hidden_vector(model, pooled)
            d = self._repr_distance(projected).mean()
            distances.append(d)
        dist_tensor = torch.stack(distances)
        if gates.numel() > 0:
            return (gates * dist_tensor).sum() / gates.sum().clamp_min(self.cas_eps)
        return dist_tensor.mean()

    def _route_sensitive_mask(
        self,
        module_names: List[str],
        route_distances: torch.Tensor,
    ) -> torch.Tensor:
        if route_distances.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.accelerator.device)

        if self.cas_sensitive_top_ratio <= 0.0:
            return torch.ones_like(route_distances, dtype=torch.bool)

        k = max(1, int(round(route_distances.numel() * self.cas_sensitive_top_ratio)))
        top_idx = torch.topk(route_distances, k=k, largest=True).indices
        mask = torch.zeros_like(route_distances, dtype=torch.bool)
        mask[top_idx] = True
        return mask

    def _route_weighted_compression_signal(self, model, route_cache: Dict[str, torch.Tensor]):
        if not route_cache:
            return torch.tensor(0.0, device=self.accelerator.device)

        module_names = list(route_cache.keys())
        route_distances = []
        for out in route_cache.values():
            pooled = self._mean_hidden_vector(out)
            projected = self._project_hidden_vector(model, pooled)
            route_distances.append(self._repr_distance(projected).mean())
        dist_tensor = torch.stack(route_distances)

        sensitive_mask = self._route_sensitive_mask(module_names, dist_tensor)
        gates = self._route_importance_gate(module_names, device=dist_tensor.device)
        if sensitive_mask.numel() > 0:
            gates = gates * sensitive_mask.float()

        excess = F.relu(dist_tensor - self.cas_compress_tau)
        weighted = gates * excess
        denom = gates.sum().clamp_min(self.cas_eps)
        return weighted.sum() / denom

    def _euclidean_contrast_loss(
        self,
        model,
        priv_batch: Dict[str, torch.Tensor],
        ref_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Euclidean centroid distance between forget and ref representations.

        Extracts route activations for both forget and ref (retain/safe) batches,
        pools and projects them, then minimises the squared Euclidean distance
        between the centroids.
        """
        self._sync_module_map(model)
        if not self.route_module_names:
            return torch.tensor(0.0, device=self.accelerator.device)

        priv_inputs = self._to_model_inputs(priv_batch, with_labels=True)
        ref_inputs = self._to_model_inputs(ref_batch, with_labels=True)

        with self._capture_module_outputs(self.route_module_names) as priv_cache:
            model(**priv_inputs)
        with self._capture_module_outputs(self.route_module_names) as ref_cache:
            model(**ref_inputs)

        if not priv_cache or not ref_cache:
            return torch.tensor(0.0, device=self.accelerator.device)

        common_names = [n for n in priv_cache.keys() if n in ref_cache]
        if not common_names:
            return torch.tensor(0.0, device=self.accelerator.device)

        gates = self._route_importance_gate(common_names, device=self.accelerator.device)
        total_loss = torch.tensor(0.0, device=self.accelerator.device)

        for i, name in enumerate(common_names):
            priv_pooled = self._mean_hidden_vector(priv_cache[name])
            ref_pooled = self._mean_hidden_vector(ref_cache[name])
            priv_proj = self._project_hidden_vector(model, priv_pooled)
            ref_proj = self._project_hidden_vector(model, ref_pooled)

            # Centroid (mean over batch) of projected representations
            priv_centroid = priv_proj.mean(dim=0)
            ref_centroid = ref_proj.mean(dim=0)

            # Squared Euclidean distance between centroids, gated by importance
            dist_sq = (priv_centroid - ref_centroid).pow(2).sum()
            total_loss = total_loss + gates[i] * dist_sq

        if gates.sum() > self.cas_eps:
            total_loss = total_loss / gates.sum()
        return total_loss

    def _compute_non_route_activation_contrast(
        self,
        model,
        priv_inputs_no_labels: Dict[str, torch.Tensor],
        safe_inputs_no_labels: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if safe_inputs_no_labels is None:
            return torch.tensor(0.0, device=self.accelerator.device)

        non_route_modules = [name for name in self.candidate_module_names if name not in self.route_module_names]
        if not non_route_modules:
            return torch.tensor(0.0, device=self.accelerator.device)

        with self._capture_module_outputs(non_route_modules) as priv_cache:
            model(**priv_inputs_no_labels)
        with self._capture_module_outputs(non_route_modules) as safe_cache:
            model(**safe_inputs_no_labels)

        if not priv_cache or not safe_cache:
            return torch.tensor(0.0, device=self.accelerator.device)

        common_names = [name for name in priv_cache.keys() if name in safe_cache]
        if not common_names:
            return torch.tensor(0.0, device=self.accelerator.device)

        diffs = []
        for name in common_names:
            priv_score = self._mean_activation_norm(priv_cache[name])
            safe_score = self._mean_activation_norm(safe_cache[name])
            diffs.append(torch.relu(priv_score - safe_score - self.cas_locality_tau))

        return torch.stack(diffs).mean()

    def _next_token_nll(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        if logits.ndim != 3:
            return torch.tensor(0.0, device=logits.device)
        if input_ids.ndim != 2:
            return torch.tensor(0.0, device=logits.device)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_mask)
        denom = shift_mask.sum().clamp_min(1.0)
        return (token_ce * shift_mask).sum() / denom

    @torch.no_grad()
    def _patching_lite_scores(
        self,
        model,
        priv_inputs_no_labels: Dict[str, torch.Tensor],
        safe_inputs_no_labels: Dict[str, torch.Tensor],
        base_activation_scores: Dict[str, float],
    ) -> Dict[str, float]:
        if not self.candidate_module_names:
            return {}

        ranked_by_activation = sorted(base_activation_scores.items(), key=lambda x: x[1], reverse=True)
        top_modules = [
            name for name, _ in ranked_by_activation[: min(self.cas_patching_topm, len(ranked_by_activation))]
        ]
        if not top_modules:
            return {}

        with self._capture_module_outputs(top_modules) as safe_cache:
            model(**safe_inputs_no_labels)
        if not safe_cache:
            return {}

        base_outputs = model(**priv_inputs_no_labels)
        base_logits = getattr(base_outputs, "logits", None)
        if base_logits is None:
            return {}
        base_nll = self._next_token_nll(
            logits=base_logits,
            input_ids=priv_inputs_no_labels["input_ids"],
            attention_mask=priv_inputs_no_labels["attention_mask"],
        )

        patch_scores: Dict[str, float] = {}
        for module_name in top_modules:
            module = self._module_name_to_module.get(module_name)
            replacement = safe_cache.get(module_name)
            if module is None or replacement is None:
                continue

            def _patch_hook(_, __, output):
                current = _module_output_to_tensor(output)
                if current is None:
                    return output
                if current.shape != replacement.shape:
                    return output
                rep = replacement.to(device=current.device, dtype=current.dtype)
                return self._replace_module_output(output, rep)

            hook = module.register_forward_hook(_patch_hook)
            try:
                patched_outputs = model(**priv_inputs_no_labels)
                patched_logits = getattr(patched_outputs, "logits", None)
                if patched_logits is None:
                    continue
                patched_nll = self._next_token_nll(
                    logits=patched_logits,
                    input_ids=priv_inputs_no_labels["input_ids"],
                    attention_mask=priv_inputs_no_labels["attention_mask"],
                )
                patch_scores[module_name] = float((patched_nll - base_nll).item())
            finally:
                hook.remove()

        return patch_scores

    @torch.no_grad()
    def _discover_route(
        self,
        model,
        priv_inputs_no_labels: Dict[str, torch.Tensor],
        safe_inputs_no_labels: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self._sync_module_map(model)
        if not self.candidate_module_names:
            self.candidate_module_names = self._build_candidate_module_names()
        if not self.candidate_module_names:
            return

        with self._capture_module_outputs(self.candidate_module_names) as priv_cache:
            model(**priv_inputs_no_labels)

        priv_scores = {name: self._mean_activation_norm(out).item() for name, out in priv_cache.items()}

        safe_scores = {}
        if safe_inputs_no_labels is not None:
            with self._capture_module_outputs(self.candidate_module_names) as safe_cache:
                model(**safe_inputs_no_labels)
            safe_scores = {name: self._mean_activation_norm(out).item() for name, out in safe_cache.items()}

        patch_scores = {}
        if (
            self.cas_route_discovery_mode.lower() in {"patching", "hybrid"}
            and safe_inputs_no_labels is not None
            and self.cas_patching_score_weight > 0.0
        ):
            patch_scores = self._patching_lite_scores(
                model=model,
                priv_inputs_no_labels=priv_inputs_no_labels,
                safe_inputs_no_labels=safe_inputs_no_labels,
                base_activation_scores={
                    k: priv_scores.get(k, 0.0) - safe_scores.get(k, 0.0) for k in self.candidate_module_names
                },
            )

        for name in self.candidate_module_names:
            # Raw private activation a_i^- (EMA-smoothed, for high_act variant)
            raw_priv = priv_scores.get(name, 0.0)
            old_priv = self.module_priv_ema[name]
            self.module_priv_ema[name] = self.route_ema_decay * old_priv + (1.0 - self.route_ema_decay) * raw_priv

            # Contrastive score s_i = a_i^- - a_i^+ (EMA-smoothed, for cascade/low_score)
            score = raw_priv
            if safe_scores:
                score = score - safe_scores.get(name, 0.0)
            if patch_scores:
                score = score + self.cas_patching_score_weight * patch_scores.get(name, 0.0)
            old_val = self.module_importance_ema[name]
            self.module_importance_ema[name] = self.route_ema_decay * old_val + (1.0 - self.route_ema_decay) * score

        # --- Route selection based on route_selection_mode ---
        mode = self.route_selection_mode.lower()
        if mode == "random":
            names = list(self.candidate_module_names)
            self._route_random.shuffle(names)
            selected = names[: self.topk_modules]
        elif mode == "high_act":
            ranked = sorted(self.module_priv_ema.items(), key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in ranked[: self.topk_modules]]
        elif mode == "low_score":
            ranked = sorted(self.module_importance_ema.items(), key=lambda x: x[1], reverse=False)
            selected = [name for name, _ in ranked[: self.topk_modules]]
        else:
            # Default: cascade (top-k by contrastive EMA score descending)
            ranked = sorted(self.module_importance_ema.items(), key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in ranked[: self.topk_modules]]

        self.route_module_names = selected

    def _compute_route_activation(self, model, model_inputs_with_labels):
        self._sync_module_map(model)
        if not self.route_module_names:
            return torch.tensor(0.0, device=self.accelerator.device), None

        with self._capture_module_outputs(self.route_module_names) as route_cache:
            outputs = model(**model_inputs_with_labels)

        if not route_cache:
            return torch.tensor(0.0, device=self.accelerator.device), outputs

        route_activation = torch.stack([self._mean_activation_norm(out) for out in route_cache.values()]).sum()

        if self.normalize_route_loss:
            route_activation = route_activation / max(1, len(route_cache))

        return route_activation, outputs

    def _compute_route_hyperbolic_stats(self, model, model_inputs_with_labels):
        self._sync_module_map(model)
        if not self.route_module_names:
            zero = torch.tensor(0.0, device=self.accelerator.device)
            return zero, zero, None

        with self._capture_module_outputs(self.route_module_names) as route_cache:
            outputs = model(**model_inputs_with_labels)

        if not route_cache:
            zero = torch.tensor(0.0, device=self.accelerator.device)
            return zero, zero, outputs

        radius = self._route_hyperbolic_radius(model, route_cache)
        compress_signal = self._route_weighted_compression_signal(model, route_cache)
        return radius, compress_signal, outputs

    def _cas_has_active_terms(self) -> bool:
        return self.cas_hierarchy_weight > 0.0 or self.cas_locality_weight > 0.0 or self.cas_compress_weight > 0.0

    def _direct_forget_active(self) -> bool:
        return (
            self.cas_direct_forget_weight > 0.0
            or self.cas_direct_safe_weight > 0.0
            or self.cas_direct_pair_weight > 0.0
        )

    def _decay_direct_safe_weight(self, weight: float, global_step: int) -> float:
        if weight <= 0.0:
            return 0.0
        start = self.cas_direct_safe_decay_start_step
        if start < 0 or global_step < start:
            return weight
        target = self.cas_direct_safe_min_weight if self.cas_direct_safe_min_weight is not None else 0.0
        target = max(0.0, float(target))
        duration = self.cas_direct_safe_decay_duration
        if duration <= 0:
            return target
        progress = min(1.0, float(global_step - start) / float(max(1, duration)))
        return weight + (target - weight) * progress

    def _pairwise_direct_loss(
        self,
        priv_nll: torch.Tensor,
        safe_nll: torch.Tensor,
    ) -> torch.Tensor:
        # Desired relation: priv_nll >= safe_nll + margin.
        violation = self.cas_direct_margin - (priv_nll - safe_nll)
        if self.cas_direct_use_softplus:
            return F.softplus(violation)
        return F.relu(violation)

    def _compute_direct_forget_losses(
        self,
        priv_outputs,
        safe_outputs=None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.tensor(0.0, device=self.accelerator.device)
        if not self._direct_forget_active() or priv_outputs is None:
            return zero, {
                "cas_direct_loss": 0.0,
                "cas_direct_priv_nll": 0.0,
                "cas_direct_safe_nll": 0.0,
                "cas_direct_pair_loss": 0.0,
                "cas_direct_weight_scale": 0.0,
                "cas_direct_forget_weight_eff": 0.0,
                "cas_direct_safe_weight_eff": 0.0,
                "cas_direct_pair_weight_eff": 0.0,
            }

        priv_nll = getattr(priv_outputs, "loss", None)
        if priv_nll is None:
            return zero, {
                "cas_direct_loss": 0.0,
                "cas_direct_priv_nll": 0.0,
                "cas_direct_safe_nll": 0.0,
                "cas_direct_pair_loss": 0.0,
                "cas_direct_weight_scale": 0.0,
                "cas_direct_forget_weight_eff": 0.0,
                "cas_direct_safe_weight_eff": 0.0,
                "cas_direct_pair_weight_eff": 0.0,
            }

        if self.cas_direct_warmup_steps > 0:
            direct_weight_scale = min(1.0, float(global_step + 1) / float(self.cas_direct_warmup_steps))
        else:
            direct_weight_scale = 1.0
        forget_w_eff = self.cas_direct_forget_weight * direct_weight_scale
        safe_w_eff = self.cas_direct_safe_weight * direct_weight_scale
        safe_w_eff = self._decay_direct_safe_weight(
            weight=safe_w_eff,
            global_step=global_step,
        )
        pair_w_eff = self.cas_direct_pair_weight * direct_weight_scale

        # Increase private answer NLL (forget) while optionally reducing safe-answer NLL.
        direct_loss = -forget_w_eff * priv_nll
        pair_loss = zero
        safe_nll_val = 0.0

        if safe_outputs is not None and getattr(safe_outputs, "loss", None) is not None:
            safe_nll = safe_outputs.loss
            safe_nll_val = float(safe_nll.detach().item())
            direct_loss = direct_loss + safe_w_eff * safe_nll
            pair_loss = self._pairwise_direct_loss(priv_nll=priv_nll, safe_nll=safe_nll)
            direct_loss = direct_loss + pair_w_eff * pair_loss

        logs = {
            "cas_direct_loss": float(direct_loss.detach().item()),
            "cas_direct_priv_nll": float(priv_nll.detach().item()),
            "cas_direct_safe_nll": safe_nll_val,
            "cas_direct_pair_loss": float(pair_loss.detach().item()),
            "cas_direct_weight_scale": float(direct_weight_scale),
            "cas_direct_forget_weight_eff": float(forget_w_eff),
            "cas_direct_safe_weight_eff": float(safe_w_eff),
            "cas_direct_pair_weight_eff": float(pair_w_eff),
        }
        return direct_loss, logs

    def _compute_cas_aux_losses(
        self,
        model,
        inputs,
        priv_batch: Dict[str, torch.Tensor],
        safe_batch: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        zero = torch.tensor(0.0, device=self.accelerator.device)
        if not self._cas_has_active_terms():
            return (
                zero,
                zero,
                zero,
                {
                    "cas_route_radius_priv": 0.0,
                    "cas_route_radius_safe": 0.0,
                    "cas_route_radius_retain": 0.0,
                    "cas_hierarchy_loss": 0.0,
                    "cas_locality_loss": 0.0,
                    "cas_compress_loss": 0.0,
                    "cas_euc_contrast_loss": 0.0,
                },
            )

        # --- Euclidean contrast mode: replace hierarchy+compress with centroid distance ---
        if self.cas_repr_mode == "euclidean_contrast" and self.cas_compress_weight > 0.0:
            ref_batch = None
            if safe_batch is not None:
                ref_batch = safe_batch
            elif "retain" in inputs:
                ref_batch = inputs["retain"]

            contrast_loss = zero
            if ref_batch is not None:
                contrast_loss = self._euclidean_contrast_loss(
                    model=model,
                    priv_batch=priv_batch,
                    ref_batch=ref_batch,
                )

            # Still compute locality loss if active
            locality_loss = zero
            if self.cas_locality_weight > 0.0:
                locality_loss = self._compute_non_route_activation_contrast(
                    model=model,
                    priv_inputs_no_labels=self._to_model_inputs(priv_batch, with_labels=False),
                    safe_inputs_no_labels=(
                        self._to_model_inputs(safe_batch, with_labels=False) if safe_batch is not None else None
                    ),
                )

            cas_logs = {
                "cas_route_radius_priv": 0.0,
                "cas_route_radius_safe": 0.0,
                "cas_route_radius_retain": 0.0,
                "cas_hierarchy_loss": 0.0,
                "cas_locality_loss": float(locality_loss.detach().item()),
                "cas_compress_loss": float(contrast_loss.detach().item()),
                "cas_euc_contrast_loss": float(contrast_loss.detach().item()),
            }
            # Return contrast_loss as compress_loss so it gets weighted by cas_compress_weight
            return zero, locality_loss, contrast_loss, cas_logs

        # --- Standard / hyperbolic / euclidean_norm mode ---
        needs_hyper_radius = self.cas_hierarchy_weight > 0.0 or self.cas_compress_weight > 0.0

        priv_radius = zero
        priv_compress_signal = zero
        if needs_hyper_radius:
            priv_inputs = self._to_model_inputs(priv_batch, with_labels=True)
            priv_radius, priv_compress_signal, _ = self._compute_route_hyperbolic_stats(
                model=model,
                model_inputs_with_labels=priv_inputs,
            )

        safe_radius = None
        if self.cas_hierarchy_weight > 0.0 and safe_batch is not None:
            safe_inputs = self._to_model_inputs(safe_batch, with_labels=True)
            safe_radius, _, _ = self._compute_route_hyperbolic_stats(
                model=model,
                model_inputs_with_labels=safe_inputs,
            )

        retain_radius = None
        if self.cas_hierarchy_weight > 0.0 and self.cas_hierarchy_use_retain and "retain" in inputs:
            retain_inputs = self._to_model_inputs(inputs["retain"], with_labels=True)
            retain_radius, _, _ = self._compute_route_hyperbolic_stats(
                model=model,
                model_inputs_with_labels=retain_inputs,
            )

        hierarchy_loss = zero
        if self.cas_hierarchy_weight > 0.0 and safe_radius is not None:
            hierarchy_loss = hierarchy_loss + F.relu(self.cas_hierarchy_margin - (priv_radius - safe_radius))
        if self.cas_hierarchy_weight > 0.0 and retain_radius is not None:
            hierarchy_loss = hierarchy_loss + F.relu(self.cas_hierarchy_margin - (priv_radius - retain_radius))

        compress_loss = zero
        if self.cas_compress_weight > 0.0:
            compress_loss = priv_compress_signal.pow(2)

        locality_loss = zero
        if self.cas_locality_weight > 0.0:
            locality_loss = self._compute_non_route_activation_contrast(
                model=model,
                priv_inputs_no_labels=self._to_model_inputs(priv_batch, with_labels=False),
                safe_inputs_no_labels=(
                    self._to_model_inputs(safe_batch, with_labels=False) if safe_batch is not None else None
                ),
            )

        cas_logs = {
            "cas_route_radius_priv": float(priv_radius.detach().item()),
            "cas_route_radius_safe": float(safe_radius.detach().item() if safe_radius is not None else 0.0),
            "cas_route_radius_retain": float(retain_radius.detach().item() if retain_radius is not None else 0.0),
            "cas_hierarchy_loss": float(hierarchy_loss.detach().item()),
            "cas_locality_loss": float(locality_loss.detach().item()),
            "cas_compress_loss": float(compress_loss.detach().item()),
            "cas_euc_contrast_loss": 0.0,
        }
        return hierarchy_loss, locality_loss, compress_loss, cas_logs

    def _compute_forget_objective(
        self,
        priv_route_activation: torch.Tensor,
        safe_route_activation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.contrastive_forget and safe_route_activation is not None:
            # Suppress the private route relative to the safe route, rather than
            # globally shrinking all route activations.
            raw_signal = priv_route_activation - safe_route_activation - self.contrastive_margin
        else:
            raw_signal = priv_route_activation

        if self.dynamic_threshold:
            shifted = raw_signal - self.threshold_tau
        else:
            shifted = raw_signal

        if self.cru_forget_use_softplus:
            route_loss = F.softplus(shifted, beta=self.cru_forget_softplus_beta)
            if self.dynamic_threshold and self.threshold_power != 1.0:
                route_loss = route_loss.pow(self.threshold_power)
        else:
            if self.dynamic_threshold:
                route_loss = torch.relu(shifted).pow(self.threshold_power)
            else:
                route_loss = torch.relu(shifted)

        return route_loss

    def _effective_lambda_forget(self, global_step: int) -> float:
        effective = float(self.lambda_forget)
        if self.lambda_warmup_steps > 0:
            warmup_ratio = min(1.0, float(global_step + 1) / float(self.lambda_warmup_steps))
            effective = effective * warmup_ratio
        if (
            self.lambda_forget_boost_start_step >= 0
            and global_step >= self.lambda_forget_boost_start_step
            and self.lambda_forget_boost_factor != 1.0
        ):
            effective = effective * self.lambda_forget_boost_factor
        return effective

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_batch = inputs["forget"]
        priv_batch, safe_batch = self._extract_priv_safe_pair(forget_batch)

        global_step = int(self.state.global_step)
        bank_route = self._lookup_route_from_bank(priv_batch)
        used_route_bank = bool(bank_route)
        if bank_route:
            self.route_module_names = bank_route
        should_update_route = ((not bank_route) and global_step % max(1, self.route_update_interval) == 0) or (
            len(self.route_module_names) == 0
        )
        discovered_route_this_step = bool(should_update_route)
        if should_update_route:
            self._discover_route(
                model=model,
                priv_inputs_no_labels=self._to_model_inputs(priv_batch, with_labels=False),
                safe_inputs_no_labels=(
                    self._to_model_inputs(safe_batch, with_labels=False) if safe_batch is not None else None
                ),
            )
            self._update_route_bank(priv_batch)

        priv_inputs = self._to_model_inputs(priv_batch, with_labels=True)
        priv_route_activation, forget_outputs = self._compute_route_activation(
            model=model,
            model_inputs_with_labels=priv_inputs,
        )

        safe_route_activation = None
        safe_outputs = None
        if safe_batch is not None:
            safe_inputs = self._to_model_inputs(safe_batch, with_labels=True)
            safe_route_activation, safe_outputs = self._compute_route_activation(
                model=model,
                model_inputs_with_labels=safe_inputs,
            )

        forget_loss = self._compute_forget_objective(
            priv_route_activation=priv_route_activation,
            safe_route_activation=safe_route_activation,
        )

        if "retain" in inputs:
            retain_inputs = self._to_model_inputs(inputs["retain"], with_labels=True)
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        else:
            retain_loss = torch.tensor(0.0, device=forget_loss.device)

        effective_lambda_forget = self._effective_lambda_forget(global_step)
        total_loss = effective_lambda_forget * forget_loss + self.alpha * retain_loss

        hierarchy_loss, locality_loss, compress_loss, cas_logs = self._compute_cas_aux_losses(
            model=model,
            inputs=inputs,
            priv_batch=priv_batch,
            safe_batch=safe_batch,
        )
        total_loss = (
            total_loss
            + self.cas_hierarchy_weight * hierarchy_loss
            + self.cas_locality_weight * locality_loss
            + self.cas_compress_weight * compress_loss
        )

        direct_forget_loss, direct_forget_logs = self._compute_direct_forget_losses(
            priv_outputs=forget_outputs,
            safe_outputs=safe_outputs,
            global_step=global_step,
        )
        total_loss = total_loss + direct_forget_loss

        weighted_hierarchy_loss = self.cas_hierarchy_weight * hierarchy_loss
        weighted_locality_loss = self.cas_locality_weight * locality_loss
        weighted_compress_loss = self.cas_compress_weight * compress_loss
        cas_terms_active = float(1.0 if self._cas_has_active_terms() else 0.0)
        discovery_mode = self.cas_route_discovery_mode.lower()

        self.log(
            {
                "cru_route_modules": float(len(self.route_module_names)),
                "cru_route_activation_priv": float(priv_route_activation.detach().item()),
                "cru_route_activation_safe": float(
                    safe_route_activation.detach().item() if safe_route_activation is not None else 0.0
                ),
                "cru_forget_loss": float(forget_loss.detach().item()),
                "cru_retain_loss": float(retain_loss.detach().item()),
                "cru_lambda_forget_eff": float(effective_lambda_forget),
                "cas_w_hierarchy": float(self.cas_hierarchy_weight),
                "cas_w_locality": float(self.cas_locality_weight),
                "cas_w_compress": float(self.cas_compress_weight),
                "cas_repr_mode": self.cas_repr_mode,
                "cas_route_selection_mode": self.route_selection_mode,
                "cas_terms_active": cas_terms_active,
                "cas_weighted_hierarchy_loss": float(weighted_hierarchy_loss.detach().item()),
                "cas_weighted_locality_loss": float(weighted_locality_loss.detach().item()),
                "cas_weighted_compress_loss": float(weighted_compress_loss.detach().item()),
                "cas_route_bank_size": float(len(self.route_bank)),
                "cas_route_bank_enable": float(1.0 if self.cas_route_bank_enable else 0.0),
                "cas_route_bank_hit": float(1.0 if self._last_route_bank_hit else 0.0),
                "cas_route_source_bank": float(1.0 if used_route_bank else 0.0),
                "cas_route_discovered_step": float(1.0 if discovered_route_this_step else 0.0),
                "cas_discovery_activation": float(1.0 if discovery_mode == "activation" else 0.0),
                "cas_discovery_patching": float(1.0 if discovery_mode == "patching" else 0.0),
                "cas_discovery_hybrid": float(1.0 if discovery_mode == "hybrid" else 0.0),
                "cas_patching_topm": float(self.cas_patching_topm),
                "cas_patching_score_weight": float(self.cas_patching_score_weight),
                "cas_direct_forget_weight": float(self.cas_direct_forget_weight),
                "cas_direct_safe_weight": float(self.cas_direct_safe_weight),
                "cas_direct_pair_weight": float(self.cas_direct_pair_weight),
                "cas_direct_margin": float(self.cas_direct_margin),
                "cas_freeze_projection_head": float(1.0 if self.cas_freeze_projection_head else 0.0),
                "cru_forget_use_softplus": float(1.0 if self.cru_forget_use_softplus else 0.0),
                "cru_forget_softplus_beta": float(self.cru_forget_softplus_beta),
                **cas_logs,
                **direct_forget_logs,
            }
        )

        return (total_loss, forget_outputs) if return_outputs else total_loss
