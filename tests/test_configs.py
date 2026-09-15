"""Config-only regression checks; no torch, model downloads or GPU required."""

import ast
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MODELS = sorted(p.stem for p in (CONFIGS / "model").glob("*.yaml"))


def resolved(entry, overrides=(), require_task=True):
    args = (["task_name=config_check"] if require_task else []) + list(overrides)
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name=entry, overrides=args, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        # Hydra's own runtime-only fields are unavailable outside a launched job.
        with open_dict(cfg):
            del cfg["hydra"]
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def bash_path():
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            bash = Path(git).resolve().parents[1] / "bin" / "bash.exe"
            if bash.exists():
                return str(bash)
        return None
    return shutil.which("bash")


class ConfigTests(unittest.TestCase):
    def test_all_entries_models_and_evaluators_resolve(self):
        for entry in ("train", "unlearn", "eval"):
            for model in MODELS:
                for evaluator in ("tofu", "muse", "lm_eval"):
                    with self.subTest(entry=entry, model=model, evaluator=evaluator):
                        cfg = resolved(entry, [f"model={model}", f"eval={evaluator}"])
                        self.assertEqual(cfg["mode"], entry)
                        self.assertEqual(set(cfg["eval"]), {evaluator})
                        self.assertTrue(cfg["model"]["model_args"]["pretrained_model_name_or_path"].endswith(model))
                        self.assertEqual(cfg["model"]["tokenizer_args"]["padding_side"], "left")

    def test_every_optional_dataset_and_metric_resolves(self):
        for path in (CONFIGS / "data" / "datasets").glob("*.yaml"):
            with self.subTest(dataset=path.stem):
                cfg = resolved("train", [f"data/datasets@data.train={path.stem}"])
                self.assertEqual(set(cfg["data"]["train"]), {path.stem})
        for evaluator in ("tofu", "muse"):
            for path in (CONFIGS / "eval" / f"{evaluator}_metrics").glob("*.yaml"):
                with self.subTest(metric=path.stem):
                    cfg = resolved(
                        "eval",
                        [
                            f"eval={evaluator}",
                            f"+eval/{evaluator}_metrics@eval.{evaluator}.metrics.selected={path.stem}",
                        ],
                    )
                    metrics = cfg["eval"][evaluator]["metrics"]
                    self.assertIn("handler", metrics["selected"])

    def test_common_overrides_reach_nested_configs(self):
        cfg = resolved(
            "unlearn", ["seed=7", "padding_side=right", "paths.root_dir=/tmp/check", "eval.tofu.batch_size=3"]
        )
        self.assertEqual(cfg["trainer"]["args"]["seed"], 7)
        self.assertEqual(cfg["trainer"]["args"]["data_seed"], 7)
        self.assertEqual(cfg["trainer"]["args"]["output_dir"], "/tmp/check/saves/unlearn/config_check")
        for node in nodes(cfg):
            if "padding_side" in node:
                self.assertEqual(node["padding_side"], "right")
        for node in nodes(cfg["eval"]["tofu"]["metrics"]):
            if "batch_size" in node:
                self.assertEqual(node["batch_size"], 3)

    def test_split_override_reaches_training_and_nested_forget_metrics(self):
        cfg = resolved(
            "unlearn", ["dataset_defaults.tofu.forget_split=forget05", "dataset_defaults.tofu.retain_split=retain95"]
        )
        self.assertEqual(cfg["data"]["forget"]["TOFU_QA_forget"]["args"]["hf_args"]["name"], "forget05")
        self.assertEqual(cfg["data"]["retain"]["TOFU_QA_retain"]["args"]["hf_args"]["name"], "retain95")
        for node in nodes(cfg["eval"]["tofu"]["metrics"]):
            if "hf_args" in node and node["hf_args"]["name"].startswith("forget"):
                self.assertEqual(node["hf_args"]["name"], "forget05_perturbed")

    def test_training_defaults_and_intentional_exceptions(self):
        train, unlearn, evaluation = (resolved(name) for name in ("train", "unlearn", "eval"))
        for cfg in (train, unlearn, evaluation):
            self.assertEqual(cfg["model"]["model_args"]["pretrained_model_name_or_path"], "Qwen/Qwen3-1.7B")
        train_args, unlearn_args = train["trainer"]["args"], unlearn["trainer"]["args"]
        self.assertEqual(train_args.pop("num_train_epochs"), 10)
        self.assertEqual(unlearn_args.pop("num_train_epochs"), 6)
        # Output paths necessarily differ by run mode.
        for key in ("output_dir", "logging_dir"):
            train_args.pop(key)
            unlearn_args.pop(key)
        self.assertEqual(train_args, unlearn_args)
        self.assertEqual(train_args["learning_rate"], 1e-5)
        self.assertEqual(train_args["per_device_train_batch_size"], 8)
        self.assertEqual(train_args["report_to"], "tensorboard")
        muse = resolved("eval", ["eval=muse"])["eval"]["muse"]["metrics"]
        self.assertEqual(muse["forget_verbmem_ROUGE"]["generation_args"]["max_new_tokens"], 128)
        self.assertEqual(muse["retain_knowmem_ROUGE"]["generation_args"]["max_new_tokens"], 32)
        tofu = evaluation["eval"]["tofu"]["metrics"]
        self.assertEqual(tofu["forget_Q_A_ROUGE"]["generation_args"]["max_new_tokens"], 200)
        self.assertIn("balanced_unlearning_score", tofu)

    def test_cascade_constructor_defaults_match_yaml(self):
        module = ast.parse((ROOT / "src/trainer/unlearn/cru.py").read_text(encoding="utf-8"))
        cascade = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Cascade")
        init = next(n for n in cascade.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        defaults = {arg.arg: ast.literal_eval(value) for arg, value in zip(init.args.args[1:], init.args.defaults)}
        for key, value in resolved("unlearn")["trainer"]["method_args"].items():
            with self.subTest(parameter=key):
                if key == "candidate_patterns":
                    assignment = next(
                        n
                        for n in ast.walk(init)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr == key for t in n.targets)
                    )
                    self.assertEqual(ast.literal_eval(assignment.value.values[1]), value)
                else:
                    self.assertEqual(defaults[key], value)

    def test_task_name_is_required(self):
        for entry in ("train", "unlearn", "eval"):
            with (
                self.subTest(entry=entry),
                self.assertRaisesRegex((MissingMandatoryValue, InterpolationToMissingValueError), "task_name"),
            ):
                resolved(entry, require_task=False)

    def test_system_prompt_override_reaches_both_model_families(self):
        for model in ("Qwen3-1.7B", "Llama-3.2-3B-Instruct"):
            cfg = resolved("eval", [f"model={model}", "model.template_args.system_prompt=Test prompt"])
            self.assertIn("Test prompt", cfg["model"]["template_args"]["system_prompt_with_special_tokens"])


@unittest.skipUnless(bash_path(), "Bash is needed to verify launch scripts")
class ScriptTests(unittest.TestCase):
    def capture(self, script, args=(), overrides=None):
        env = os.environ.copy()
        for name in ("MODEL", "TASK_NAME", "FORGET_SPLIT", "RETAIN_SPLIT"):
            env.pop(name, None)
        env.update(overrides or {})
        # Export a fake Python function, including into nested wrapper scripts.
        # Printing NUL-separated args preserves spaces without loading model code.
        command = 'python() { printf "%s\\0" "$@"; }; export -f python; bash "$@"'
        proc = subprocess.run(
            [bash_path(), "-c", command, "--", f"scripts/{script}.sh", *args],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
        )
        values = proc.stdout.decode().rstrip("\0").split("\0")
        entry = next(v.split("=", 1)[1] for v in values if v.startswith("--config-name="))
        return resolved(entry, [v for v in values if "=" in v and not v.startswith("--")])

    def test_script_syntax(self):
        for path in (ROOT / "scripts").glob("*.sh"):
            subprocess.run([bash_path(), "-n", str(path)], check=True, capture_output=True)

    def test_cascade_dataset_branches_and_explicit_overrides(self):
        for dataset, evaluator in (("tofu", "tofu"), ("muse", "muse"), ("wmdp", "lm_eval")):
            with self.subTest(dataset=dataset):
                cfg = self.capture("run_cascade", [dataset])
                self.assertEqual(set(cfg["eval"]), {evaluator})
                self.assertEqual(cfg["model"]["model_args"]["pretrained_model_name_or_path"], "Qwen/Qwen3-1.7B")
        cfg = self.capture(
            "run_cascade",
            ["tofu", "trainer.args.learning_rate=2e-5"],
            {"MODEL": "Qwen3-4B", "FORGET_SPLIT": "forget05", "RETAIN_SPLIT": "retain95"},
        )
        self.assertEqual(cfg["data"]["forget"]["TOFU_QA_forget"]["args"]["hf_args"]["name"], "forget05")
        self.assertEqual(cfg["data"]["retain"]["TOFU_QA_retain"]["args"]["hf_args"]["name"], "retain95")
        self.assertEqual(cfg["eval"]["tofu"]["forget_split"], "forget05")
        self.assertEqual(cfg["trainer"]["args"]["learning_rate"], 2e-5)

    def test_finetuning_wrappers_use_yaml_defaults(self):
        for script in ("run_original", "run_retrained"):
            for dataset in ("tofu", "muse"):
                with self.subTest(script=script, dataset=dataset):
                    cfg = self.capture(script, [dataset])
                    self.assertEqual(set(cfg["eval"]), {dataset})
                    self.assertEqual(cfg["trainer"]["args"]["learning_rate"], 1e-5)
                    self.assertEqual(cfg["trainer"]["args"]["per_device_train_batch_size"], 8)
                    self.assertTrue(cfg["trainer"]["args"]["do_eval"])
        cfg = self.capture("run_original", ["tofu", "Llama-3.2-1B-Instruct", "trainer.args.do_eval=false"])
        self.assertFalse(cfg["trainer"]["args"]["do_eval"])
        self.assertEqual(
            cfg["model"]["model_args"]["pretrained_model_name_or_path"], "meta-llama/Llama-3.2-1B-Instruct"
        )

    def test_eval_optional_arguments_and_paths_with_spaces(self):
        cfg = self.capture("run_eval", ["/tmp/model path", "eval_check"])
        self.assertEqual(cfg["model"]["model_args"]["pretrained_model_name_or_path"], "/tmp/model path")
        self.assertEqual(cfg["eval"]["tofu"]["forget_split"], "forget10")
        self.assertIsNone(cfg["eval"]["tofu"]["retain_logs_path"])
        cfg = self.capture("run_eval", ["/tmp/model", "eval_check", "forget05", "holdout05", "/tmp/retain logs.json"])
        self.assertEqual(cfg["eval"]["tofu"]["forget_split"], "forget05")
        self.assertEqual(cfg["eval"]["tofu"]["holdout_split"], "holdout05")
        self.assertEqual(cfg["eval"]["tofu"]["retain_logs_path"], "/tmp/retain logs.json")


if __name__ == "__main__":
    unittest.main()
