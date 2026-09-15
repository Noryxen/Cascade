# Data

## Datasets

We use three benchmarks covering controlled fact unlearning, realistic text unlearning, and safety-sensitive knowledge removal.

### TOFU
Synthetically generated fictitious author profiles. We use the **Forget10** split.

- Hugging Face: [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- Split mapping: `forget10` (removal), `retain90` (preservation), `holdout10` (privacy)
- Auxiliary: `world_facts`, `real_authors` (utility preservation)

### MUSE-News
BBC news articles for realistic text unlearning.

- Hugging Face: [muse-bench/MUSE-News](https://huggingface.co/datasets/muse-bench/MUSE-News)
- Splits: `forget` (removal), `retain` (preservation)

### WMDP
Hazardous knowledge evaluation. We use the **cybersecurity** subset (`wmdp_cyber`).

- Hugging Face: [cais/wmdp](https://huggingface.co/datasets/cais/wmdp)
- Unlearning corpus: [wmdp-corpora.zip](https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip) (password: `wmdpcorpora`)
- Evaluation: `lm-evaluation-harness` multiple-choice accuracy