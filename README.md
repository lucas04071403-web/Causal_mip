# MIP-Editor

MIP-Editor is a multimodal unlearning project for MLLMs. It keeps the original MIP-Editor baseline pipeline and includes the current causal selective editing extension used in this workspace.

![MIP-Editor](pictures/mainfig.png)

This README is the project-level running guide. Detailed implementation notes, ablations, and Science Q1 experiment records are in `docs/`.

## 1. Project Scope

This repository supports two project-level modes:

1. **Original MIP-Editor baseline**
   - Entry: `main.py`
   - Core files: `ours.py`, `fisher.py`, `ig.py`, `unlearning.py`, `manu.py`
   - Purpose: run the original MIP-Editor / baseline unlearning methods.

2. **Causal selective editing extension**
   - Entry: `main.py --use_masked_rmisu` plus `causal_mip/evaluation/*`
   - Core directory: `causal_mip/`
   - Purpose: run the completed causal path classification + masked RMisU editing + Step8 evaluation workflow.

For internal Step2-Step8 implementation details, use:

```text
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
docs/PROJECT_STRUCTURE.md
```

## 2. Environment

Activate the project environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mip-editor
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Check CUDA:

```bash
python - <<'PY'
import sys, torch
print(sys.executable)
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Large-model runs should use `mip-editor`, not the base conda environment.

## 3. Workspace

The repository is:

```text
MIP-Editor/
```

The experiment workspace is outside the repository:

```text
../mip_workspace/
├── datasets/
├── llms/
├── influential_paths/
└── outputs/
```

`causal_mip/project_paths.py` resolves this workspace automatically. To override:

```bash
export MIP_WORKSPACE_ROOT=/absolute/path/to/mip_workspace
```

Expected local resources:

```text
../mip_workspace/llms/Qwen2.5-VL-3B-Instruct/
../mip_workspace/datasets/CLEAR/
../mip_workspace/datasets/MLLMU-Bench/
../mip_workspace/influential_paths/
../mip_workspace/outputs/model_caches/
```

Typical PEFT baseline adapter:

```text
../mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
```

## 4. Run Original Baseline

Evaluate the PEFT baseline:

```bash
PYTHONPATH=. python main.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --forget_ratio 5 \
  --batch_size 2 \
  --ptm_ckpt_batch_size 2 \
  --eval_flag \
  --unlearning "" \
  --device cuda
```

Run original MIP-Editor:

```bash
PYTHONPATH=. python main.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --forget_ratio 5 \
  --batch_size 2 \
  --ptm_ckpt_batch_size 2 \
  --unlearning our \
  --use_neuron_cache_flag \
  --device cuda
```

Other supported baseline methods:

```text
ga
kl
npo
manu
our
```

## 5. Run Causal Selective Editing

The completed causal selective editing run uses classified candidate paths as input to masked RMisU. The current project-level entry is `main.py --use_masked_rmisu`.

Example small diagnostic run:

```bash
PYTHONPATH=. python main.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --forget_ratio 5 \
  --batch_size 2 \
  --ptm_ckpt_batch_size 2 \
  --unlearning our \
  --use_masked_rmisu \
  --masked_rmisu_candidate_paths ../mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl \
  --masked_rmisu_p_forget ../mip_workspace/outputs/paths/P_forget_stable_retainaware_node_projdim_20pairs_0529.jsonl \
  --masked_rmisu_p_shared ../mip_workspace/outputs/paths/P_shared_stable_retainaware_node_projdim_20pairs_0529.jsonl \
  --masked_rmisu_forget_objective name_ce_ascent \
  --masked_rmisu_forget_ce_alpha 0.2 \
  --masked_rmisu_target_ce_scope name \
  --masked_rmisu_max_steps 5 \
  --masked_rmisu_output ../mip_workspace/outputs/masked_rmisu_smoke.json \
  --masked_rmisu_checkpoint_dir ../mip_workspace/outputs/checkpoints/masked_rmisu_smoke \
  --skip_post_unlearning_eval \
  --device cuda
```

Main outputs:

```text
../mip_workspace/outputs/masked_rmisu_smoke.json
../mip_workspace/outputs/checkpoints/masked_rmisu_smoke/
```

Use the Science Q1 docs before scaling this run. The latest evidence says single-candidate smoke is diagnostic only, not a successful unlearning claim.

## 6. Evaluate A Run

Pair-level Step8 evaluation:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_final_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_smoke \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_val.jsonl \
  --output ../mip_workspace/outputs/step8_eval_masked_rmisu_smoke.json \
  --split val \
  --device cuda \
  --max_new_tokens 80
```

Probability diagnostic for cases where name-hit is already uninformative:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_probability_diagnostic.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --baseline_peft_checkpoint ../mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth \
  --candidate_peft_checkpoint ../mip_workspace/outputs/checkpoints/step7_single_stable_smoke_retainaware_0529_152017 \
  --pair_jsonl ../mip_workspace/outputs/diagnostics/causal_pair_000089_step8_diagnostic_0529.jsonl \
  --output ../mip_workspace/outputs/diagnostics/step8_probability_diag_pair000089_baseline_vs_smoke_with_generated_0529.json \
  --split diagnostic \
  --device cuda \
  --generated_name_source self
```

Full CLEAR remote evaluation:

```bash
PYTHONPATH=. python causal_mip/evaluation/full_clear_remote_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_smoke \
  --base_path ../mip_workspace/datasets \
  --llm_directory ../mip_workspace/llms \
  --output_dir ../mip_workspace/outputs/full_clear_remote_eval \
  --run_id masked_rmisu_smoke_full_remote \
  --forget_ratio 5 \
  --device cuda \
  --remote_scoring_url http://localhost:9192/v1 \
  --score_llm Qwen2.5-VL-7B-Instruct
```

Combine pair eval and Full CLEAR summary:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_protocol.py \
  --pair_eval ../mip_workspace/outputs/step8_eval_masked_rmisu_smoke.json \
  --pair_baseline ../mip_workspace/outputs/step8_eval_peft_baseline.json \
  --full_clear_summary ../mip_workspace/outputs/full_clear_remote_eval/masked_rmisu_smoke_full_remote/full_clear_remote_protocol_summary.json \
  --full_clear_baseline ../mip_workspace/outputs/full_clear_remote_eval/peft_baseline_full_remote/full_clear_remote_protocol_summary.json \
  --output ../mip_workspace/outputs/step8_protocol_masked_rmisu_smoke.json
```

## 7. Tests

Run core tests:

```bash
PYTHONPATH=. python causal_mip/test_step5_causal_scores.py
PYTHONPATH=. python causal_mip/test_step6_classify_paths.py
PYTHONPATH=. python causal_mip/test_step7_masked_rmisu.py
PYTHONPATH=. python causal_mip/test_step8_protocol.py
PYTHONPATH=. python causal_mip/test_step8_probability_diagnostic.py
PYTHONPATH=. python causal_mip/test_retain_aware_localization.py
```

Compile-check key modules:

```bash
python -m py_compile \
  causal_mip/causal_scores/build_scores.py \
  causal_mip/causal_scores/classify_paths.py \
  causal_mip/causal_scores/saliency_specificity.py \
  causal_mip/causal_scores/stability_report.py \
  causal_mip/evaluation/step8_probability_diagnostic.py \
  causal_mip/editing/masked_rmisu.py
```

## 8. Outputs

Common output directories:

```text
../mip_workspace/outputs/paths/
../mip_workspace/outputs/scores/
../mip_workspace/outputs/checkpoints/
../mip_workspace/outputs/diagnostics/
../mip_workspace/outputs/full_clear_remote_eval/
../mip_workspace/outputs/logs/
```

Recent project-level artifacts:

```text
../mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl
../mip_workspace/outputs/paths/P_forget_stable_retainaware_node_projdim_20pairs_0529.jsonl
../mip_workspace/outputs/checkpoints/step7_single_stable_smoke_retainaware_0529_152017/
../mip_workspace/outputs/diagnostics/step8_probability_diag_pair000089_baseline_vs_smoke_with_generated_0529.json
```

## 9. Documentation

Read these for implementation and experiment details:

```text
docs/PROJECT_STRUCTURE.md
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
docs/SCIENCE_Q1_MAIN_REPORT.md
docs/SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md
docs/SCIENCE_Q1_NEXT_STEPS.md
```

`README.md` is intentionally the completed project running guide, not the per-step implementation manual.

## 10. Citation

```bibtex
@article{
Li_Li_Wu_Yang_Bai_Jia_Xue_2026,
title={Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models},
volume={40},
url={https://ojs.aaai.org/index.php/AAAI/article/view/40870},
DOI={10.1609/aaai.v40i42.40870},
number={42},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Li, Kunhao and Li, Wenhao and Wu, Di and Yang, Lei and Bai, Jun and Jia, Ju and Xue, Jason},
year={2026},
month={Mar.},
pages={35589-35597}
}
```
