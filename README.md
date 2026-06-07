# MIP-Editor

MIP-Editor is used in this workspace for multimodal unlearning experiments on Qwen2.5-VL with a causal selective editing extension.

This README is a short running guide. Detailed project structure and experiment analysis are kept in `docs/`.

## Current Successful Run

The current successful candidate is H5:

```text
H5 = H1 retain-safe base + p000006 top15 small add-back
objective = pii_name_token_noise
forget_ce_alpha = 0.40
pii_noise_alpha = 0.08
max_steps = 200
projector_edit_mode = qwen_merger_mlp
target_ce_scope = name
```

Main checkpoint:

```text
../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603/
```

H5 passed the local Step8 cheap gate, stayed stable on a larger split, and completed Full CLEAR remote evaluation. The result is a small but stable improvement over baseline: forget metrics improve slightly while retain metrics remain almost unchanged.

## Key Results

Step8 val cheap gate:

| Metric | Gate | H5 |
| --- | ---: | ---: |
| forget_clean target_name_ce delta | > +0.01 | +0.011417 |
| forget_clean name_hit | <= 0.6111 | 0.5556 |
| hard_retain name_hit | >= 0.6444 | 0.6944 |
| counterfactual_retain name_hit | >= 0.6722 | 0.7222 |

Larger split checks:

| Metric | H5 delta |
| --- | ---: |
| forget_clean target_name_ce | +0.010743 |
| forget_clean name_hit vs baseline | -0.029412 |
| hard_retain name_hit vs baseline | 0.000000 |
| hard_retain::same_topic name_hit vs baseline | 0.000000 |
| counterfactual_retain name_hit vs baseline | -0.017647 |

Full CLEAR remote eval:

| Metric | baseline | H5 | delta |
| --- | ---: | ---: | ---: |
| forget_classification_remote_acc | 0.686170 | 0.680851 | -0.005319 |
| forget_generation_remote_acc | 0.462766 | 0.446809 | -0.015957 |
| retain_classification_remote_acc | 0.703631 | 0.705587 | +0.001955 |
| retain_generation_remote_acc | 0.432123 | 0.431564 | -0.000559 |
| retain_generation_bleu | 0.239301 | 0.239647 | +0.000345 |
| retain_generation_rougeL | 0.423476 | 0.424192 | +0.000716 |

Full CLEAR output:

```text
../mip_workspace/outputs/full_clear_remote_eval/hybrid_h5_top16_p000006_to_top15_full_remote_full_0607/full_clear_remote_protocol_summary.json
```

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mip-editor
export PYTHONPATH=.
```

Optional CUDA check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Expected workspace:

```text
../mip_workspace/llms/Qwen2.5-VL-3B-Instruct/
../mip_workspace/datasets/CLEAR/
../mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
```

## Run H5 Training

```bash
PYTHONPATH=. python main.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --forget_ratio 5 \
  --unlearning our \
  --use_masked_rmisu \
  --masked_rmisu_candidate_paths ../mip_workspace/outputs/paths/P_cand_projector_dim_hybrid_h5_top16_p000006_to_top15_0603.jsonl \
  --masked_rmisu_p_forget ../mip_workspace/outputs/paths/step6_hybrid_h5_top16_p000006_to_top15_0603/P_forget.jsonl \
  --masked_rmisu_p_shared ../mip_workspace/outputs/paths/step6_hybrid_h5_top16_p000006_to_top15_0603/P_shared.jsonl \
  --masked_rmisu_forget_objective pii_name_token_noise \
  --masked_rmisu_forget_ce_alpha 0.40 \
  --masked_rmisu_pii_noise_alpha 0.08 \
  --masked_rmisu_target_ce_scope name \
  --masked_rmisu_projector_edit_mode qwen_merger_mlp \
  --masked_rmisu_shared_alpha 0.0 \
  --masked_rmisu_probe_beta 0.0 \
  --masked_rmisu_max_steps 200 \
  --masked_rmisu_output ../mip_workspace/outputs/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603.json \
  --masked_rmisu_checkpoint_dir ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --skip_post_unlearning_eval \
  --batch_size 2 \
  --ptm_ckpt_batch_size 2 \
  --device cuda
```

## Step8 Val Evaluation

Probability diagnostic:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_probability_diagnostic.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --baseline_peft_checkpoint ../mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth \
  --candidate_peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_val.jsonl \
  --output ../mip_workspace/outputs/diagnostics/step8_probability_diag_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_val.json \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80 \
  --generated_name_source self
```

Generation eval:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_final_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_val.jsonl \
  --output ../mip_workspace/outputs/step8_eval_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_val.json \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80
```

## Larger Split Evaluation

Probability diagnostic:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_probability_diagnostic.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --baseline_peft_checkpoint ../mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth \
  --candidate_peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_train.jsonl \
  --output ../mip_workspace/outputs/diagnostics/step8_probability_diag_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_train.json \
  --split train \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80 \
  --generated_name_source self
```

Generation eval:

```bash
PYTHONPATH=. python causal_mip/evaluation/step8_final_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_train.jsonl \
  --output ../mip_workspace/outputs/step8_eval_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_train.json \
  --split train \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80
```

## Full CLEAR Remote Evaluation

Use a reachable remote scoring endpoint. In the latest successful run, `http://210.40.56.85:21936/v1` was used because localhost scoring was unavailable.

```bash
PYTHONPATH=. python causal_mip/evaluation/full_clear_remote_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603 \
  --base_path ../mip_workspace/datasets \
  --llm_directory ../mip_workspace/llms \
  --output_dir ../mip_workspace/outputs/full_clear_remote_eval \
  --run_id hybrid_h5_top16_p000006_to_top15_full_remote_full_0607 \
  --forget_ratio 5 \
  --device cuda \
  --remote_scoring_url http://210.40.56.85:21936/v1 \
  --score_llm Qwen2.5-VL-7B-Instruct \
  --remote_workers 4 \
  --pred_batch_size 4 \
  --save_every_batches 5 \
  --save_every_scores 20
```

## Quick Result Checks

Compare Full CLEAR baseline and H5:

```bash
jq -n --slurpfile b ../mip_workspace/outputs/full_clear_remote_eval/peft_baseline_full_remote_step9_bound_projector_0526_214028/full_clear_remote_protocol_summary.json \
      --slurpfile h ../mip_workspace/outputs/full_clear_remote_eval/hybrid_h5_top16_p000006_to_top15_full_remote_full_0607/full_clear_remote_protocol_summary.json '
  ($b[0].metrics | keys_unsorted) as $ks |
  [$ks[] | {metric: ., baseline: $b[0].metrics[.], h5: $h[0].metrics[.], delta: ($h[0].metrics[.] - $b[0].metrics[.])}]
'
```

Read larger split CE deltas:

```bash
jq '[.comparison.groups[] | select(.group=="forget_clean" or .group=="hard_retain::same_reasoning" or .group=="hard_retain::same_topic" or .group=="counterfactual_retain") | {group, target_name_ce_delta: .target_name_ce.delta, target_answer_ce_delta: .target_answer_ce.delta, target_name_logprob_delta: .target_name_mean_logprob.delta}]' \
  ../mip_workspace/outputs/diagnostics/step8_probability_diag_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_train.json
```

## Tests

```bash
PYTHONPATH=. python causal_mip/test_retain_aware_localization.py
PYTHONPATH=. python causal_mip/test_step7_masked_rmisu.py
PYTHONPATH=. python causal_mip/test_step8_probability_diagnostic.py
PYTHONPATH=. python causal_mip/test_step8_protocol.py
```

## Documents

```text
docs/CAUSAL_MIP_PROJECT_SUMMARY_0607.md
docs/PROJECT_STRUCTURE.md
```
