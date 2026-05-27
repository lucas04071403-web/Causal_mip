# Step9 Causal-Selective Editing Closure Runbook

本文档用于把 `CHIP_EDITOR_TECHNICAL_ROADMAP.md` 的四层路线真正闭环：

```text
Step2 candidate paths
-> Step3 causal pairs
-> Step4 activation patching
-> Step5 Nec/Suf/Ret causal scores
-> Step6 P_forget/P_shared/P_retain/P_irrelevant
-> Step7 selective masked editing
-> Step8 pair eval + Full CLEAR eval
```

当前代码已补齐 P0-P6 的关键工程入口。完整科学结论仍需要按本文档重跑 Step5-Step8 的 CUDA 实验产物。

---

## 0. 统一工作区

默认 workspace 为仓库父目录下的 `mip_workspace`：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
export MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

不要把新产物写入 `MIP-Editor/mip_workspace`。该目录目前不是主产物目录。

---

## 1. 输入产物检查

必须存在：

```text
$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl
$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_train.jsonl
$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_val.jsonl
$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train.jsonl
$MIP_WORKSPACE_ROOT/outputs/causal_pairs_val.jsonl
```

快速检查：

```bash
wc -l \
  "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_train.jsonl" \
  "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_val.jsonl" \
  "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train.jsonl" \
  "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_val.jsonl"
```

当前已见主产物规模：

```text
P_cand.jsonl = 2312
P_cand_bound_train.jsonl = 1360
P_cand_bound_val.jsonl = 144
causal_pairs_train.jsonl = 170
causal_pairs_val.jsonl = 18
```

---

## 2. 重跑 Step5：绑定样本级 causal scores

核心要求：

1. 使用 `P_cand_bound_train.jsonl`，避免 path 与 pair 的笛卡尔组合。
2. 使用当前代码的 sufficiency diagnostics。
3. 使用当前代码的 `mm_projector` whole-vector patching。

推荐先跑小规模 smoke：

```bash
python causal_mip/causal_scores/build_scores.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory "$MIP_WORKSPACE_ROOT/llms/" \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs" \
  --device cuda \
  --image_resize 224 \
  --pairs_path "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --path_pair_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_train.jsonl" \
  --path_modality all \
  --num_pairs 2 \
  --output "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_bound_projector_smoke.jsonl"
```

确认 smoke 没有系统性跳过后，跑主版本：

```bash
python causal_mip/causal_scores/build_scores.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory "$MIP_WORKSPACE_ROOT/llms/" \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs" \
  --device cuda \
  --image_resize 224 \
  --pairs_path "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --path_pair_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_bound_train.jsonl" \
  --path_modality all \
  --output "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_bound_projector_v1.jsonl"
```

Step5 验收：

```text
status=ok 占主体
vision_text path 不再系统性少 patch mm_projector
num_patchable_nodes 对 4 节点 vision_text path 应可达到 4
sufficiency.clean_target / corrupt_target / restored_nodes 字段存在
至少一部分 path 出现 positive Suf
```

---

## 3. 重跑 Step6：生成新的主分类

```bash
python causal_mip/causal_scores/classify_paths.py \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_bound_projector_v1.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_bound_suf_projector_v1" \
  --alpha 1.0 \
  --forget_quantile 0.75 \
  --retain_quantile 0.75 \
  --min_forget_effect 0.0 \
  --min_retain_impact 0.0
```

Step6 验收：

```text
classification_summary.json 存在
P_forget.jsonl / P_shared.jsonl / P_retain.jsonl / P_irrelevant.jsonl 存在
P_forget 不应完全由旧的 Nec-only 信号决定
vision_text path 可进入分类结果
```

---

## 4. 重跑 Step7：选择性编辑

当前 Step7 支持三种 forget objective：

```text
activation_random     默认，沿用 RMisU 风格 activation random direction
ce_ascent             只做 forget CE ascent
activation_random_ce  activation random + forget CE ascent
```

当前 Step7 支持两种 projector 编辑策略：

```text
qwen_merger_mlp  默认，把 mm_projector 映射到 Qwen2.5-VL 的 model.visual.merger.mlp.0
skip             显式跳过 projector 参数编辑
```

推荐先跑 `activation_random_ce`：

```bash
python main.py \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory "$MIP_WORKSPACE_ROOT/llms/" \
  --base_path "$MIP_WORKSPACE_ROOT/datasets/" \
  --path_path "$MIP_WORKSPACE_ROOT/influential_paths/" \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs" \
  --device cuda \
  --unlearning our \
  --use_masked_rmisu \
  --masked_rmisu_candidate_paths "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --masked_rmisu_p_forget "$MIP_WORKSPACE_ROOT/outputs/paths/step6_bound_suf_projector_v1/P_forget.jsonl" \
  --masked_rmisu_p_shared "$MIP_WORKSPACE_ROOT/outputs/paths/step6_bound_suf_projector_v1/P_shared.jsonl" \
  --masked_rmisu_forget_objective activation_random_ce \
  --masked_rmisu_forget_ce_alpha 0.1 \
  --masked_rmisu_projector_edit_mode qwen_merger_mlp \
  --masked_rmisu_shared_alpha 1.0 \
  --rmu_alpha 0.5 \
  --rmu_beta 0.5 \
  --finetune_epochs 1 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --skip_post_unlearning_eval \
  --masked_rmisu_output "$MIP_WORKSPACE_ROOT/outputs/masked_rmisu_step9_bound_projector_ce01.json" \
  --masked_rmisu_checkpoint_dir "$MIP_WORKSPACE_ROOT/outputs/checkpoints/masked_rmisu_step9_bound_projector_ce01"
```

Step7 验收：

```text
mask_summary.num_modules > 0
mask_summary.modules 中 projector module 若存在，应有 edit_module=model.visual.merger.mlp.0
losses 中 forget_objective=activation_random_ce
forget_ce_loss 被记录
checkpoint_dir 存在
```

---

## 5. Step8 快速 pair eval

先用 pair-based Step8 做筛选：

```bash
python causal_mip/evaluation/step8_final_eval.py \
  --model_path "$MIP_WORKSPACE_ROOT/llms/Qwen2.5-VL-3B-Instruct" \
  --peft_checkpoint "$MIP_WORKSPACE_ROOT/outputs/checkpoints/masked_rmisu_step9_bound_projector_ce01" \
  --pair_jsonl "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_val.jsonl" \
  --output "$MIP_WORKSPACE_ROOT/outputs/step8_eval_step9_bound_projector_ce01_val.json" \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80
```

快速验收线：

```text
forget_clean name_hit_rate < 0.6111
hard_retain name_hit_rate 不明显低于 0.6944
counterfactual_retain name_hit_rate 不明显低于 0.7222
```

如果 pair eval 通过，再进入 Full CLEAR remote judge：

```bash
python causal_mip/evaluation/full_clear_remote_eval.py \
  --model_path "$MIP_WORKSPACE_ROOT/llms/Qwen2.5-VL-3B-Instruct" \
  --peft_checkpoint "$MIP_WORKSPACE_ROOT/outputs/checkpoints/masked_rmisu_step9_bound_projector_ce01" \
  --base_path "$MIP_WORKSPACE_ROOT/datasets/" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/full_clear_remote_eval" \
  --run_id step9_bound_projector_ce01_full_remote \
  --output "$MIP_WORKSPACE_ROOT/outputs/full_clear_step9_bound_projector_ce01.json" \
  --device cuda \
  --tasks clf_forget clf_retain gen_forget gen_retain
```

Full CLEAR baseline 也必须按同一协议存在，例如：

```text
$MIP_WORKSPACE_ROOT/outputs/full_clear_baseline_protocol_summary.json
```

最后用固定门控器生成主协议判定：

```bash
python causal_mip/evaluation/step8_protocol.py \
  --pair_eval "$MIP_WORKSPACE_ROOT/outputs/step8_eval_step9_bound_projector_ce01_val.json" \
  --pair_baseline "$MIP_WORKSPACE_ROOT/outputs/step8_eval_peft_baseline_val_full_0522.json" \
  --full_clear_summary "$MIP_WORKSPACE_ROOT/outputs/full_clear_step9_bound_projector_ce01.json" \
  --full_clear_baseline "$MIP_WORKSPACE_ROOT/outputs/full_clear_baseline_protocol_summary.json" \
  --output "$MIP_WORKSPACE_ROOT/outputs/step8_protocol_step9_bound_projector_ce01.json"
```

判定规则：

```text
pair_screen 必须通过：
  forget_clean name_hit_rate 低于 PEFT baseline
  hard_retain / counterfactual_retain name_hit_rate 下降不超过 0.05

Full CLEAR main 必须通过：
  forget classification/generation remote acc 低于 baseline
  retain classification/generation remote acc 下降不超过 0.05
  四个 task 必须全部完成 remote scoring
```

只有 `step8_protocol.py` 输出：

```json
{"status": "pass", "can_claim_success": true}
```

才允许声明 Step8 主协议通过。

---

## 6. 结果判定

只有同时满足下面条件，才能说四层技术路线形成完整闭环：

```text
Step5: bound path scoring 有 positive Suf，且 vision_text 不再跳过 projector
Step6: 新 P_forget/P_shared 来自 bound+sufficiency+projector scores
Step7: 只编辑 P_forget - P_shared 的 masked neurons，并记录 projector/CE objective
Step8: pair screen 通过
Full CLEAR: remote main protocol 通过
Step8 protocol report: can_claim_success=true
```

如果 `step8_protocol.py` 未通过，则当前只能声明：

```text
工程闭环完成，但科学效果尚未达到最终验收线。
```
