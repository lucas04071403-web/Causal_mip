# CHIP-Editor Route Status

更新时间：2026-05-29

本文合并并取代旧的 CHIP-Editor 技术路线图和阶段性审计文档，作为当前路线状态的主文档。实现审计见 `CHIP_EDITOR_IMPLEMENTATION_AUDIT.md`，实验结果见 `CHIP_EDITOR_EXPERIMENTS_SUMMARY.md`。

## 1. 目标

CHIP-Editor 的目标是把原 MIP-Editor 升级为一条四段式因果选择性编辑管线：

```text
candidate path localization
-> causal pair + activation patching
-> P_forget / P_shared / P_retain / P_irrelevant classification
-> masked RMisU selective editing
-> Step8 / Full CLEAR evaluation
```

对应三个科学问题：

1. 哪些路径是真正承载目标知识的因果路径，如何区分 harmful identity path 和 benign/shared topic path？
2. 如何定位跨模态因果路径中的有害知识？
3. 如何在删除目标知识时避免破坏 retain knowledge？

## 2. 当前总判断

当前状态：

```text
Step2-Step8 工程 MVP 已基本跑通；
P0-P6 工程能力已基本补齐；
但科学闭环尚未成立，不能声明三个科学问题已经完整回答。
```

按工程能力：

```text
完成度约 75%-80%
```

按科学结论：

```text
完成度约 55%-60%
```

最准确的表述：

```text
CHIP-Editor 已具备“候选路径 -> 因果验证 -> 路径分类 -> 选择性编辑 -> 固定评估”的可运行 MVP；
但因果路径选择质量、projector/cross-modal coverage 和最终 forget 效果仍不足以支撑论文式成功结论。
```

## 3. 四层路线状态

| 层级 | 路线目标 | 当前状态 | 判断 |
| --- | --- | --- | --- |
| 第一层 | 找到 candidate influential paths | 已有 `P_cand.jsonl`，覆盖 text / vision / vision_text | 工程基本完成 |
| 第二层 | 用 causal pair + patching 证明因果路径 | Step3/4/5 已实现，后续又补了 bound path 与 projector patching | 工程补齐，但科学验证仍要看新实验 |
| 第三层 | 分类 `P_forget/P_shared/P_retain/P_irrelevant` | Step6 已实现并支持 guard / saliency gate / pair-level 扩展方向 | 工程完成，依赖 Step5 质量 |
| 第四层 | 只编辑 `P_forget`，保护 `P_shared` | Step7 masked RMisU 已实现，可保存 checkpoint，支持 projector edit / CE objectives | 工程完成，效果未证明 |
| Final Eval | 验证 forget 下降且 retain 保持 | Step8 pair eval + Full CLEAR + protocol gate 已固定 | 工程完成，当前主要实验未通过 |

## 4. 技术路线

### Step 1: 固定 baseline

目标：提供可复现、可对照的 baseline。

主要产物：

```text
baseline checkpoint
baseline forget / retain evaluation
PEFT baseline Step8 pair eval
Full CLEAR baseline remote summary
```

当前重要 baseline：

```text
mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
mip_workspace/outputs/step8_eval_peft_baseline_val_full_0522.json
```

PEFT baseline Step8:

| eval set | name_hit_rate |
| --- | ---: |
| forget_clean | 0.6111 |
| hard_retain | 0.6944 |
| counterfactual_retain | 0.7222 |

### Step 2: 导出 candidate paths

目标：找到可能承载目标知识的路径。

主要模块：

```text
causal_mip/path_localization/path_schema.py
causal_mip/path_localization/cached_path_export.py
causal_mip/path_localization/mip_topk_wrapper.py
causal_mip/path_localization/beam_search_paths.py
causal_mip/path_localization/cross_modal_path_builder.py
```

历史产物：

```text
mip_workspace/outputs/paths/P_cand.jsonl
total = 2312
text = 1552
vision = 752
vision_text = 8
```

边界：原始 `vision_text` 数量少，且早期 cross-modal path 中的 `mm_projector` 在 Step4/5/7 支持不足。

### Step 3: 构造 causal pairs

目标：构造 clean/corrupt/retain anchors，用于 causal tracing。

主要模块：

```text
causal_mip/data_pairs/build_pairs.py
causal_mip/data_pairs/text_corruption.py
causal_mip/data_pairs/image_corruption.py
causal_mip/data_pairs/hard_retain_builder.py
causal_mip/data_pairs/bind_paths_to_pairs.py
```

主要 schema：

```text
forget_clean
forget_corrupt
hard_retain.same_topic
hard_retain.same_reasoning
counterfactual_retain
```

当前已支持 sample-bound path validation：

```text
--path_pair_bindings
```

### Step 4: activation cache + path patching

目标：对候选路径做 ablation/restoration。

主要模块：

```text
causal_mip/interventions/hooks.py
causal_mip/interventions/activation_cache.py
causal_mip/interventions/patching.py
causal_mip/interventions/ablation.py
causal_mip/interventions/restoration.py
```

当前支持：

```text
LLM *.mlp.down_proj
Vision *.mlp.down_proj
Qwen2.5-VL mm_projector alias -> model.visual.merger
```

边界：

```text
attention path 尚未纳入主线；
projector 支持是 Qwen2.5-VL visual.merger MVP，不是全量 projector/attention 编辑。
```

### Step 5: causal scores

目标：计算必要性、充分性和 retain impact。

核心量：

```text
Nec(P) = clean_score - ablated_score
Suf(P) = restored_score - corrupt_score
Ret(P) = retain impact on same_topic / same_reasoning / counterfactual_retain
```

主要模块：

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/necessity.py
causal_mip/causal_scores/sufficiency.py
causal_mip/causal_scores/retain_impact.py
causal_mip/causal_scores/saliency_specificity.py
causal_mip/causal_scores/build_scores.py
```

当前关键修正：

```text
没有 positive Suf，不允许进入正式 P_forget。
```

### Step 6: path classification

目标：把路径分类为：

```text
P_forget
P_shared
P_retain
P_irrelevant
```

核心逻辑：

```text
P_forget:
  forget_effect > threshold
  Suf > 0
  Ret <= retain_threshold
  saliency specificity 通过
  fully patchable

P_shared:
  forget_effect 高，但 Ret 或 retain-anchor saliency 高
```

主要模块：

```text
causal_mip/causal_scores/classify_paths.py
causal_mip/causal_scores/stability_report.py
```

当前方向：path-level 聚合已可用，但 Science Q1 结果显示 pair-level、node-level、dim-level 是下一步重点。

### Step 7: masked RMisU selective editing

目标：只编辑 `P_forget`，保护 `P_shared`。

主要模块：

```text
causal_mip/editing/masked_rmisu.py
partial_linear.py
ours.py
main.py
```

已支持：

```text
P_forget / P_shared mask loading
PartialLinear wrapping
projector edit mode qwen_merger_mlp
forget objective: activation_random / ce_ascent / activation_random_ce
scoped CE: answer / name
checkpoint saving
```

当前判断：

```text
Step7 plumbing 可用；
但 activation_random / CE ascent 目标函数未证明能稳定降低 forget_clean name-hit。
```

### Step 8: final evaluation

目标：用固定协议判断是否能声明成功。

主要模块：

```text
causal_mip/evaluation/step8_final_eval.py
causal_mip/evaluation/full_clear_remote_eval.py
causal_mip/evaluation/step8_protocol.py
```

主协议：

```text
pair screen:
  forget_clean name_hit_rate 必须低于 PEFT baseline
  hard_retain / counterfactual_retain name_hit_rate 下降不超过 0.05

Full CLEAR main:
  forget classification/generation remote acc 必须低于 baseline
  retain classification/generation remote acc 下降不超过 0.05
  四个 remote-scored 文件必须完整
```

只有输出：

```json
{"status": "pass", "can_claim_success": true}
```

才允许声明成功。

## 5. 当前路线决策

不建议优先做：

```text
attention head patching
pyvene backend
jailbreak / OCR trigger
safe-anchor 完整语义空间
直接扩大 Step7 训练
```

当前优先级：

```text
1. 提高 causal path selection 质量。
2. 修复 / 提升 projector 和 cross-modal path 的 low-Ret 子路径定位。
3. 用 probability / scoped CE diagnostic 弥补 name_hit 不可判定的问题。
4. 只有在 stable candidates 足够后再做更大规模 Step7。
```

## 6. 与 Science Q1 的关系

Science Q1 的最新主文档已经迁移到：

```text
docs/SCIENCE_Q1_MAIN_REPORT.md
docs/SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md
docs/SCIENCE_Q1_NEXT_STEPS.md
```

CHIP-Editor 的当前路线应服从 Q1 的最新结论：

```text
先定位稳定、低 Ret、可编辑的 harmful identity path；
再进入 Step7 主实验。
```

