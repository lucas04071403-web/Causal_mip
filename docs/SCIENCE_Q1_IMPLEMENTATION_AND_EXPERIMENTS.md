# Science Q1 Implementation And Experiments

更新时间：2026-05-29

本文记录 Science Q1 的关键实现路线、实验链路、结果与产物。当前结论见 `SCIENCE_Q1_MAIN_REPORT.md`，下一步计划见 `SCIENCE_Q1_NEXT_STEPS.md`。

## 1. 实现路线

### 1.1 Step5/Step6 guard

目标：避免缺少 sufficiency 证据的路径静默进入正式 `P_forget`。

核心原则：

```text
没有 positive Suf，不允许进入正式 P_forget。
```

主要修改文件：

```text
causal_mip/interventions/activation_cache.py
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/classify_paths.py
```

Step5 新增诊断字段：

```text
num_skipped_nodes
all_nodes_patchable
contains_projector
projector_patchable
num_projector_nodes
num_patchable_projector_nodes
resolved_nodes
skipped_nodes
necessity_clean_score
necessity_ablated_score
sufficiency_clean_score
sufficiency_corrupt_score
sufficiency_restored_score
sufficiency_positive
target_answer_text
clean_answer_token_positions
corrupt_answer_token_positions
```

Step6 eligibility：

```text
require_positive_forget_sufficiency = true
require_full_patchable_forget = true
```

兼容旧实验的调试开关：

```bash
--allow_zero_sufficiency_forget
--allow_partial_patchable_forget
```

这些开关只用于历史对照，不用于正式 Q1 结论。

### 1.2 SalUn/SSD retain-aware specificity

新增文件：

```text
causal_mip/causal_scores/saliency_specificity.py
```

新增能力：

```text
compute_batch_path_saliency(...)
compute_path_saliency_specificity(...)
```

核心字段：

```text
forget_saliency
retain_anchor_saliency
saliency_specificity_margin = forget_saliency - gamma * retain_anchor_saliency
saliency_specificity_ratio  = forget_saliency / (retain_anchor_saliency + eps)

forget_fisher_saliency
retain_anchor_fisher_saliency
fisher_specificity_margin
fisher_specificity_ratio
```

Step5 CLI：

```bash
--compute_saliency_specificity
--saliency_gamma 1.0
--saliency_eps 1e-6
```

Step6 CLI：

```bash
--require_saliency_specificity
--saliency_specificity_key saliency_specificity_margin
--min_saliency_specificity 0.0
--max_retain_anchor_saliency <float>
--demote_high_retain_anchor_to_irrelevant
```

### 1.3 Saliency-first candidate export

新增文件：

```text
causal_mip/path_localization/saliency_specific_export.py
```

功能：

```text
输入：
  Step5 saliency score JSONL
  原始 P_cand.jsonl

输出：
  saliency-specific P_cand JSONL
  saliency-specific path-pair binding JSONL
  summary JSON
```

默认筛选：

```text
saliency_specificity_margin > 0
all_nodes_patchable = true
vision_text path 的 projector_patchable = true
```

### 1.4 Node-level / dim-level retain-aware localization

新增或更新文件：

```text
causal_mip/path_localization/node_specific_export.py
causal_mip/path_localization/projector_dim_export.py
causal_mip/path_localization/filter_candidates.py
causal_mip/causal_scores/stability_report.py
```

目的：

```text
把 path-level specificity 推进到 node-level / projector-dim level，
从 high-Suf high-Ret shared path 中拆出 low-Ret 的 identity-specific 子路径。
```

新增 worst-anchor 字段：

```text
max_anchor_retain_saliency
min_anchor_margin
min_anchor_ratio
retain_anchor_margins
retain_anchor_ratios
```

## 2. 实验链路

### 2.1 Historical Step9 Recheck

输入：

```text
mip_workspace/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl
```

历史现象：

| metric | value |
| --- | ---: |
| Step5 records | 2720 |
| status ok | 2720 |
| positive `Suf` | 0 |
| old Step6 `P_forget` | 100 |

加 guard 后：

| category | count |
| --- | ---: |
| `P_forget` | 0 |
| `P_shared` | 110 |
| `P_retain` | 221 |
| `P_irrelevant` | 1037 |
| `demoted_from_forget` | 100 |

结论：旧 `P_forget` 只是 Nec 驱动，不能证明路径足以恢复目标知识。

### 2.2 Step5 Sufficiency Fix Check

输出：

```text
mip_workspace/outputs/scores/path_scores_suf_fix_step9.jsonl
```

结果：

| metric | value |
| --- | ---: |
| records | 2720 |
| positive `Suf` | 1807 |
| vision_text positive `Suf` | 1360 / 1360 |

未加 saliency gate 的 Step6：

```text
P_forget = 175
```

结论：Step5 已能产生 positive `Suf`，但仅靠 `Suf` 仍不能区分 harmful identity path 和 shared topic path。

### 2.3 Small Saliency Probe A: pair_000155

输入：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair155_0527.jsonl
```

结果：

| metric | value |
| --- | ---: |
| records | 16 |
| ok | 16 |
| `Suf > 0` | 11 |
| saliency margin > 0 | 14 |
| `Suf > 0` and margin > 0 | 11 |

Step6：

| category | count |
| --- | ---: |
| `P_forget` | 0 |
| `P_shared` | 9 |
| `P_retain` | 2 |
| `P_irrelevant` | 5 |

代表性 projector / vision_text：

| path | `Suf` | `Ret` | saliency margin | category |
| --- | ---: | ---: | ---: | --- |
| `cross_modal_p000000` | 2.72265625 | 0.953125 | 0.0000157381 | `P_shared` |
| `cross_modal_p000003` | 2.72265625 | 0.94531250 | 0.0000049894 | `P_shared` |
| `cross_modal_p000007` | 2.72265625 | 0.953125 | 0.0000090624 | `P_shared` |

结论：projector / vision_text causal effect 很强，但 retain impact 也很高，应归为 shared path。

### 2.4 Small Saliency Probe B: pair_000002 + pair_000064

输入：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl
```

结果：

| metric | value |
| --- | ---: |
| records | 32 |
| ok | 32 |
| `Suf > 0` | 26 |
| saliency margin > 0 | 0 |
| avg forget saliency | 0.0000532902 |
| avg retain anchor saliency | 0.0000801873 |
| avg margin | -0.0000268971 |

Step6 对照：

| setting | `P_forget` | `P_shared` | `P_retain` | `P_irrelevant` |
| --- | ---: | ---: | ---: | ---: |
| no saliency gate | 5 | 9 | 1 | 9 |
| with saliency gate | 0 | 9 | 1 | 14 |

结论：saliency gate 能识别 retain-biased path，避免误判为 `P_forget`。

### 2.5 First Saliency-First Candidate Regeneration

输入：

```text
path_scores_scienceq1_saliency_pair155_0527.jsonl
path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl
mip_workspace/outputs/paths/P_cand.jsonl
```

输出：

```text
mip_workspace/outputs/paths/P_cand_saliency_specific_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_bound_0527.jsonl
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_candidates_0527.jsonl
mip_workspace/outputs/paths/step6_saliency_specific_candidates_0527
```

candidate export：

| metric | value |
| --- | ---: |
| input score records | 48 |
| positive-specific filtered records | 14 |
| selected bindings | 6 |
| selected candidate paths | 6 |

Step5：

| metric | value |
| --- | ---: |
| records | 6 |
| ok | 6 |
| `Suf > 0` | 5 |
| saliency margin > 0 | 6 |
| `Suf > 0` and margin > 0 | 5 |

Step6：

| category | count |
| --- | ---: |
| `P_forget` | 0 |
| `P_shared` | 3 |
| `P_retain` | 1 |
| `P_irrelevant` | 2 |

结论：saliency-first candidate generation 生效，但小规模下仍没有严格 `P_forget`。

### 2.6 20-Pair Expansion

pair 文件：

```text
mip_workspace/outputs/causal_pairs_train_scienceq1_20pairs_0527.jsonl
```

20-pair saliency probe：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl
```

| metric | value |
| --- | ---: |
| records | 320 |
| ok | 320 |
| `Suf > 0` | 244 |
| saliency margin > 0 | 84 |
| `Suf > 0` and margin > 0 | 59 |

按 modality：

| modality | n | `Suf > 0` | margin > 0 | both |
| --- | ---: | ---: | ---: | ---: |
| text | 80 | 31 | 17 | 7 |
| vision | 80 | 53 | 25 | 10 |
| vision_text | 160 | 160 | 42 | 42 |

saliency-first export：

```text
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527_summary.json
```

| metric | value |
| --- | ---: |
| input score records | 320 |
| positive-specific filtered records | 84 |
| selected bindings | 42 |
| selected candidate paths | 34 |

Step5 on regenerated candidates：

```text
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl
```

| metric | value |
| --- | ---: |
| records | 42 |
| ok | 42 |
| `Suf > 0` | 28 |
| saliency margin > 0 | 42 |
| `Suf > 0` and margin > 0 | 28 |
| `Ret <= retain_threshold` | 20 |
| record-level strict forget-like | 8 |

按 modality：

| modality | n | `Suf > 0` | margin > 0 | low `Ret` | record-level forget-like |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 12 | 5 | 12 | 10 | 4 |
| vision | 14 | 7 | 14 | 10 | 4 |
| vision_text | 16 | 16 | 16 | 0 | 0 |

Step6 strict classification：

```text
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527
```

| category | count |
| --- | ---: |
| `P_forget` | 4 |
| `P_shared` | 12 |
| `P_retain` | 2 |
| `P_irrelevant` | 16 |

严格 `P_forget`：

| path_id | pair | modality | `Suf` | `Ret` | saliency margin | forget saliency | retain anchor saliency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `saliency20_p000027` | `pair_000185` | vision | 0.0234375 | -0.0169271 | 0.0000020433 | 0.0000330731 | 0.0000310298 |
| `saliency20_p000028` | `pair_000185` | vision | 0.0234375 | -0.0182292 | 0.0000020036 | 0.0000334606 | 0.0000314570 |
| `saliency20_p000031` | `pair_000067` | vision | 0.0156250 | -0.0130208 | 0.0000009385 | 0.0000288475 | 0.0000279089 |
| `saliency20_p000033` | `pair_000067` | vision | 0.0312500 | -0.0039063 | 0.0000000438 | 0.0000472693 | 0.0000472255 |

P_shared projector 对照：

| path_id | modality | `Suf` | `Ret` | saliency margin | contains projector |
| --- | --- | ---: | ---: | ---: | --- |
| `saliency20_p000000` | vision_text | 2.15625 | 0.892578 | 0.0000706877 | true |
| `saliency20_p000001` | vision_text | 2.15625 | 0.892578 | 0.0000684283 | true |
| `saliency20_p000004` | vision_text | 2.62305 | 0.840495 | 0.0000195429 | true |
| `saliency20_p000007` | vision_text | 2.52344 | 0.727865 | 0.0000283855 | true |

结论：20-pair 扩展首次产生严格 `P_forget`，但只覆盖 `pair_000185` 和 `pair_000067`，且全部为 vision-only。

### 2.7 Retain-aware Node/Dim Run

输入候选：

```text
mip_workspace/outputs/paths/P_cand_node_specific_retainaware_dedup_20pairs_0529.jsonl
mip_workspace/outputs/paths/P_cand_projector_dim_retainaware_top4_dedup_20pairs_0529.jsonl
```

合并输出：

```text
mip_workspace/outputs/paths/P_cand_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/paths/P_cand_retainaware_node_projdim_20pairs_bound_0529.jsonl
```

规模：

| source | count |
| --- | ---: |
| node-specific | 10 |
| projector-dim | 9 |
| merged dedup | 19 |

Step5：

```text
mip_workspace/outputs/scores/path_scores_retainaware_node_projdim_20pairs_0529.jsonl
```

| metric | value |
| --- | ---: |
| records | 19 |
| ok | 19 |
| `Suf > 0` | 9 |
| low `Ret` | 13 |
| positive `min_anchor_margin` | 13 |

Step6 pair-level SSD gate：

```text
mip_workspace/outputs/paths/step6_retainaware_node_projdim_20pairs_pairlevel_0529
```

| category | count |
| --- | ---: |
| `P_forget` | 1 |
| `P_shared` | 3 |
| `P_retain` | 3 |
| `P_irrelevant` | 12 |

Repeat stability：

```text
mip_workspace/outputs/scores/stability_retainaware_node_projdim_3anchors_0529.jsonl
mip_workspace/outputs/scores/stability_retainaware_node_projdim_3anchors_0529_summary.json
mip_workspace/outputs/scores/stability_retainaware_node_projdim_3anchors_minsuf_0529.jsonl
mip_workspace/outputs/scores/stability_retainaware_node_projdim_3anchors_minsuf_0529_summary.json
```

| metric | value |
| --- | ---: |
| loaded score records | 57 |
| report records | 19 |
| stable | 1 |

唯一 stable candidate：

| field | value |
| --- | --- |
| `path_id` | `projdimRA20d_p000003` |
| `pair_id` | `pair_000089` |
| modality | `vision_text` |
| original path | `saliency20_p000004` |
| selected dims | `[387, 318, 886, 984]` |
| pass count | 3/3 |
| pass rate | 1.0 |
| `Suf.mean` | 0.03125 |
| `Ret.mean` | -0.00390625 |
| `Ret.max` | 0.00390625 |

stable exports：

```text
mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_bound_0529.jsonl
mip_workspace/outputs/paths/P_forget_stable_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/paths/P_shared_stable_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/paths/stable_retainaware_node_projdim_20pairs_0529_summary.json
```

### 2.8 Step7 Formal Stable Single-Candidate Smoke

run id：

```text
step7_single_stable_smoke_retainaware_0529_152017
```

输入：

| file | records |
| --- | ---: |
| `P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl` | 1 |
| `P_forget_stable_retainaware_node_projdim_20pairs_0529.jsonl` | 1 |
| `P_shared_stable_retainaware_node_projdim_20pairs_0529.jsonl` | 0 |

配置：

| item | value |
| --- | --- |
| max steps | 5 |
| post eval | skipped during Step7 |
| forget objective | `name_ce_ascent` |
| forget CE alpha | 0.2 |
| target CE scope | `name` |
| RMU alpha | 0.0 |
| RMU beta | 1.0 |
| shared alpha | 0.0 |

输出：

```text
mip_workspace/outputs/masked_rmisu_step7_single_stable_smoke_retainaware_0529_152017.json
mip_workspace/outputs/checkpoints/step7_single_stable_smoke_retainaware_0529_152017
mip_workspace/outputs/logs/step7_single_stable_smoke_retainaware_0529_152017.log
```

mask summary：

| field | value |
| --- | ---: |
| `num_loss_records` | 5 |
| `mask_summary.num_modules` | 1 |
| `mask_summary.num_editable_neurons` | 4 |
| skipped modules | 0 |

module resolution：

| field | value |
| --- | --- |
| logical module | `mm_projector` |
| trace module | `visual.merger.mlp.0` |
| edit module | `visual.merger.mlp.0` |
| editable dims | `[318, 387, 886, 984]` |

结论：smoke 证明 formal stable 单候选能进入 Step7，projector-dim path 能落到真实可编辑模块；但它不是正式 Step7。

### 2.9 Step8 Pair Diagnostic

`pair_000089` 不在 `causal_pairs_val.jsonl`，所以从 20-pair train diagnostic 文件抽出：

```text
mip_workspace/outputs/diagnostics/causal_pair_000089_step8_diagnostic_0529.jsonl
```

输出：

```text
mip_workspace/outputs/diagnostics/step8_diag_pair000089_baseline_clear_adapter_0529.json
mip_workspace/outputs/diagnostics/step8_diag_pair000089_step7_single_stable_smoke_0529.json
```

对比：

| sample type | baseline prediction | smoke prediction | observation |
| --- | --- | --- | --- |
| `forget_clean` | `Luciano Valdez... war equipment.` | `Luciano Valdez... weapons.` | baseline 已不命中目标姓名，无法判定新增遗忘信号 |
| `same_topic` | `Luciano Valdez... weapons.` | `Luciano Valdez... war equipment.` | 无明显恶化 |
| `same_reasoning` | `Cheong Yew Han...` | `Cheong Yew Han...` | 完全不变 |
| `counterfactual_retain` | `Simon Makoni...` | `Ismail Jengo...` | retain-side 正向信号 |

结论：

```text
有一个 retain-side 正向方向信号；
没有 forget-side 可判定信号。
```

## 3. 测试

已通过的关键测试：

```bash
PYTHONPATH=. python causal_mip/test_retain_aware_localization.py
PYTHONPATH=. python causal_mip/test_step5_causal_scores.py
PYTHONPATH=. python causal_mip/test_step6_classify_paths.py
```

历史相关测试：

```bash
python -m py_compile causal_mip/causal_scores/saliency_specificity.py \
  causal_mip/causal_scores/build_scores.py \
  causal_mip/causal_scores/classify_paths.py \
  causal_mip/path_localization/saliency_specific_export.py

PYTHONPATH=. python causal_mip/test_step5_causal_scores.py
PYTHONPATH=. python causal_mip/test_step6_classify_paths.py
```

## 4. 主要产物索引

20-pair saliency：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527/P_forget.jsonl
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527/P_shared.jsonl
```

retain-aware node/dim：

```text
mip_workspace/outputs/paths/P_cand_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/scores/path_scores_retainaware_node_projdim_20pairs_0529.jsonl
mip_workspace/outputs/paths/step6_retainaware_node_projdim_20pairs_pairlevel_0529
mip_workspace/outputs/scores/stability_retainaware_node_projdim_3anchors_0529_summary.json
mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl
```

Step7/Step8 smoke:

```text
mip_workspace/outputs/masked_rmisu_step7_single_stable_smoke_retainaware_0529_152017.json
mip_workspace/outputs/checkpoints/step7_single_stable_smoke_retainaware_0529_152017
mip_workspace/outputs/diagnostics/step8_diag_pair000089_baseline_clear_adapter_0529.json
mip_workspace/outputs/diagnostics/step8_diag_pair000089_step7_single_stable_smoke_0529.json
```

