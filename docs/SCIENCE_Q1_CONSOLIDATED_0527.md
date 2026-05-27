# Science Q1 Consolidated Report

更新时间：2026-05-27

本文合并以下 Science Q1 中间文档，作为当前主文档使用：

```text
SCIENCE_Q1_STEP5_STEP6_FIX_0527.md
SCIENCE_Q1_SALUN_SSD_CAUSAL_ANCHOR_OPTIMIZATION_0527.md
SCIENCE_Q1_SALIENCY_PROBE_RESULT_0527.md
SCIENCE_Q1_SALIENCY_CANDIDATE_REGEN_RESULT_0527.md
SCIENCE_Q1_20PAIR_SALIENCY_EXPANSION_RESULT_0527.md
```

本文删除了重复背景、长命令日志和阶段性解释，只保留：

```text
1. 科学问题 1 的当前回答；
2. 已实现的关键代码改动；
3. 可复现实验链路；
4. 关键实验结果；
5. 当前问题与下一步优化路线。
```

---

## 1. Science Q1

科学问题 1：

```text
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分 harmful identity path 和 benign/shared topic path？
```

当前回答：

```text
可以初步回答，但还没有完全解决。
```

已经成立的判断：

```text
1. 只用 Nec 不够，必须加入 Suf。
2. 只看 Suf 也不够，必须加入 Ret / retain anchor。
3. saliency_specificity_margin 可以作为候选生成前置筛选。
4. Suf > 0 且 Ret 高的 projector / vision_text path 应归入 P_shared。
5. Suf > 0、Ret 低、saliency-specific、fully patchable 的 path 可以作为 P_forget。
```

当前不足：

```text
1. 严格 P_forget 只覆盖 2/20 个 pair。
2. 严格 P_forget 只出现在 vision branch。
3. vision_text / projector path 仍然主要被 retain anchor 判为 P_shared。
4. 当前 saliency specificity 是 path-level，粒度太粗。
5. 已发现的 P_forget saliency margin 较小，需要 repeat validation。
```

---

## 2. Final Protocol

当前 Science Q1 的有效协议是：

```text
SalUn/SSD:
  用 forget_saliency - gamma * retain_anchor_saliency
  生成 forget-specific candidate path。

CaMU / causal tracing:
  用 Nec / Suf / Ret 验证 path 是否真有因果作用。

SMFA / retain anchor:
  用 retain anchors 区分 harmful identity path 和 benign/shared topic path。
```

分类口径：

```text
P_forget:
  forget_effect > tau_f
  Ret <= tau_r
  Suf > 0
  saliency_specificity_margin > tau_s
  all_score_records_fully_patchable = true

P_shared:
  forget_effect > tau_f
  and (
    Ret > tau_r
    or retain_anchor_saliency too high
  )

P_retain:
  forget_effect <= tau_f
  retain_impact > tau_r

P_irrelevant:
  其他路径
```

本轮主要阈值：

```text
forget_threshold = 0.015625
retain_threshold = 0.005208333333333333
saliency_gamma = 1.0
min_saliency_specificity = 0.0
```

---

## 3. Key Implementation Changes

## 3.1 Step5/Step6 Guard

核心修正：

```text
没有 positive Suf，不允许进入正式 P_forget。
```

修改文件：

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

Step6 新增 eligibility：

```text
require_positive_forget_sufficiency = true
require_full_patchable_forget = true
```

兼容旧实验的调试开关：

```bash
--allow_zero_sufficiency_forget
--allow_partial_patchable_forget
```

这些开关只用于历史对照，不用于正式 Science Q1 结论。

## 3.2 Saliency Specificity

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

## 3.3 Saliency-First Candidate Generation

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

默认筛选规则：

```text
saliency_specificity_margin > 0
all_nodes_patchable = true
vision_text path 的 projector_patchable = true
```

---

## 4. Experiment Chain

## 4.1 Historical Step9 Recheck

历史文件：

```text
mip_workspace/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl
```

历史现象：

| metric | value |
| --- | ---: |
| Step5 records | 2720 |
| status ok | 2720 |
| positive Suf | 0 |
| old Step6 P_forget | 100 |

修正后 Step6 guard：

| category | count |
| --- | ---: |
| P_forget | 0 |
| P_shared | 110 |
| P_retain | 221 |
| P_irrelevant | 1037 |
| demoted_from_forget | 100 |

结论：

```text
历史 Step9 的 100 条 P_forget 全部缺少 positive Suf，
不能作为可靠 target-knowledge causal path。
```

## 4.2 Step5 Sufficiency Fix Check

修复后的中间分数：

```text
mip_workspace/outputs/scores/path_scores_suf_fix_step9.jsonl
```

核心结果：

| metric | value |
| --- | ---: |
| records | 2720 |
| positive Suf | 1807 |
| vision_text positive Suf | 1360 / 1360 |

未加 saliency gate 的 Step6：

```text
P_forget = 175
```

结论：

```text
Step5 已经可以产生 positive Suf。
但仅有 Suf 仍不足以区分 harmful identity path 和 shared topic path，
因此必须继续加入 saliency specificity 与 retain anchor。
```

## 4.3 Small Saliency Probe A: pair_000155

输入：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair155_0527.jsonl
```

结果：

| metric | value |
| --- | ---: |
| records | 16 |
| ok | 16 |
| Suf > 0 | 11 |
| saliency margin > 0 | 14 |
| Suf > 0 and margin > 0 | 11 |

Step6：

| category | count |
| --- | ---: |
| P_forget | 0 |
| P_shared | 9 |
| P_retain | 2 |
| P_irrelevant | 5 |

代表性 projector / vision_text：

| path | Suf | Ret | saliency margin | category |
| --- | ---: | ---: | ---: | --- |
| `cross_modal_p000000` | 2.72265625 | 0.953125 | 0.0000157381 | P_shared |
| `cross_modal_p000003` | 2.72265625 | 0.94531250 | 0.0000049894 | P_shared |
| `cross_modal_p000007` | 2.72265625 | 0.953125 | 0.0000090624 | P_shared |

结论：

```text
projector / vision_text path 的 causal effect 很强，
但 Ret 也很高，所以是 shared path，不是严格 P_forget。
```

## 4.4 Small Saliency Probe B: pair_000002 + pair_000064

输入：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl
```

结果：

| metric | value |
| --- | ---: |
| records | 32 |
| ok | 32 |
| Suf > 0 | 26 |
| saliency margin > 0 | 0 |
| Suf > 0 and margin > 0 | 0 |
| avg forget saliency | 0.0000532902 |
| avg retain anchor saliency | 0.0000801873 |
| avg margin | -0.0000268971 |

Step6 对照：

| setting | P_forget | P_shared | P_retain | P_irrelevant |
| --- | ---: | ---: | ---: | ---: |
| no saliency gate | 5 | 9 | 1 | 9 |
| with saliency gate | 0 | 9 | 1 | 14 |

结论：

```text
这些路径按 Nec/Suf/Ret 会进入 P_forget，
但 retain_anchor_saliency > forget_saliency。
saliency gate 正确阻止了 retain-biased path 被误判为 harmful identity-specific path。
```

## 4.5 First Saliency-First Candidate Regeneration

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
| Suf > 0 | 5 |
| saliency margin > 0 | 6 |
| Suf > 0 and margin > 0 | 5 |

Step6：

| category | count |
| --- | ---: |
| P_forget | 0 |
| P_shared | 3 |
| P_retain | 1 |
| P_irrelevant | 2 |

结论：

```text
saliency-first candidate generation 生效；
但小规模下仍没有严格 P_forget。
主要原因是 projector path Ret 过高，text/vision path forget_effect 偏弱或 retain_impact 超阈值。
```

## 4.6 20-Pair Expansion

pair 文件：

```text
mip_workspace/outputs/causal_pairs_train_scienceq1_20pairs_0527.jsonl
```

20 个 pair：

```text
pair_000064
pair_000044
pair_000042
pair_000030
pair_000126
pair_000142
pair_000079
pair_000068
pair_000124
pair_000155
pair_000089
pair_000025
pair_000138
pair_000166
pair_000017
pair_000031
pair_000067
pair_000180
pair_000185
pair_000129
```

20-pair saliency probe：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl
```

| metric | value |
| --- | ---: |
| records | 320 |
| ok | 320 |
| Suf > 0 | 244 |
| saliency margin > 0 | 84 |
| Suf > 0 and margin > 0 | 59 |

按 modality：

| modality | n | Suf > 0 | margin > 0 | both |
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
| Suf > 0 | 28 |
| saliency margin > 0 | 42 |
| Suf > 0 and margin > 0 | 28 |
| Ret <= retain_threshold | 20 |
| record-level strict forget-like | 8 |

按 modality：

| modality | n | Suf > 0 | margin > 0 | low Ret | record-level forget-like |
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
| P_forget | 4 |
| P_shared | 12 |
| P_retain | 2 |
| P_irrelevant | 16 |

严格 P_forget：

| path_id | pair | modality | Suf | Ret | saliency margin | forget saliency | retain anchor saliency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `saliency20_p000027` | `pair_000185` | vision | 0.0234375 | -0.0169271 | 0.0000020433 | 0.0000330731 | 0.0000310298 |
| `saliency20_p000028` | `pair_000185` | vision | 0.0234375 | -0.0182292 | 0.0000020036 | 0.0000334606 | 0.0000314570 |
| `saliency20_p000031` | `pair_000067` | vision | 0.0156250 | -0.0130208 | 0.0000009385 | 0.0000288475 | 0.0000279089 |
| `saliency20_p000033` | `pair_000067` | vision | 0.0312500 | -0.0039063 | 0.0000000438 | 0.0000472693 | 0.0000472255 |

P_shared projector 对照：

| path_id | modality | Suf | Ret | saliency margin | contains projector |
| --- | --- | ---: | ---: | ---: | --- |
| `saliency20_p000000` | vision_text | 2.15625 | 0.892578 | 0.0000706877 | true |
| `saliency20_p000001` | vision_text | 2.15625 | 0.892578 | 0.0000684283 | true |
| `saliency20_p000004` | vision_text | 2.62305 | 0.840495 | 0.0000195429 | true |
| `saliency20_p000007` | vision_text | 2.52344 | 0.727865 | 0.0000283855 | true |

结论：

```text
20-pair 扩展首次产生严格 P_forget。
但 P_forget 只出现在 vision-only path，且只覆盖 pair_000185 和 pair_000067。

vision_text / projector path 的 Suf 全部为正，但 low Ret = 0，
因此当前粒度下应归为 P_shared，不应直接编辑。
```

---

## 5. Current Findings

## 5.1 已经解决的问题

```text
1. 历史 Step9 中 Suf=0 仍产生 P_forget 的问题已经被 guard 修正。
2. Step5 已经能产生 positive Suf，不再是全部 Suf=0。
3. saliency specificity 已接入 Step5/Step6。
4. saliency-first Step2 已能生成正 margin 候选。
5. retain anchor 能有效阻止 retain-biased path 被误判为 P_forget。
6. 20-pair 扩展中首次得到 4 条严格 P_forget。
```

## 5.2 仍然存在的问题

```text
1. P_forget recall 低，只覆盖 2/20 个 pair。
2. P_forget 都是 vision-only，没有 projector / vision_text。
3. projector path causality 很强，但 Ret 也很高，当前属于 shared path。
4. text path 有 record-level forget-like 信号，但聚合后不稳定。
5. saliency margin 值较小，需要 repeat validation。
6. path-level specificity 太粗，无法拆分 shared path 内部的 harmful identity 子空间。
```

---

## 6. Decision

当前不建议：

```text
1. 直接做总体超参优化。
2. 直接扩大 Step7 编辑。
3. 直接编辑 high-Suf projector path。
```

原因：

```text
projector / vision_text path 的 causal effect 很强，
但 retain impact 同样很强。
直接编辑这些 path 很可能破坏 retain knowledge。
```

当前建议：

```text
继续优先科学问题 1。
先把 path-level specificity 推进到 component-level specificity。
```

下一步任务：

```text
1. 对当前 4 条 P_forget 做 repeated causal validation。
2. 对 20-pair 中 high-Suf high-Ret 的 projector path 做 dim-level 分解。
3. 对 text / vision path 做 node-level ablation 或 leave-one-layer-out。
4. 将长 path 压缩为少量真正有贡献的 nodes/dims。
5. 在 20-pair 上把 strict P_forget coverage 从 2/20 提升到至少 6/20 后，再考虑扩展到 50 pair。
```

下一阶段目标公式：

```text
score(node_or_dim) =
  forget_saliency(node_or_dim)
  - gamma * retain_anchor_saliency(node_or_dim)

then validate:
  Suf > 0
  Ret <= retain_threshold
  saliency_specificity_margin > 0
```

目标：

```text
把当前 P_shared projector path 拆开，
找出其中真正 harmful identity-specific 的子路径或子空间。
```

---

## 7. Main Artifacts

代码：

```text
causal_mip/causal_scores/saliency_specificity.py
causal_mip/path_localization/saliency_specific_export.py
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/build_scores.py
causal_mip/causal_scores/classify_paths.py
causal_mip/interventions/activation_cache.py
```

测试：

```text
causal_mip/test_step2.py
causal_mip/test_step5_causal_scores.py
causal_mip/test_step6_classify_paths.py
```

关键输出：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527/P_forget.jsonl
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527/P_shared.jsonl
```

---

## 8. Read Order

后续建议以本文作为 Science Q1 主文档。

旧文档保留为原始实验日志：

```text
SCIENCE_Q1_STEP5_STEP6_FIX_0527.md
SCIENCE_Q1_SALUN_SSD_CAUSAL_ANCHOR_OPTIMIZATION_0527.md
SCIENCE_Q1_SALIENCY_PROBE_RESULT_0527.md
SCIENCE_Q1_SALIENCY_CANDIDATE_REGEN_RESULT_0527.md
SCIENCE_Q1_20PAIR_SALIENCY_EXPANSION_RESULT_0527.md
```

