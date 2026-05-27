# Science Q1 20-Pair Saliency-Specific Expansion Result

更新时间：2026-05-27

本轮目标：

```text
把科学问题 1 的 saliency-specific candidate generation 从小规模 pair 扩大到 10-20 个 pair，
验证 Step2 -> Step5 -> Step6 是否能产生 Suf > 0 且 saliency-specific 的严格 P_forget。
```

本轮结论：

```text
20-pair 扩展已经跑通。

相比前一轮小规模实验，本轮首次产生严格 P_forget：
  P_forget = 4
  P_shared = 12
  P_retain = 2
  P_irrelevant = 16

但 P_forget 仍然很窄：
  只出现在 vision 单模态
  只覆盖 pair_000185 和 pair_000067
  没有 vision_text / projector path 进入 P_forget
```

因此，科学问题 1 当前不是“完全解决”，而是进入了可验证阶段：

```text
当前 protocol 已经能找到少量 harmful-identity-like causal path；
但还不能稳定地区分所有 harmful identity path 与 benign/shared topic path。
```

---

## 1. Protocol

本轮沿用 SalUn/SSD + CaMU/causal tracing + SMFA/retain anchor 的三段式设计：

```text
1. SalUn/SSD:
   用 forget_saliency - gamma * retain_anchor_saliency 生成 forget-specific candidate path。

2. CaMU / causal tracing:
   用 Nec / Suf / Ret 证明候选 path 是否有因果作用。

3. SMFA / retain anchor:
   用 retain anchor 区分 harmful identity path 与 benign/shared topic path。
```

严格 P_forget 需要同时满足：

```text
Suf > 0
forget_effect > forget_threshold
Ret <= retain_threshold
saliency_specificity_margin > 0
all_score_records_fully_patchable = true
```

本轮使用阈值：

```text
forget_threshold = 0.015625
retain_threshold = 0.005208333333333333
saliency_gamma = 1.0
min_saliency_specificity = 0.0
```

---

## 2. 20-Pair Subset

输入 pair 文件：

```text
mip_workspace/outputs/causal_pairs_train_scienceq1_20pairs_0527.jsonl
```

pair 数量：

```text
20
```

pair ids：

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

选择逻辑：

```text
一半来自历史低 Ret / P_forget-like 候选；
一半来自 vision_text / projector Suf 较高的候选。
```

目的不是随机估计总体分布，而是提高发现 forget-specific path 的概率。

---

## 3. 20-Pair Saliency Probe

输入分数文件：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl
```

由 4 个 chunk 合并：

```text
path_scores_scienceq1_saliency_20pairs_chunk00_0527.jsonl
path_scores_scienceq1_saliency_20pairs_chunk01_0527.jsonl
path_scores_scienceq1_saliency_20pairs_chunk02_0527.jsonl
path_scores_scienceq1_saliency_20pairs_chunk03_0527.jsonl
```

总体统计：

| metric | value |
| --- | ---: |
| records | 320 |
| ok | 320 |
| Suf > 0 | 244 |
| saliency_specificity_margin > 0 | 84 |
| Suf > 0 且 margin > 0 | 59 |

按 modality：

| modality | n | Suf > 0 | margin > 0 | both |
| --- | ---: | ---: | ---: | ---: |
| text | 80 | 31 | 17 | 7 |
| vision | 80 | 53 | 25 | 10 |
| vision_text | 160 | 160 | 42 | 42 |

解释：

```text
vision_text / projector path 的 Suf 很强，160/160 全部 Suf > 0。
但这不等于 P_forget，因为这些 path 也可能强烈影响 retain anchor。
```

---

## 4. Saliency-First Candidate Regeneration

命令：

```bash
MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace

/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.path_localization.saliency_specific_export \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_scienceq1_saliency_20pairs_0527.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --output_candidates "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl" \
  --output_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl" \
  --summary "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_20pairs_0527_summary.json" \
  --top_k_per_pair_modality 2 \
  --path_id_prefix saliency20
```

输出：

```text
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_20pairs_0527_summary.json
```

导出统计：

| metric | value |
| --- | ---: |
| input score records | 320 |
| filtered positive-specific records | 84 |
| selected pair-path bindings | 42 |
| selected unique candidate paths | 34 |

按 modality 的 selected bindings：

| modality | bindings |
| --- | ---: |
| text | 12 |
| vision | 14 |
| vision_text | 16 |

---

## 5. Step5 Causal Tracing on Regenerated Candidates

命令：

```bash
MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace

/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.causal_scores.build_scores \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory "$MIP_WORKSPACE_ROOT/llms/" \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs/" \
  --device cuda \
  --image_resize 224 \
  --pairs_path "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train_scienceq1_20pairs_0527.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_20pairs_0527.jsonl" \
  --path_pair_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_20pairs_bound_0527.jsonl" \
  --path_modality all \
  --compute_saliency_specificity \
  --saliency_gamma 1.0 \
  --output "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl"
```

输出：

```text
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl
```

Step5 统计：

| metric | value |
| --- | ---: |
| records | 42 |
| ok | 42 |
| Suf > 0 | 28 |
| saliency_specificity_margin > 0 | 42 |
| Suf > 0 且 margin > 0 | 28 |
| Ret <= retain_threshold | 20 |
| record-level strict forget-like | 8 |

按 modality：

| modality | n | Suf > 0 | margin > 0 | low Ret | record-level forget-like |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 12 | 5 | 12 | 10 | 4 |
| vision | 14 | 7 | 14 | 10 | 4 |
| vision_text | 16 | 16 | 16 | 0 | 0 |

关键解释：

```text
saliency-first generation 有效：
  所有 42 条重评分候选都保持 saliency_specificity_margin > 0。

但 vision_text / projector path 仍然不是 P_forget：
  vision_text 的 Suf 全部为正，但 low Ret = 0。
  这说明它们更像 shared cross-modal/topic path，而不是可安全删除的 harmful identity path。
```

---

## 6. Step6 Strict Classification

命令：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.causal_scores.classify_paths \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_step5_saliency_specific_20pairs_0527.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_saliency_specific_20pairs_0527" \
  --alpha 1.0 \
  --forget_threshold 0.015625 \
  --retain_threshold 0.005208333333333333 \
  --require_saliency_specificity \
  --saliency_specificity_key saliency_specificity_margin \
  --min_saliency_specificity 0.0
```

输出目录：

```text
mip_workspace/outputs/paths/step6_saliency_specific_20pairs_0527
```

分类结果：

| category | count |
| --- | ---: |
| P_forget | 4 |
| P_shared | 12 |
| P_retain | 2 |
| P_irrelevant | 16 |

按 classified path modality：

| modality | count |
| --- | ---: |
| text | 12 |
| vision | 14 |
| vision_text | 8 |

P_forget 的 modality 分布：

| modality | count |
| --- | ---: |
| vision | 4 |

---

## 7. Strict P_forget Paths

本轮得到 4 条严格 P_forget：

| path_id | pair | modality | Suf | Ret | saliency margin | forget saliency | retain anchor saliency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `saliency20_p000027` | `pair_000185` | vision | 0.0234375 | -0.0169271 | 0.0000020433 | 0.0000330731 | 0.0000310298 |
| `saliency20_p000028` | `pair_000185` | vision | 0.0234375 | -0.0182292 | 0.0000020036 | 0.0000334606 | 0.0000314570 |
| `saliency20_p000031` | `pair_000067` | vision | 0.0156250 | -0.0130208 | 0.0000009385 | 0.0000288475 | 0.0000279089 |
| `saliency20_p000033` | `pair_000067` | vision | 0.0312500 | -0.0039063 | 0.0000000438 | 0.0000472693 | 0.0000472255 |

共同特征：

```text
1. 都是 vision-only path。
2. 都不包含 projector。
3. 都 fully patchable。
4. Suf 为正，Ret 不高。
5. saliency_specificity_margin 为正，但 margin 很小。
```

这说明：

```text
当前 pipeline 可以找到少量目标知识相关的因果 path。
但这些 path 主要在 vision branch，而不是跨模态 projector branch。
```

---

## 8. P_shared 对照

代表性的 P_shared：

| path_id | modality | pairs | Suf | Ret | saliency margin | contains projector |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `saliency20_p000000` | vision_text | `pair_000025,pair_000030,pair_000138,pair_000166` | 2.15625 | 0.892578 | 0.0000706877 | true |
| `saliency20_p000001` | vision_text | `pair_000025,pair_000030,pair_000138,pair_000166` | 2.15625 | 0.892578 | 0.0000684283 | true |
| `saliency20_p000004` | vision_text | `pair_000089,pair_000155` | 2.62305 | 0.840495 | 0.0000195429 | true |
| `saliency20_p000007` | vision_text | `pair_000089` | 2.52344 | 0.727865 | 0.0000283855 | true |

P_shared 平均值：

| category | n | mean Suf | mean Ret | mean saliency margin |
| --- | ---: | ---: | ---: | ---: |
| P_shared | 12 | 1.54639 | 0.55995 | 0.0000245248 |

解释：

```text
vision_text / projector path 的因果作用非常强，
但 retain impact 也非常强。

所以它们不是可以直接删除的 harmful identity path，
更像是 shared cross-modal grounding / topic path。
```

这正是科学问题 1 要解决的核心区分：

```text
有 causal effect 不等于应该忘记；
必须同时看 retain anchor。
```

---

## 9. 对科学问题 1 的当前回答

科学问题 1：

```text
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分 harmful identity path 和 benign/shared topic path？
```

当前可以回答到以下程度：

```text
可以初步回答，但还不能完全回答。
```

已经成立的部分：

```text
1. 只用 Nec 不够，必须加入 Suf。
2. 只看 Suf 也不够，必须加入 Ret / retain anchor。
3. saliency_specificity_margin 可以作为候选生成前置筛选。
4. Suf > 0 且 Ret 高的 projector / vision_text path 应归入 P_shared。
5. Suf > 0、Ret 低、saliency-specific、fully patchable 的 path 可以作为 P_forget。
```

仍未成立的部分：

```text
1. P_forget 只覆盖 2/20 个 pair，覆盖率太低。
2. P_forget 只出现在 vision branch，跨模态 harmful identity path 还没有被分离出来。
3. saliency margin 数值很小，稳定性需要 bootstrap / repeat seed 验证。
4. text path 有 record-level forget-like 信号，但多在阈值边界或聚合后未进入 P_forget。
5. 当前 candidate 是 path-level reranking，不是 neuron/dim-level specificity mining。
```

---

## 10. 当前主要问题

## 10.1 Cross-modal projector path 仍然被 retain anchor 判为 shared

现象：

```text
vision_text:
  n = 16
  Suf > 0 = 16
  saliency_specificity_margin > 0 = 16
  low Ret = 0
  P_forget = 0
```

判断：

```text
projector path 确实承载跨模态信息，
但当前粒度下它更像共享语义/主题通路，而不是可安全删除的 identity-specific 通路。
```

下一步不能直接编辑这些 projector path，否则大概率破坏 retain。

## 10.2 P_forget 覆盖率不足

现象：

```text
20 个 pair 中，最终严格 P_forget 只覆盖 pair_000185 和 pair_000067。
```

判断：

```text
当前 protocol 可以发现 positive case，
但 recall 明显不足。
```

这说明下一步需要提高候选生成的分辨率，而不是直接扩大 Step7 编辑。

## 10.3 Path-level saliency specificity 太粗

当前 saliency-first Step2 做的是：

```text
整条 path 的 forget_saliency - retain_anchor_saliency
```

问题：

```text
一条 path 内可能同时包含 harmful identity neuron 和 benign topic neuron。
整条 path 平均后，projector/cross-modal path 很容易被 retain anchor 拉成 P_shared。
```

这解释了为什么：

```text
vision_text Suf 很强，但 Ret 也很强；
text/vision 有些低 Ret，但 forget_effect 又经常卡在阈值边界。
```

---

## 11. 下一步优化建议

不建议现在进入总体优化或 Step7 大规模编辑。

推荐下一步仍然优先修科学问题 1，但从 path-level 推进到 component-level：

```text
第一优先级：
  做 neuron/dim-level specificity mining。

第二优先级：
  对当前 4 条 P_forget 做 repeated causal validation。

第三优先级：
  再专门进入科学问题 2，处理 projector 内部的 harmful identity 子空间。
```

具体执行顺序：

```text
1. 对 20-pair 的 positive Suf + high Ret vision_text path 做 projector dim-level 分解。
   目标：在 shared projector path 内找 identity-specific dims。

2. 对 text / vision path 做节点级 ablation / leave-one-layer-out。
   目标：把 32/36 节点长 path 压缩成少量真正有贡献的节点。

3. 对当前 4 条 P_forget 做 repeat validation。
   目标：确认不是单次 sampling 或阈值噪声。

4. 扩展到 50 pair 前，先把 strict P_forget coverage 从 2/20 提高到至少 6/20。

5. 只有当 P_forget 稳定后，再进入 Step7 selective editing。
```

建议停止条件：

```text
如果 20-pair 上仍只能得到 vision-only P_forget，
则说明科学问题 2 必须提前介入：
需要把 projector path 从整路径级别拆到 dim/subspace 级别。
```

---

## 12. Decision

当前阶段的决策：

```text
继续优先科学问题 1。
不要直接总体优化。
不要直接扩大 Step7 编辑。
```

下一步最合理的任务定义：

```text
实现 projector / neuron dim-level saliency-specific candidate generation：

score(node_or_dim) =
  forget_saliency(node_or_dim)
  - gamma * retain_anchor_saliency(node_or_dim)

然后再做 causal tracing：
  Suf > 0
  Ret <= retain_threshold
  saliency_specificity_margin > 0
```

目标：

```text
把当前的 P_shared projector path 拆开，
找出其中真正 harmful identity-specific 的子路径或子空间。
```

