# Science Q1 Main Report

更新时间：2026-05-29

本文是 Science Q1 的主报告，合并并取代旧的 Q1 分散文档。详细实验流水与产物见 `SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md`，下一步执行计划见 `SCIENCE_Q1_NEXT_STEPS.md`。

## 1. 科学问题

Science Q1：

```text
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分 harmful identity path 和 benign/shared topic path？
```

当前总判断：

```text
已经建立了更可靠的判定协议，并找到少量符合协议的候选；
但还没有稳定解决“哪些路径”这个定位问题，也不能声明 Step7 已经成功遗忘。
```

更具体地说：

1. `Nec` 只能说明 ablation 会影响输出，不能单独证明路径承载目标知识。
2. `Suf > 0` 是进入正式 `P_forget` 的必要条件；没有 positive sufficiency 的路径不能作为目标知识因果路径。
3. `Ret` 和 retain-anchor saliency 必须参与分类；高 `Suf` 但高 `Ret` 的 projector / vision_text path 应归为 `P_shared`。
4. SalUn/SSD 风格的 `forget_saliency - retain_anchor_saliency` 能过滤 retain-biased path。
5. 当前 path-level 方法能发现少量 positive case，但覆盖率低，且 projector path 仍主要是 shared trunk。
6. 2026-05-29 的 Step7 只完成了 `formal stable single-candidate smoke`，验证了链路和最小编辑，不是正式大规模 Step7。

## 2. 判定协议

当前有效协议：

```text
SalUn/SSD:
  用 forget_saliency - gamma * retain_anchor_saliency
  生成 forget-specific candidate path。

CaMU / causal tracing:
  用 Nec / Suf / Ret 验证 path 是否真有因果作用。

SMFA / retain anchor:
  用 same_topic / same_reasoning / counterfactual_retain
  区分 harmful identity path 和 benign/shared topic path。
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

主要阈值：

| field | value |
| --- | ---: |
| `forget_threshold` | 0.015625 |
| `retain_threshold` | 0.005208333333333333 |
| `saliency_gamma` | 1.0 |
| `min_saliency_specificity` | 0.0 |
| repeat `min_pass_rate` | 0.8 |

后续更严格版本不应只用 `margin > 0`，还应加入 ratio 与 anchor cap：

```text
saliency_specificity_margin > tau_margin
saliency_specificity_ratio > tau_ratio
retain_anchor_saliency < tau_anchor
```

## 3. 当前证据

### 3.1 旧 Step9 P_forget 被推翻

历史 `step9_bound_projector_0526_214028`：

| metric | value |
| --- | ---: |
| Step5 records | 2720 |
| positive `Suf` | 0 |
| old Step6 `P_forget` | 100 |

加入 `Suf > 0` guard 后：

| category | count |
| --- | ---: |
| `P_forget` | 0 |
| `P_shared` | 110 |
| `P_retain` | 221 |
| `P_irrelevant` | 1037 |
| `demoted_from_forget` | 100 |

结论：历史 100 条 `P_forget` 全部缺少 positive sufficiency，不能作为可靠目标知识因果路径。

### 3.2 Step5 sufficiency 已修复

修复后：

| metric | value |
| --- | ---: |
| records | 2720 |
| positive `Suf` | 1807 |
| vision_text positive `Suf` | 1360 / 1360 |

这说明原先 `Suf=0` 不是模型中没有充分因果路径，而是 Step5 计分/patch 口径没有正确暴露它们。

### 3.3 retain-aware saliency gate 有效

`pair_000002 + pair_000064` 对照：

| setting | `P_forget` | `P_shared` | `P_retain` | `P_irrelevant` |
| --- | ---: | ---: | ---: | ---: |
| no saliency gate | 5 | 9 | 1 | 9 |
| with saliency gate | 0 | 9 | 1 | 14 |

这些路径按 `Nec/Suf/Ret` 会进入 `P_forget`，但 `retain_anchor_saliency > forget_saliency`。saliency gate 正确阻止了 retain-biased path 被误判为 harmful identity path。

### 3.4 20-pair 首次产生严格 P_forget

20-pair strict Step6：

| category | count |
| --- | ---: |
| `P_forget` | 4 |
| `P_shared` | 12 |
| `P_retain` | 2 |
| `P_irrelevant` | 16 |

严格 `P_forget`：

| path_id | pair | modality | `Suf` | `Ret` | saliency margin |
| --- | --- | --- | ---: | ---: | ---: |
| `saliency20_p000027` | `pair_000185` | vision | 0.0234375 | -0.0169271 | 0.0000020433 |
| `saliency20_p000028` | `pair_000185` | vision | 0.0234375 | -0.0182292 | 0.0000020036 |
| `saliency20_p000031` | `pair_000067` | vision | 0.0156250 | -0.0130208 | 0.0000009385 |
| `saliency20_p000033` | `pair_000067` | vision | 0.0312500 | -0.0039063 | 0.0000000438 |

结论：protocol 不是只会拒绝候选，确实能找到少量 harmful-identity-like causal path。但它只覆盖 2/20 个 pair，且全是 vision-only。

### 3.5 projector / vision_text 当前主要是 P_shared

代表性 projector / vision_text path：

| path_id | modality | `Suf` | `Ret` | saliency margin | category |
| --- | --- | ---: | ---: | ---: | --- |
| `saliency20_p000000` | vision_text | 2.15625 | 0.892578 | 0.0000706877 | `P_shared` |
| `saliency20_p000004` | vision_text | 2.62305 | 0.840495 | 0.0000195429 | `P_shared` |
| `saliency20_p000007` | vision_text | 2.52344 | 0.727865 | 0.0000283855 | `P_shared` |

解释：这些 path 的 `Suf` 很强，说明它们确实承载目标相关信息；但 `Ret` 也很高，说明它们同时承载 retain / shared topic 信息。当前不应直接编辑整条 projector path。

## 4. 2026-05-29 最新状态

### 4.1 retain-aware node/projector-dim 结果

在 20-pair 上进一步做 retain-aware node-level 和 projector dim-level 筛选：

| stage | result |
| --- | ---: |
| node-specific candidates | 10 |
| projector-dim candidates | 9 |
| merged dedup candidates | 19 |
| Step5 ok records | 19 |
| `Suf > 0` | 9 |
| low `Ret` | 13 |
| positive worst-anchor margin | 13 |

Step6 pair-level + SSD gate：

| category | count |
| --- | ---: |
| `P_forget` | 1 |
| `P_shared` | 3 |
| `P_retain` | 3 |
| `P_irrelevant` | 12 |

Repeat stability across 3 anchors：

| metric | value |
| --- | ---: |
| report records | 19 |
| stable candidates | 1 |

唯一 formal stable candidate：

| field | value |
| --- | --- |
| `path_id` | `projdimRA20d_p000003` |
| `pair_id` | `pair_000089` |
| modality | `vision_text` |
| original path | `saliency20_p000004` |
| selected dims | `[387, 318, 886, 984]` |
| repeat pass count | 3/3 |
| `Suf.mean` | 0.03125 |
| `Ret.mean` | -0.00390625 |
| `Ret.max` | 0.00390625 |

这说明 dim-level 方向能从 high-Suf high-Ret projector shared path 中拆出一个更 specific 的候选，但当前只有 1 条 stable candidate。

### 4.2 Step7 single-candidate smoke

run id：

```text
step7_single_stable_smoke_retainaware_0529_152017
```

性质：

```text
formal stable single-candidate smoke
```

它不是大规模 Step7，不是正式 unlearning 结果，也不构成 Science Q1 的行为层结论。

Step7 结果：

| field | value |
| --- | ---: |
| `num_loss_records` | 5 |
| `mask_summary.num_modules` | 1 |
| `mask_summary.num_editable_neurons` | 4 |
| skipped modules | 0 |

mask 解析：

| field | value |
| --- | --- |
| logical module | `mm_projector` |
| trace module | `visual.merger.mlp.0` |
| edit module | `visual.merger.mlp.0` |
| editable dims | `[318, 387, 886, 984]` |

Step8 single-pair diagnostic：

| sample | baseline | smoke | 判断 |
| --- | --- | --- | --- |
| `forget_clean` | 未命中目标姓名 | 仍未命中 | 无可判定新增遗忘信号 |
| `same_topic` | 场景基本正确 | 类似输出 | 无明显恶化 |
| `same_reasoning` | 错误姓名 | 完全相同 | 无变化 |
| `counterfactual_retain` | 错误姓名 `Simon Makoni` | 正确姓名 `Ismail Jengo` | 唯一明确正向信号 |

结论：

```text
有一个 retain-side 正向方向信号，
但没有 forget-side 可判定信号。
```

## 5. 当前未解决问题

1. `P_forget` recall 低：20-pair strict path-level 只覆盖 2/20；最新 stable dim-level 只有 1 条。
2. path-level `P_forget` 全部是 vision-only，不能充分解释跨模态目标知识如何进入语言输出。
3. projector path causality 很强，但 Ret 也强；当前大多是 shared trunk，需要 dim/subspace 拆分。
4. 旧 4 条 path-level `P_forget` saliency margin 很小，需要 repeat validation；最新 stable dim-level candidate 虽通过 repeat，但样本量只有 1。
5. Step7 smoke 只证明链路可跑和 dim 级编辑可落点，不能证明遗忘成功。
6. name-hit 在 `pair_000089` baseline 已经为 0，无法判断 smoke 是否进一步压低 forget target。

## 6. 当前决策

不建议：

1. 宣称 Science Q1 已解决。
2. 直接扩大 Step7 主实验。
3. 直接编辑 high-Suf high-Ret projector path。
4. 用当前 single-candidate smoke 作为 unlearning 成功证据。

建议：

1. 继续优先 Science Q1 的定位精度。
2. 将 path-level specificity 推进到 pair-level、node-level、dim-level。
3. 对 stable candidate 增加 pre/post exact target name probability 或 scoped CE diagnostic。
4. 只有当出现稳定 forget-side 下降且 retain-side 不退化后，再进入更大规模 Step7。

