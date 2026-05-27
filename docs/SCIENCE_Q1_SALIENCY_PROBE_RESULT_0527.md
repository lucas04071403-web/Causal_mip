# Science Q1 Saliency Specificity 小规模实验结果

更新时间：2026-05-27

本次任务：

```text
小规模重跑带 --compute_saliency_specificity 的 Step5/Step6，
确认是否能产生 Suf > 0 且 saliency-specific 的 P_forget。
```

结论：

```text
没有产生满足当前 Science Q1 严格定义的 P_forget。
```

但这不是因为 Step5 无法计算 saliency 或 sufficiency。相反：

```text
Suf > 0 可以产生；
saliency_specificity 可以正常计算；
projector / vision_text path 可以完整 patch；
真正失败点是：满足 causal forget effect 的候选路径不是 forget-specific。
```

---

## 1. 运行环境

代码目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
```

workspace：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

Python：

```text
/home/lucas/miniconda3/envs/mip-editor/bin/python
```

GPU：

```text
NVIDIA GeForce RTX 4090
```

模型：

```text
mip_workspace/llms/Qwen2.5-VL-3B-Instruct
```

PEFT checkpoint：

```text
mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
```

输入：

```text
mip_workspace/outputs/causal_pairs_train.jsonl
mip_workspace/outputs/paths/P_cand.jsonl
mip_workspace/outputs/paths/P_cand_bound_train_step9_bound_projector_0526_214028.jsonl
```

每个 pair 绑定：

```text
16 paths = 4 text + 4 vision + 8 vision_text
```

---

## 2. Probe A: pair_000155

选择理由：

```text
历史 suf_fix_step9 中 pair_000155 的 vision_text Suf 最高。
```

Step5 输出：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair155_0527.jsonl
```

运行规模：

```text
num_pairs = 1
num_records = 16
```

Step5 统计：

```text
status ok = 16 / 16
Suf > 0 = 11 / 16
saliency_status ok = 16 / 16
saliency_specificity_margin > 0 = 14 / 16
Suf > 0 且 saliency_specificity_margin > 0 = 11 / 16
```

modality：

| modality | n | Suf > 0 | projector | projector_patchable |
| --- | ---: | ---: | ---: | ---: |
| text | 4 | 2 | 0 | 0 |
| vision | 4 | 1 | 0 | 0 |
| vision_text | 8 | 8 | 8 | 8 |

代表性 vision_text 结果：

| path | Suf | Ret | saliency margin | category |
| --- | ---: | ---: | ---: | --- |
| `cross_modal_p000000` | 2.72265625 | 0.953125 | 0.0000157381 | `P_shared` |
| `cross_modal_p000003` | 2.72265625 | 0.94531250 | 0.0000049894 | `P_shared` |
| `cross_modal_p000007` | 2.72265625 | 0.953125 | 0.0000090624 | `P_shared` |

Step6 使用全量 `suf_fix_step9` 阈值近似：

```text
forget_threshold = 0.015625
retain_threshold = 0.005208333333333333
require_saliency_specificity = true
```

Step6 输出：

```text
mip_workspace/outputs/paths/step6_scienceq1_saliency_pair155_global_threshold_0527
```

分类：

```text
P_forget = 0
P_shared = 9
P_retain = 2
P_irrelevant = 5
```

解释：

```text
pair_000155 的 projector / cross-modal path 同时满足：
  Suf > 0
  saliency_specificity_margin > 0
  projector_patchable = true

但 Ret 非常高，大约 0.945-0.953。

因此它们不是纯 harmful identity path，
而是 harmful identity 与 retain/benign topic 共用的 shared path。
```

---

## 3. Probe B: pair_000002 + pair_000064

选择理由：

```text
历史 suf_fix_step9 中这两个 pair 含有低 Ret 的 P_forget 候选。
```

Step5 输出：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl
```

运行规模：

```text
num_pairs = 2
num_records = 32
```

Step5 统计：

```text
status ok = 32 / 32
Suf > 0 = 26 / 32
saliency_status ok = 32 / 32
saliency_specificity_margin > 0 = 0 / 32
Suf > 0 且 saliency_specificity_margin > 0 = 0 / 32
```

平均 saliency：

```text
avg_forget_saliency        = 0.0000532902
avg_retain_anchor_saliency = 0.0000801873
avg_margin                 = -0.0000268971
min_margin                 = -0.0000744409
max_margin                 = -0.0000022757
```

不启用 saliency gate 的 Step6 对照：

```text
mip_workspace/outputs/paths/step6_scienceq1_saliency_pair000002_000064_global_threshold_no_salgate_0527

P_forget = 5
P_shared = 9
P_retain = 1
P_irrelevant = 9
```

启用 saliency gate 的 Step6：

```text
mip_workspace/outputs/paths/step6_scienceq1_saliency_pair000002_000064_global_threshold_0527

P_forget = 0
P_shared = 9
P_retain = 1
P_irrelevant = 14
num_demoted_from_forget = 5
```

被 saliency gate 降级的候选：

| path | modality | Suf | Ret | saliency margin | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `text_igi_p000258` | text | 0.0234375 | 0.0 | -0.0000022757 | `saliency_specificity_too_low` |
| `vision_fisher_p000010` | vision | 0.0781250 | -0.0052083 | -0.0000209346 | `saliency_specificity_too_low` |
| `vision_fisher_p000257` | vision | 0.0312500 | -0.0026042 | -0.0000121743 | `saliency_specificity_too_low` |
| `vision_fisher_p000258` | vision | 0.0234375 | 0.0 | -0.0000079004 | `saliency_specificity_too_low` |
| `vision_fisher_p000259` | vision | 0.0468750 | 0.0026042 | -0.0000196304 | `saliency_specificity_too_low` |

解释：

```text
这些路径按 Nec/Suf/Ret 会进入 P_forget；
但 retain_anchor_saliency > forget_saliency，
说明它们对 retain anchors 的梯度显著性更高。

按 SalUn/SSD + SMFA 的科学问题 1 口径，
它们不能叫 harmful identity-specific path。
```

---

## 4. 对科学问题 1 的回答

本次小规模实验后，当前答案是：

```text
还不能正确回答“哪些路径是真正承载目标知识的 harmful identity-specific causal path”。
```

更精确地说：

```text
当前系统能找到 causal path：
  Nec/Suf 有效，Suf > 0 已经出现。

当前系统能区分 shared path：
  pair_000155 的 projector / cross-modal path 是典型 P_shared。

当前系统尚未找到严格 P_forget：
  low-Ret causal candidates 的 saliency specificity 全为负。
```

因此：

```text
P_forget = 0 不是坏消息；
它说明新的 Science Q1 guard 起作用了。
旧规则会把非 forget-specific 的路径误当成 P_forget。
```

---

## 5. 当前存在的问题

### 5.1 Retain anchor 太强，暴露出候选路径不是 identity-specific

低 Ret 的 text/vision 候选：

```text
Suf > 0
Ret 低
但 saliency_specificity_margin < 0
```

这说明这些路径不是 forget-specific，而是对 retain anchors 更敏感。

### 5.2 Projector / vision_text 路径是 shared，不是 forget-only

pair_000155 中：

```text
vision_text:
  Suf 高
  saliency margin 正
  projector patchable
  但 Ret 高
```

因此 projector 当前更像跨模态共享语义/视觉主题路径，不应直接强编辑。

### 5.3 当前 Step2 candidate 仍未使用 saliency specificity 生成

本次 saliency 是在已有 candidate path 上追加评分。

下一步应反过来：

```text
先用 saliency specificity 生成 candidate，
再做 causal tracing。
```

否则候选集合可能天然偏 shared utility path。

---

## 6. 下一步建议

优先继续科学问题 1，不要进入 Step7 主实验。

下一步应做：

```text
1. 扩大 saliency probe 到 10-20 个 pair。
2. 统计是否存在：
   Suf > 0
   Ret <= retain_threshold
   saliency_specificity_margin > 0
3. 如果仍然没有 P_forget，
   重做 Step2 candidate generation：
   用 forget_saliency - gamma * retain_anchor_saliency 排 projector dim / text neuron / vision neuron。
4. 对 projector path 不直接编辑 P_shared，
   只作为弱 probe 或 retain-constrained target。
```

推荐下一个实验：

```text
Science Q1 candidate regeneration:
  SalUn/SSD specificity-first path mining
  -> CaMU causal validation
  -> SMFA retain-anchor split
```

验收标准：

```text
P_forget > 0
P_forget 中：
  Suf > 0
  saliency_specificity_margin > 0
  Ret <= threshold
  all_score_records_fully_patchable = true
```

如果该标准达不到，不能继续声明科学问题 1 已解决。

---

## 7. 已运行测试

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile \
  causal_mip/causal_scores/saliency_specificity.py

/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step5_causal_scores.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step6_classify_paths.py
```

结果：

```text
Step 5 causal-score tests passed.
Step 6 path-classification tests passed.
```
