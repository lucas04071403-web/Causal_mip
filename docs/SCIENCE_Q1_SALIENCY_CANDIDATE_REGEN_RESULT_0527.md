# Science Q1 Saliency-First Candidate Generation 结果

更新时间：2026-05-27

本次任务：

```text
重做 candidate generation，
让 Step2 先按 forget_saliency - gamma * retain_anchor_saliency 生成候选，
再做 causal tracing。
```

结论：

```text
已实现并跑通 saliency-first Step2 -> Step5 -> Step6。

新的候选确实满足 saliency_specificity_margin > 0，
并且 causal tracing 后多数 Suf > 0。

但仍没有产生严格 P_forget。
```

主要原因：

```text
当前能找到的 saliency-specific cross-modal/projector path 仍然 Ret 很高，
所以进入 P_shared；
text/vision path 虽然 specificity 为正，但 forget_effect 太弱或 retain_impact 超阈值。
```

---

## 1. 新增代码

新增：

```text
causal_mip/path_localization/saliency_specific_export.py
```

接入：

```text
causal_mip/path_localization/__init__.py
docs/PROJECT_STRUCTURE.md
causal_mip/test_step2.py
```

功能：

```text
输入：
  Step5 saliency score JSONL
  原始 P_cand.jsonl

输出：
  新的 saliency-specific P_cand JSONL
  新的 path-pair binding JSONL
  summary JSON
```

筛选规则：

```text
specificity_score = forget_saliency - gamma * retain_anchor_saliency

默认要求：
  saliency_specificity_margin > 0
  all_nodes_patchable = true
  vision_text path 的 projector_patchable = true
```

---

## 2. Step2 Saliency-Specific Candidate Export

使用输入：

```text
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair155_0527.jsonl
mip_workspace/outputs/scores/path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl
mip_workspace/outputs/paths/P_cand.jsonl
```

命令：

```bash
MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace

/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.path_localization.saliency_specific_export \
  --scores_path \
    "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_scienceq1_saliency_pair155_0527.jsonl" \
    "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_scienceq1_saliency_pair000002_000064_0527.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --output_candidates "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_0527.jsonl" \
  --output_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_bound_0527.jsonl" \
  --summary "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_0527_summary.json" \
  --top_k_per_pair_modality 2 \
  --path_id_prefix saliency_specific
```

输出：

```text
mip_workspace/outputs/paths/P_cand_saliency_specific_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_bound_0527.jsonl
mip_workspace/outputs/paths/P_cand_saliency_specific_0527_summary.json
```

summary：

```text
num_score_records = 48
num_pair_path_scores = 48
num_filtered_pair_path_scores = 14
num_selected_bindings = 6
num_selected_candidate_paths = 6
modalities:
  text = 2
  vision = 2
  vision_text = 2
```

被选中的 6 条候选：

| new path | original path | modality | specificity margin |
| --- | --- | --- | ---: |
| `saliency_specific_p000000` | `cross_modal_p000000` | vision_text | 0.0000157381 |
| `saliency_specific_p000001` | `cross_modal_p000007` | vision_text | 0.0000090624 |
| `saliency_specific_p000002` | `vision_fisher_p000620` | vision | 0.0000048033 |
| `saliency_specific_p000003` | `vision_fisher_p000623` | vision | 0.0000044353 |
| `saliency_specific_p000004` | `text_igi_p000623` | text | 0.0000014170 |
| `saliency_specific_p000005` | `text_igi_p000620` | text | 0.0000014056 |

注意：

```text
所有被选中的候选都来自 pair_000155。
```

原因是：

```text
pair_000002 + pair_000064 的 saliency_specificity_margin 全部为负，
被 saliency-first Step2 正确过滤。
```

---

## 3. Step5 Causal Tracing on New Candidates

命令：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.causal_scores.build_scores \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory "$MIP_WORKSPACE_ROOT/llms/" \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs/" \
  --device cuda \
  --image_resize 224 \
  --pairs_path "$MIP_WORKSPACE_ROOT/outputs/causal_pairs_train.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_0527.jsonl" \
  --path_pair_bindings "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand_saliency_specific_bound_0527.jsonl" \
  --path_modality all \
  --pair_start 114 \
  --num_pairs 1 \
  --compute_saliency_specificity \
  --saliency_gamma 1.0 \
  --output "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_step5_saliency_specific_candidates_0527.jsonl"
```

输出：

```text
mip_workspace/outputs/scores/path_scores_step5_saliency_specific_candidates_0527.jsonl
```

Step5 统计：

```text
num_records = 6
status ok = 6
Suf > 0 = 5
saliency_specificity_margin > 0 = 6
Suf > 0 且 saliency_specificity_margin > 0 = 5
```

modality：

| modality | n | Suf > 0 | saliency margin > 0 | projector | projector_patchable |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 2 | 2 | 2 | 0 | 0 |
| vision | 2 | 1 | 2 | 0 | 0 |
| vision_text | 2 | 2 | 2 | 2 | 2 |

---

## 4. Step6 Classification

命令：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.causal_scores.classify_paths \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_step5_saliency_specific_candidates_0527.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_saliency_specific_candidates_0527" \
  --alpha 1.0 \
  --forget_threshold 0.015625 \
  --retain_threshold 0.005208333333333333 \
  --require_saliency_specificity \
  --saliency_specificity_key saliency_specificity_margin \
  --min_saliency_specificity 0.0
```

输出：

```text
mip_workspace/outputs/paths/step6_saliency_specific_candidates_0527
```

分类：

```text
P_forget = 0
P_shared = 3
P_retain = 1
P_irrelevant = 2
```

明细：

| path | modality | Nec | Suf | Ret | forget_effect | retain_impact | saliency margin | category |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `saliency_specific_p000000` | vision_text | 1.48828125 | 2.72265625 | 0.953125 | 4.2109375 | 0.953125 | 0.0000157381 | `P_shared` |
| `saliency_specific_p000001` | vision_text | 1.48828125 | 2.72265625 | 0.953125 | 4.2109375 | 0.953125 | 0.0000090624 | `P_shared` |
| `saliency_specific_p000003` | vision | 0.015625 | 0.015625 | 0.01171875 | 0.03125 | 0.01171875 | 0.0000044353 | `P_shared` |
| `saliency_specific_p000002` | vision | 0.015625 | 0.0 | 0.01041667 | 0.015625 | 0.01041667 | 0.0000048033 | `P_retain` |
| `saliency_specific_p000004` | text | 0.0 | 0.015625 | 0.0 | 0.015625 | 0.0 | 0.0000014170 | `P_irrelevant` |
| `saliency_specific_p000005` | text | 0.0 | 0.015625 | 0.00520833 | 0.015625 | 0.00520833 | 0.0000014056 | `P_irrelevant` |

---

## 5. 科学解释

这轮实验说明：

```text
SalUn/SSD-first candidate generation 生效了。
```

证据：

```text
新候选全部 saliency_specificity_margin > 0。
Step5 后 5/6 同时满足 Suf > 0 和 saliency_specificity_margin > 0。
vision_text projector path 完整 patchable。
```

但它仍不能给出 `P_forget`：

```text
最强的 vision_text / projector path:
  forget_effect 很高
  specificity 为正
  但 retain_impact 极高
  -> P_shared

text/vision path:
  specificity 为正
  Suf 有正值
  但 forget_effect 太弱或 retain_impact 达到阈值
  -> P_irrelevant / P_retain / P_shared
```

因此，当前科学问题 1 的状态是：

```text
可以区分 shared path；
还没有定位到 harmful identity-only path。
```

这比旧结论更可靠：

```text
旧 Step6 会把部分 low-Ret causal path 误当 P_forget；
新 saliency-first 路线显示这些 path 不是稳定的 forget-specific path。
```

---

## 6. 下一步

不要进入 Step7 主实验。

下一步应扩大 saliency-first candidate generation 的覆盖范围：

```text
1. 对 10-20 个 pair 跑 Step5 saliency probe。
2. 用 saliency_specific_export 生成更大的 P_cand_saliency_specific。
3. 重新跑 Step5 causal tracing。
4. 检查是否出现：
   P_forget > 0
   Suf > 0
   saliency_specificity_margin > 0
   retain_impact <= threshold
```

如果扩大后仍然没有 `P_forget`：

```text
当前 candidate pool 的 neuron/path 级别仍太粗；
需要把 specificity 计算下沉到 projector dim / neuron dim，
而不是只在已有 path 上重排序。
```

具体优化方向：

```text
projector_dim_specificity = forget_projector_dim_saliency - gamma * retain_projector_dim_saliency
text_neuron_specificity = forget_text_neuron_saliency - gamma * retain_text_neuron_saliency
vision_neuron_specificity = forget_vision_neuron_saliency - gamma * retain_vision_neuron_saliency
```

再用这些 dim/neuron 重新组合 cross-modal path。

---

## 7. 测试

已运行：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile \
  causal_mip/path_localization/saliency_specific_export.py \
  causal_mip/path_localization/__init__.py \
  causal_mip/test_step2.py

/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step2.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step5_causal_scores.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step6_classify_paths.py
```

结果：

```text
Step2 tests passed.
Step5 causal-score tests passed.
Step6 path-classification tests passed.
```
