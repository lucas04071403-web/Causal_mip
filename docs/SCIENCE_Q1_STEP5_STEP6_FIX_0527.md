# Science Q1 Step5/Step6 修正记录

更新时间：2026-05-27

本文记录“第一阶段：优先修科学问题 1”的第一批代码修改。目标是让 Step5/Step6 不再把缺少 sufficiency 证据的路径静默当作 `P_forget`，并为后续诊断 `Suf=0` 的根因提供更完整字段。

## 1. 背景问题

`step9_bound_projector_0526_214028` 的历史 Step5/Step6 结果中：

```text
Step5 records = 2720
status ok = 2720
positive Suf = 0
Step6 P_forget = 100
```

这说明 `P_forget` 实际主要由 `Nec` 驱动，而不是由 `Nec + Suf` 的充分因果证据共同支撑。

因此本轮修正的原则是：

```text
没有 positive Suf，不允许进入正式 P_forget。
```

---

## 2. Step5 新增诊断字段

修改文件：

```text
causal_mip/interventions/activation_cache.py
causal_mip/causal_scores/metrics.py
```

新增 path/node 诊断：

```text
num_skipped_nodes
all_nodes_patchable
contains_projector
projector_patchable
num_projector_nodes
num_patchable_projector_nodes
resolved_nodes
skipped_nodes
```

新增 score 顶层字段：

```text
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

目的：

- 直接看出每条 path 的节点是否被完整 patch。
- 直接看出 projector 是否存在并被成功 patch。
- 直接看出 `Suf = restored_score - corrupt_score` 为什么为正、为零或为负。

---

## 3. Step6 新增 P_forget eligibility

修改文件：

```text
causal_mip/causal_scores/classify_paths.py
```

默认规则：

```text
P_forget 必须满足：
  aggregated Suf > min_forget_sufficiency
  all_score_records_fully_patchable = true
```

默认参数：

```text
min_forget_sufficiency = 0.0
require_positive_forget_sufficiency = true
require_full_patchable_forget = true
```

如果原本按阈值会进入 `P_forget`，但不满足 eligibility，则会被降级到 `P_irrelevant`，并记录：

```json
{
  "pre_eligibility_category": "P_forget",
  "demoted_from": "P_forget",
  "forget_eligibility": {
    "eligible": false,
    "reasons": ["sufficiency_not_positive"]
  }
}
```

兼容旧口径的开关：

```bash
--allow_zero_sufficiency_forget
--allow_partial_patchable_forget
```

这些开关只应用于历史对照，不建议用于正式科学结论。

---

## 4. 历史 Step9 score 的新分类结果

使用默认科学问题 1 guard 规则重跑 Step6：

```bash
python causal_mip/causal_scores/classify_paths.py \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_bound_suf_projector_step9_bound_projector_0526_214028_scienceq1_guard_0527" \
  --alpha 1.0 \
  --forget_quantile 0.75 \
  --retain_quantile 0.75 \
  --min_forget_effect 0.0 \
  --min_retain_impact 0.0
```

结果：

```text
P_forget = 0
P_shared = 110
P_retain = 221
P_irrelevant = 1037
num_demoted_from_forget = 100
```

解释：

```text
旧 P_forget 的 100 条路径全部因为 Suf=0 被降级。
```

这符合当前科学判断：历史 Step9 的 `P_forget` 不能作为可靠目标知识因果路径集合。

使用兼容旧口径：

```bash
python causal_mip/causal_scores/classify_paths.py \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_bound_suf_projector_step9_bound_projector_0526_214028_compat_0527" \
  --alpha 1.0 \
  --forget_quantile 0.75 \
  --retain_quantile 0.75 \
  --allow_zero_sufficiency_forget
```

可复现旧分类规模：

```text
P_forget = 100
P_shared = 110
P_retain = 221
P_irrelevant = 937
```

---

## 5. 测试

已运行：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile \
  causal_mip/interventions/activation_cache.py \
  causal_mip/causal_scores/metrics.py \
  causal_mip/causal_scores/classify_paths.py \
  causal_mip/test_step5_causal_scores.py \
  causal_mip/test_step6_classify_paths.py

/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step5_causal_scores.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step6_classify_paths.py
```

结果：

```text
Step 5 causal-score tests passed.
Step 6 path-classification tests passed.
```

---

## 6. 下一步

这次修改只是建立了科学问题 1 的“安全门”：

```text
Suf=0 时不再允许进入正式 P_forget。
```

下一步仍需修复 Step5 sufficiency 本身，重点检查：

```text
corrupt input 是否真的破坏目标身份证据
restore 的 clean activation 是否 patch 到正确 token/module
target answer/name token logprob 是否和 Step8 name-hit 指标对齐
projector/image-token restoration 是否需要单独计分
```

只有当新 Step5 产生 positive Suf，并且 `P_forget` 中路径满足：

```text
Nec 高
Suf 正
Ret 低
节点完整 patchable
```

才应进入下一轮 Step7 主实验。
