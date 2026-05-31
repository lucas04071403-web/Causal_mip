# Step11 projector-dim target-name 优化与实验验证结果

更新时间：2026-05-31

本文档记录按照 `NEXT_ROUTE_TARGET_NAME_PROJECTOR_DIM_OPTIMIZATION.md` 执行后的优化、实验验证和下一轮建议。

结论先行：

```text
本轮有明确改善，但 cheap gate 仍未完全通过。

改善点：
1. top8 retain-aware projector-dim 把可编辑维度从旧 16 dims 扩到 31 dims。
2. forget_clean name_hit 最好从旧 0.7778 降到 0.5556。
3. forget_clean target_name_ce delta 从旧 +0.000038 提高到最高 +0.003591。
4. redacted_name_preference 已接入并通过训练/单测/实验验证。

未通过点：
1. target_name_ce delta 仍低于 gate 要求的 +0.01。
2. counterfactual_retain 在最佳 forget run 中仍略低于 0.6722。
3. 单纯加大 CE 到 0.50 会提高概率 delta，但生成 name_hit 反而变差。
```

因此当前不建议进入 Full CLEAR。下一轮应继续在 Step8 cheap gate 上做小网格，而不是扩大远程评估。

---

## 1. 本轮实际优化

### 1.1 top8 retain-aware projector-dim 扩容

新增 candidate 产物：

```text
../mip_workspace/outputs/paths/P_cand_projector_dim_retainaware_top8_dedup_20pairs_0531.jsonl
../mip_workspace/outputs/paths/P_cand_projector_dim_retainaware_top8_dedup_20pairs_bound_0531.jsonl
../mip_workspace/outputs/paths/P_cand_projector_dim_retainaware_top8_dedup_20pairs_0531_summary.json
```

Step6 name-aware 分类产物：

```text
../mip_workspace/outputs/paths/step6_name_aware_projector_dim_top8_0531/
```

分类结果：

| category | count | projector count |
| --- | ---: | ---: |
| P_forget | 4 | 4 |
| P_shared | 0 | 0 |
| P_retain | 1 | 1 |
| P_irrelevant | 4 | 4 |

P_forget：

```text
projdimRA20d8_p000006
projdimRA20d8_p000007
projdimRA20d8_p000008
projdimRA20d8_p000003
```

Step7 preflight 结果：

```text
module = mm_projector / visual.merger.mlp.0
editable dims = 31
uses_whole_vector_neuron = false
P_shared projector dims = 0
```

可编辑维度：

```text
34, 103, 146, 204, 229, 231, 275, 292, 318, 376,
387, 410, 420, 459, 490, 530, 608, 627, 640, 690,
714, 757, 769, 837, 886, 916, 983, 984, 993, 995, 1220
```

处理理由：

```text
旧 16-dim run 的行为影响太弱，说明 target-name 子空间覆盖不足或 objective 不够强。
top8 retain-aware projector-dim 在不退回 whole-vector projector 的前提下扩大编辑强度。
P_shared=0 且 whole-vector=false，说明这次没有再次发生 coarse projector 被 shared 抵消的问题。
```

### 1.2 redacted_name_preference 接入

代码改动：

| 文件 | 改动 |
| --- | --- |
| `clear.py` | 为样本 metadata 增加 `redacted_positive_answer`、`redacted_positive_token_ids`、`redacted_name_token_ids` |
| `main.py` | CLI choices 增加 `redacted_name_preference` |
| `causal_mip/editing/masked_rmisu.py` | 增加 redacted positive CE labels 与 objective 分支 |
| `causal_mip/test_step7_masked_rmisu.py` | 增加 redacted positive label 与 objective smoke test |

最终采用的 redacted positive 语义：

```text
负向：对原 target name token 做 CE ascent。
正向：只在 name span 上监督 redacted/generic name token，例如 "The person"。
```

处理理由：

```text
第一次全答案 redacted positive 会把整段 redacted answer token 填到原 answer positions。
这会带来两个问题：
1. token 对齐容易错位；
2. positive token_count 过大，positive loss 会抵消 target-name CE ascent。

因此修正为只在 name span 上提供替代身份方向。
这样仍然告诉模型不要输出目标姓名，但不会强迫整段答案结构完全改写。
```

---

## 2. 固定验证协议

所有实验使用同一套 Step8 cheap gate：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

验证命令类型：

```text
Step8 probability diagnostic:
causal_mip/evaluation/step8_probability_diagnostic.py

Step8 pair generation eval:
causal_mip/evaluation/step8_final_eval.py
```

处理理由：

```text
训练 loss 只能说明 objective 接入并参与优化，不能证明行为遗忘成功。
target_name_ce delta 直接衡量目标姓名 token 概率是否被压低。
name_hit 则衡量生成行为是否真的越过阈值。
retain name_hit 用来防止编辑只是在整体破坏身份识别能力。
```

---

## 3. 实验结果

| run | forget_clean target_name_ce delta | forget_clean name_hit | counterfactual_retain name_hit | hard_retain name_hit | gate | 说明 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `step10_16dim_name_pref` | +0.000038 | 0.7778 | 0.6667 | 0.6944 | FAIL | 旧 16-dim 对照 |
| `step11_top8_name_ce030` | +0.002199 | 0.5556 | 0.6667 | 0.7222 | FAIL | 当前最佳生成 forget 行为 |
| `step11_top8_redacted_full_a030_b010` | -0.001489 | 0.6111 | 0.7222 | 0.6944 | FAIL | 全答案 redacted positive，方向被抵消 |
| `step11_top8_name_ce050` | +0.003591 | 0.6667 | 0.7222 | 0.6944 | FAIL | 概率 delta 更高，但生成 forget 变差 |
| `step11_top8_redacted_namespan_a030_b002` | +0.002282 | 0.6111 | 0.6667 | 0.6944 | FAIL | 修正 name-span positive 后恢复正向压制 |

关键产物：

```text
../mip_workspace/outputs/masked_rmisu_step11_top8_name_ce030_200steps_0531.json
../mip_workspace/outputs/checkpoints/masked_rmisu_step11_top8_name_ce030_200steps_0531
../mip_workspace/outputs/diagnostics/step8_probability_diag_step11_top8_name_ce030_200steps_0531_val.json
../mip_workspace/outputs/step8_eval_step11_top8_name_ce030_200steps_0531_val.json

../mip_workspace/outputs/masked_rmisu_step11_top8_redacted_namespan_a030_b002_200steps_0531.json
../mip_workspace/outputs/checkpoints/masked_rmisu_step11_top8_redacted_namespan_a030_b002_200steps_0531
../mip_workspace/outputs/diagnostics/step8_probability_diag_step11_top8_redacted_namespan_a030_b002_200steps_0531_val.json
../mip_workspace/outputs/step8_eval_step11_top8_redacted_namespan_a030_b002_200steps_0531_val.json
```

---

## 4. 结果解释

### 4.1 31 dims 是有效方向

旧 16-dim name preference：

```text
forget_clean target_name_ce delta = +0.000038
forget_clean name_hit = 0.7778
```

top8 / 31-dim name CE：

```text
forget_clean target_name_ce delta = +0.002199
forget_clean name_hit = 0.5556
```

解释：

```text
扩大 retain-aware projector dims 后，生成层面的 target-name 命中率明显下降。
这说明问题不只是 objective，原 16 dims 的编辑覆盖确实偏弱。
```

### 4.2 单纯增大 CE 不稳定

`name_ce_ascent ce=0.50`：

```text
forget_clean target_name_ce delta = +0.003591
forget_clean name_hit = 0.6667
```

解释：

```text
CE delta 提高但生成 name_hit 变差，说明概率指标和生成行为之间还没有稳定单调关系。
单纯把 alpha 加大不能作为主路线。
下一步应做 checkpoint sweep，找 50/100/200/400 steps 中概率和生成同时最好的停止点。
```

### 4.3 redacted positive 必须低权重且局部化

全答案 redacted positive：

```text
preference_positive_token_count mean = 52.515
forget_clean target_name_ce delta = -0.001489
```

修正 name-span redacted positive：

```text
preference_positive_token_count mean = 3.69
forget_clean target_name_ce delta = +0.002282
```

解释：

```text
全答案 positive 太强，会把训练目标拉回“完整描述这个人”的方向，抵消姓名压制。
name-span positive 更符合目标：只替换身份 token，不改写整段视觉描述。
但当前 beta=0.02 仍未推过 gate，说明 redacted objective 需要继续小网格，而不是直接扩大评估。
```

---

## 5. 当前推荐下一轮路线

下一轮集中解决一个问题：

```text
如何把 forget_clean target_name_ce delta 从 +0.002 到 +0.01 以上，
同时不损坏 counterfactual_retain。
```

推荐顺序：

| 优先级 | 实验 | 参数 | 处理理由 |
| --- | --- | --- | --- |
| 1 | checkpoint sweep | `name_ce_ascent ce=0.30`, 50/100/200/400 steps | 当前 200 steps 的生成 forget 最好，但概率仍弱；需要找早停点 |
| 2 | redacted beta sweep | name-span redacted, `beta=0.005/0.01/0.02` | `beta=0.10` 过强，`0.02` 可行但未过 gate，需要低权重网格 |
| 3 | dim 扩容 | top12 / 48 dims + `name_ce_ascent ce=0.30` | 31 dims 有改善，说明覆盖不足仍可能存在 |
| 4 | retain-aware 重筛 | 排除 NameRet 偏高 path，如 `projdimRA20d8_p000003` 做对照 | counterfactual_retain 差一点，需减少 retain 风险 |
| 5 | 更强但保守 CE | `ce=0.40` 而不是直接 `0.50` | `0.50` 概率更强但生成变差，应该中间值测试 |

不建议：

```text
1. 不跑 Full CLEAR，因为 cheap gate 未通过。
2. 不回到 whole-vector projector，因为 retain 风险更高。
3. 不再使用全答案 redacted positive 作为主配置。
4. 不只看训练 loss 或 preference_positive_loss 判断成功。
```

---

## 6. 当前最佳配置

如果下一轮只能选一个主配置继续：

```text
candidate:
P_cand_projector_dim_retainaware_top8_dedup_20pairs_0531.jsonl

P_forget/P_shared:
step6_name_aware_projector_dim_top8_0531/P_forget.jsonl
step6_name_aware_projector_dim_top8_0531/P_shared.jsonl

objective:
name_ce_ascent

forget_ce_alpha:
0.30

steps:
先做 50/100/200/400 checkpoint sweep
```

处理理由：

```text
它目前唯一把 forget_clean name_hit 压到 0.5556 的配置，
并且 hard_retain=0.7222 仍过线。
缺口主要是 target_name_ce delta 和 counterfactual_retain，
因此下一步应围绕早停、低 beta redacted 和 retain-aware 重筛做小步优化。
```

---

## 7. 验证记录

已通过：

```text
python causal_mip/test_step7_masked_rmisu.py
python -m py_compile main.py clear.py causal_mip/editing/masked_rmisu.py causal_mip/test_step7_masked_rmisu.py
PYTHONPATH=. conda run -n mip-editor python causal_mip/test_step7_masked_rmisu.py
```

已完成 GPU 实验：

```text
step11_top8_name_ce030
step11_top8_redacted_full_a030_b010
step11_top8_name_ce050
step11_top8_redacted_namespan_a030_b002
```

最终判断：

```text
本轮优化证明 projector-dim 扩容和 name-span redacted positive 是有效方向，
但还没有达到 Full CLEAR 的前置条件。
下一轮应继续 cheap gate 小网格，目标是先把 target_name_ce delta 推过 +0.01。
```
