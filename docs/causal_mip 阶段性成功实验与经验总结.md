# causal_mip 阶段性成功实验与经验总结

更新时间：2026-05-31

本文总结 `causal_mip` 路线中已经成立的阶段性成功实验、工程闭环和经验。

需要先明确：

```text
这些结果不是最终 unlearning 成功。
成功点是：causal path 定位、name-token 对齐、projector 编辑、projector-dim 编辑、
Step7 训练链路和 Step8 cheap diagnostic 闭环已经被验证。
```

换句话说，当前阶段已经把问题从“链路是否能跑通”推进到“objective 和编辑粒度是否足够有效”。

---

## 1. 实验范围

当前阶段主要包含三个连续实验：

| 实验 | 文档 | 目标 | 状态 |
| --- | --- | --- | --- |
| Step10 name-token diagnostic 20-pair | `EXPERIMENT_RESULT_STEP10_NAME_TOKEN_DIAG_20PAIRS_0530.md` | 验证 target-name sensitive path 是否存在，并让 Step6 name-aware 分类 | 成功 |
| Step10 Step7 small grid | `EXPERIMENT_RESULT_STEP10_STEP7_SMALL_GRID_0530.md` | 验证 name-token objective 和 projector edit 是否真正进入训练 | 工程成功，行为未通过 |
| Step10 projector-dim name preference | `EXPERIMENT_RESULT_STEP10_PROJECTOR_DIM_NAME_PREF_0530.md` | 验证 projector-dim + retain-aware selection + name preference objective 闭环 | 工程成功，行为未通过 |

当前没有宣称：

```text
forget_clean name_hit 已经低于 baseline；
Full CLEAR 已经通过；
选择性遗忘已经完成。
```

本轮已经证明：

```text
1. 当前候选路径中确实有强 target-name sensitive projector / vision_text path。
2. Step6 name-aware 分类可以把这些 path 从 P_shared 拉回 P_forget。
3. Step7 strict name-token CE scope 已经生效。
4. Step7 projector mask 语义修正后，projector 可以真正进入可编辑模块。
5. Cheap diagnostic 能提前发现当前 Step7 小网格还不足以支撑 Full CLEAR。
6. projector-dim candidate 能被重新 name-score，并进入 dim-level P_forget。
7. Step7 可以只编辑 16 个具体 projector dims，避免 whole-vector projector。
8. name_preference_unlearning 的训练目标已经接入并可记录 token 覆盖。
```

---

## 2. 成功点一：找到了 target-name sensitive projector path

Step10 diagnostic 的核心结果：

| group | records | paths | NameSuf positive | mean NameSuf | p75 NameSuf | max NameSuf | mean name_forget_effect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 320 | 168 | 220 | 1.3626 | 2.6269 | 5.5148 | 1.8475 |
| projector / vision_text | 160 | 8 | 144 | 2.7202 | 3.7616 | 5.5148 | 3.6776 |
| non-projector | 160 | 160 | 76 | 0.0050 | 0.0142 | 0.2043 | 0.0174 |

关键结论：

```text
projector / vision_text path 的 NameSuf 明显高于普通 text / vision path。
这说明 target name token 的因果信号确实集中在 cross-modal projector 路径上。
```

这一步很重要，因为它推翻了一个可能的坏假设：

```text
不是 projector path 没有身份信号；
而是旧 Step6 用 answer-level retain guard 把这些 path 全部归入了 P_shared。
```

---

## 3. 成功点二：Step6 name-aware 分类修复了 Step9 的核心失败点

Step9 corrected 中，projector path 的状态是：

```text
P_forget projector path = 0
P_shared projector path = 8
```

这导致 Step7 实际没有编辑到身份相关的 cross-modal projector path。

Step10 name-aware Step6 后：

### max_forget_projector_paths = 8

| category | count | text | vision | vision_text | contains_projector |
| --- | ---: | ---: | ---: | ---: | ---: |
| P_forget | 22 | 5 | 9 | 8 | 8 |
| P_shared | 16 | 0 | 16 | 0 | 0 |
| P_retain | 16 | 9 | 7 | 0 | 0 |
| P_irrelevant | 114 | 66 | 48 | 0 | 0 |

### max_forget_projector_paths = 4

| category | count | text | vision | vision_text | contains_projector |
| --- | ---: | ---: | ---: | ---: | ---: |
| P_forget | 18 | 5 | 9 | 4 | 4 |
| P_shared | 20 | 0 | 16 | 4 | 4 |
| P_retain | 16 | 9 | 7 | 0 | 0 |
| P_irrelevant | 114 | 66 | 48 | 0 | 0 |

经验：

```text
Step6 不能只看 answer-level Nec / Suf / Ret。
对身份遗忘任务，必须同时看 NameNec / NameSuf / NameRet。
```

更具体地说：

```text
answer-level retain risk 高，不等于该 path 不能编辑。
如果该 path 同时有很强的 target-name sensitivity，需要进入 P_forget 的候选集合，
再由 retain/shared loss 和小剂量 objective 控制副作用。
```

---

## 4. 成功点三：Step7 strict name-token CE scope 跑通

本轮修正了 Step7 的 name CE 行为：

```text
forget_ce_scope = name 时，只使用 name_token_positions。
如果 name_token_positions 为空，则 forget_ce_token_count = 0。
不再 fallback 到 answer_token_positions。
```

Step7 小网格中，三个 run 的 CE token 覆盖都正常：

| run | loss records | forget_ce_token_count mean | zero CE steps | 结论 |
| --- | ---: | ---: | ---: | --- |
| A maxproj4 ce0.05 | 200 | 9.87 | 0 | name-token CE 生效 |
| B maxproj4 ce0.10 | 200 | 9.87 | 0 | name-token CE 生效 |
| C maxproj8 ce0.10 | 200 | 9.87 | 0 | name-token CE 生效 |

经验：

```text
Step7 必须显式记录 forget_ce_token_count。
只要 name CE token count 为 0，就不能把该 run 当作 name-token objective 实验。
```

这个检查比看总 loss 更可靠，因为总 loss 可能仍然下降，但它不一定在优化目标姓名 token。

---

## 5. 成功点四：projector whole-vector mask bug 被定位并修正

小网格 A/B 暴露出一个关键问题：

```text
P_forget 和 P_shared 都包含 mm_projector placeholder neuron=0。
Step7 使用 editable = forget - shared。
因此 projector 被抵消，实际没有进入训练。
```

修正后，非 projector_dim_level 的 projector placeholder 不再被当作单个 neuron=0，而是按 whole-vector 语义展开：

```text
WHOLE_VECTOR_NEURON = -1
mm_projector placeholder -> visual.merger.mlp.0 whole-vector output
```

C run 的 Step7 summary 证明 projector 已经真实进入编辑集合：

| run | mask modules | editable neurons | projector modules | projector editable neurons |
| --- | ---: | ---: | ---: | ---: |
| A maxproj4 ce0.05 | 68 | 275 | 0 | 0 |
| B maxproj4 ce0.10 | 68 | 275 | 0 | 0 |
| C maxproj8 ce0.10 | 69 | 5401 | 1 | 5120 |

C 的 projector 信息：

```text
module = mm_projector
edit_module = visual.merger.mlp.0
trace_module = visual.merger.mlp.0
trace_kind = output
uses_whole_vector_neuron = true
num_forget_editable_neurons = 5120
```

经验：

```text
projector path 不能只看 Step6 中是否进入 P_forget。
必须看 Step7 summary 中 projector_modules 和 projector_editable_neurons。
```

否则会出现路径分类看似正确，但训练实际没有编辑 projector 的假阳性。

---

## 6. 成功点五：cheap diagnostic 阻止了错误扩大实验

Step7 小网格没有达到行为成功标准：

| run | forget_clean target_name_ce delta | counterfactual_retain name CE delta | hard_retain name CE delta | 判断 |
| --- | ---: | ---: | ---: | --- |
| Step9 corrected | -0.000596 | -0.001988 | -0.003443 | forget 方向错误 |
| A maxproj4 ce0.05 | -0.003828 | +0.000551 | -0.000112 | forget 方向错误 |
| B maxproj4 ce0.10 | +0.000429 | -0.002383 | -0.000270 | forget 方向微弱正确 |
| C maxproj8 ce0.10 | +0.000421 | +0.002512 | +0.001998 | forget 微弱正确，但 retain 风险上升 |

generation eval 也没有通过：

| run | forget_clean name_hit | counterfactual_retain name_hit | hard_retain name_hit |
| --- | ---: | ---: | ---: |
| PEFT baseline | 0.6111 | 0.7222 | 0.6944 |
| Step9 corrected | 0.7222 | 0.6667 | 0.7222 |
| A maxproj4 ce0.05 | 0.6667 | 0.6667 | 0.7222 |
| B maxproj4 ce0.10 | 0.7222 | 0.7778 | 0.6944 |
| C maxproj8 ce0.10 | 0.6667 | 0.7222 | 0.7222 |

这里的成功经验是：

```text
cheap probability diagnostic + pair generation eval 能快速否决不合格 run。
不需要直接进入 Full CLEAR 才发现失败。
```

当前 gate 应继续保留：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= baseline 0.6111
counterfactual_retain name_hit drop <= 0.05
hard_retain name_hit drop <= 0.05
```

---

## 7. 成功点六：projector-dim + name preference 工程闭环跑通

根据 `STEP10_NEXT_OPTIMIZATION_FROM_UNLEARNING_SURVEY_0530.md`，本轮继续做了一个最小可执行优化：

```text
projector-dim candidate
+ name-aware Step6
+ name_editable_score
+ name_preference_unlearning
+ Step8 cheap diagnostic
```

对应实验文档：

```text
docs/EXPERIMENT_RESULT_STEP10_PROJECTOR_DIM_NAME_PREF_0530.md
```

### 7.1 代码成功点

新增或优化：

| 模块 | 成功点 |
| --- | --- |
| `causal_mip/causal_scores/classify_paths.py` | 新增 `name_editable_score = name_forget_effect / (1e-6 + name_retain_impact)` |
| `causal_mip/causal_scores/classify_paths.py` | 新增 `--projector_topk_metric name_editable_score`，projector top-k 可按 retain-aware 分数排序 |
| `causal_mip/editing/masked_rmisu.py` | 新增 `name_preference_unlearning` objective |
| `main.py` / `ours.py` | 接入 `--masked_rmisu_preference_positive_alpha` |
| `causal_mip/test_step6_classify_paths.py` | 覆盖 projector top-k 使用 `name_editable_score` |
| `causal_mip/test_step7_masked_rmisu.py` | 覆盖 answer-without-name positive mask 和 preference objective smoke test |

测试通过：

```text
python causal_mip/test_step6_classify_paths.py
python causal_mip/test_step7_masked_rmisu.py
python causal_mip/test_step5_causal_scores.py
python causal_mip/test_name_token_metrics.py
python causal_mip/test_step8_probability_diagnostic.py
```

### 7.2 projector-dim name score 成功点

使用 projector-dim candidates：

```text
P_cand_projector_dim_retainaware_top4_dedup_20pairs_0529.jsonl
P_cand_projector_dim_retainaware_top4_dedup_20pairs_bound_0529.jsonl
```

重要经验：

```text
binding 文件包含 9 个分散 pair。
不能用 --num_pairs 20 截断，否则只覆盖 1 条 record。
必须用 --pair_ids 精确跑绑定 pair。
```

9 条 projector-dim path 的 name score 结果：

| metric | n | mean | min | max | positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| NameNec | 8 | -0.016174 | -0.071665 | 0.003452 | 1 |
| NameSuf | 8 | 0.019690 | -0.020774 | 0.080461 | 4 |
| NameRet | 9 | -0.014879 | -0.093367 | 0.001933 | 2 |
| name_forget_effect | 9 | 0.021907 | 0.000000 | 0.080461 | 4 |
| name_retain_impact | 9 | 0.000318 | 0.000000 | 0.001933 | 2 |

成功点：

```text
projector-dim candidates 中确实存在 NameSuf > 0 的 target-name sensitive dims。
这说明从 whole-vector projector 转向 dim-level projector 是可执行的，不是空路线。
```

### 7.3 Step6 projector-dim 分类成功点

Step6 参数：

```text
--name_aware_forget
--name_forget_quantile 0.5
--name_retain_quantile 0.75
--max_forget_projector_paths 4
--projector_name_effect_ratio_threshold 0.0
--projector_topk_metric name_editable_score
```

分类结果：

| category | count | contains_projector |
| --- | ---: | ---: |
| P_forget | 4 | 4 |
| P_shared | 0 | 0 |
| P_retain | 2 | 2 |
| P_irrelevant | 3 | 3 |

P_forget projector dims：

| path_id | dims |
| --- | --- |
| projdimRA20d_p000003 | 387, 318, 886, 984 |
| projdimRA20d_p000004 | 872, 612, 1197, 1264 |
| projdimRA20d_p000007 | 1220, 410, 420, 983 |
| projdimRA20d_p000008 | 769, 231, 229, 993 |

成功点：

```text
Step6 已经能输出具体 projector dims。
P_forget 不再依赖 coarse mm_projector placeholder。
这直接修复了 A/B 中 projector 被 P_shared 抵消的问题。
```

### 7.4 Step7 projector-dim 编辑成功点

run：

```text
step10_projector_dim_name_pref_200steps_0530
```

Step7 mask summary：

| item | value |
| --- | ---: |
| loss records | 200 |
| mask modules | 1 |
| editable neurons | 16 |
| projector modules | 1 |
| whole-vector projector | false |
| edit module | `visual.merger.mlp.0` |

实际编辑 dims：

```text
229, 231, 318, 387, 410, 420, 612, 769,
872, 886, 983, 984, 993, 1197, 1220, 1264
```

训练 loss 覆盖：

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| forget_ce_loss | 0.229484 | 0.015503 | 1.148438 |
| preference_positive_loss | 1.023467 | 0.330078 | 2.609375 |
| forget_ce_token_count | 9.87 | 3 | 17 |
| preference_positive_token_count | 50.945 | 26 | 92 |
| retain_loss | 0.000009 | 0.000004 | 0.000018 |

成功点：

```text
Step7 已经能只编辑 projector 的 16 个具体 dims。
没有 whole-vector projector。
name_preference_unlearning 的 target name CE 和 answer_without_name positive CE 都有非零 token 覆盖。
```

### 7.5 本轮 cheap diagnostic 的成功作用

虽然工程闭环成功，但行为没有通过：

| gate | result | pass |
| --- | --- | --- |
| forget_clean target_name_ce delta > 0.01 | +0.000038 | false |
| forget_clean name_hit <= 0.6111 | 0.7778 | false |
| hard_retain name_hit >= 0.6444 | 0.6944 | true |
| counterfactual_retain name_hit >= 0.6722 | 0.6667 | false |

这次 cheap diagnostic 的价值是：

```text
它证明 projector-dim + name_preference 工程链路能跑通；
也证明当前 objective 强度和正样本构造不足；
因此不应进入 Full CLEAR。
```

阶段性成功定义：

```text
成功的是工程闭环和问题定位；
未成功的是最终 target-name forgetting 行为。
```

---

## 8. 关键经验沉淀

### 8.1 路径选择必须对准 target name token

旧路线的问题：

```text
answer-level path score 可以恢复整体答案 sufficiency，
但不保证恢复或压制 target name token。
```

新路线的经验：

```text
身份遗忘任务必须单独计算 NameNec / NameSuf / NameRet。
target_name_ce 和 name_hit 应该优先于整体 answer 指标。
```

优先级建议：

```text
target_name_ce > pair name_hit > Full CLEAR classification > generation text overlap
```

### 8.2 projector 是关键，但不能 whole-vector 粗编辑

C run 证明：

```text
whole-vector projector 编辑能产生 forget_clean target_name_ce 的微弱正向变化。
```

但它也证明：

```text
whole-vector projector retain 风险高。
counterfactual_retain 和 hard_retain 的 name CE 同时上升。
```

经验：

```text
下一轮应使用 projector-dim candidate，而不是 whole-vector projector。
```

最新 projector-dim run 进一步证明：

```text
dim-level projector 编辑可执行；
但只编辑 16 个 dims 时 forget signal 仍然太弱。
```

### 8.3 P_shared 不能粗粒度吞掉 projector

A/B 的问题不是训练参数弱，而是：

```text
projector placeholder 同时在 P_forget 和 P_shared，
最终 editable = forget - shared 后 projector 被抵消。
```

经验：

```text
P_forget / P_shared 的 projector guard 必须是 dim-level 或 subspace-level。
不能用一个 coarse mm_projector placeholder 同时代表所有 projector 维度。
```

### 8.4 name_preference_unlearning 需要真正的 redacted target

当前实现是最小版本：

```text
压低 name token；
保留 answer_without_name token。
```

它还不是完整 preference learning，因为没有显式构造：

```text
y_pos = redacted / unknown-person / generic identity answer
y_neg = original answer with target name
```

经验：

```text
下一步不能只依赖 answer_without_name preserve。
需要真正构造 redacted-name positive target，
否则 positive preserve 可能主要保持原 answer 结构，稀释 target name suppression。
```

### 8.5 不要用训练 loss 替代行为诊断

Step7 loss 和 CE token count 只能证明训练目标接入了。

是否真的遗忘，必须看：

```text
forget_clean target_name_ce 是否明显上升；
forget_clean name_hit 是否下降到 baseline 以下；
retain name_hit 是否没有明显下降。
```

本轮 B/C 的 forget_clean target_name_ce 只上升约 `+0.00042`，不足以支撑有效遗忘。
projector-dim name-pref 的 forget_clean target_name_ce 只上升 `+0.000038`，更不能支撑有效遗忘。

---

## 9. 可复用规则

后续实验建议固定以下检查：

| 阶段 | 必查项 | 失败时动作 |
| --- | --- | --- |
| Step5 | NameSuf positive 是否存在，projector / vision_text 是否显著高于 non-projector | 不进入 Step6 |
| Step6 | P_forget 是否包含 target-name sensitive projector / dim-level path | 重新调 name-aware 分类 |
| Step6 projector-dim | 是否使用 `name_editable_score` 或等价 retain-aware 指标 | 避免只按 forget effect 选高 retain-risk dims |
| Step7 preflight | projector_modules、projector_editable_neurons、forget_ce_token_count | 任一为 0 则不算有效训练 |
| Step7 projector-dim | `whole-vector projector = false` 且 editable dims > 0 | 否则回到 candidate / mask |
| Step7 preference | positive token count 和 name CE token count 都非零 | 否则 objective 没有接入 |
| Step8 probability | forget_clean target_name_ce delta | 不明显大于 0 则不跑 Full CLEAR |
| Step8 generation | forget_clean name_hit、retain name_hit drop | 未过 gate 则回到 candidate/objective |

推荐的最小成功标准：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

---

## 10. 下一步路线

当前结果支持继续 `causal_mip` 的 Step10 优化，但不支持直接扩大 Full CLEAR。

下一步应做：

```text
projector-dim candidate
+ NameNec / NameSuf / NameRet
+ name-aware Step6
+ dim-level P_forget / P_shared guard
+ 真正的 redacted-name preference objective
```

建议下一轮小网格：

| run | candidate | objective | 参数 | 目标 |
| --- | --- | --- | --- | --- |
| I | projector-dim | `name_preference_unlearning` | ce=0.30, pos=0.05 | 减少 positive preserve 抵消 |
| J | projector-dim | `name_ce_ascent` | ce=0.30 | 测试 dim-level 纯姓名压制强度 |
| K | projector-dim | full `redacted_name_preference` | beta TBD | 构造真正 y_pos/y_neg preference |
| L | projector-dim + more dims | `name_ce_ascent` or preference | top 8 paths / 32 dims | 测试是否 16 dims 太少 |

进入 Full CLEAR 前仍然必须先过 cheap gate。

---

## 11. 一句话总结

```text
causal_mip 当前阶段性成功不是“已经遗忘成功”，而是“已经把错误定位到可操作的层面”：
target name signal 在 projector / vision_text path 中存在；
name-aware Step6 能把它们送入 P_forget；
Step7 name-token objective、whole-vector projector edit、projector-dim edit 和 name_preference objective 都已跑通；
但最终行为还未通过，下一步必须做真正的 redacted-name preference 或更强的 dim-level name suppression。
```
