# Step10 Name-Token Aligned Route

更新时间：2026-05-30

本文基于 `step9_bound_projector_current_corrected_0529_193413` 的失败分析，整理下一步优化路线：

```text
把路径选择和训练目标都对准 target name token。
```

核心判断：

```text
Step9 current corrected 不是训练没跑通，而是科学目标没有对齐。
Step5/6 已恢复 sufficiency 信号，Step7 也完成了更强编辑；
但 target-name CE 没有上升，P_forget 也没有包含 projector / vision_text path。

因此下一步不能继续盲目增强 Step7。
必须先让 Step5/6 选择 name-token sensitive path，
再用 Step7 的 name-token scoped objective 编辑这些路径。
```

---

## 1. 当前证据

来自实验：

```text
step9_bound_projector_current_corrected_0529_193413
```

关键结果：

| metric | baseline | current | delta | 结论 |
| --- | ---: | ---: | ---: | --- |
| pair forget_clean name_hit | 0.6111 | 0.7222 | +0.1111 | 失败，目标是降低 |
| pair counterfactual_retain name_hit | 0.7222 | 0.6667 | -0.0556 | 失败，超过 0.05 drop |
| Full CLEAR forget classification acc | 0.6862 | 0.6915 | +0.0053 | 失败，目标是降低 |
| Full CLEAR forget generation acc | 0.4628 | 0.4415 | -0.0213 | 通过但较弱 |
| forget_clean target_name_ce | 0.183437 | 0.182840 | -0.000596 | 失败，姓名概率没有被压低 |

Step6 分类结果：

| category | count |
| --- | ---: |
| P_forget | 154 |
| P_shared | 120 |
| P_retain | 211 |
| P_irrelevant | 883 |

projector 分布：

| category | contains_projector=true | contains_projector=false |
| --- | ---: | ---: |
| P_forget | 0 | 154 |
| P_shared | 8 | 112 |
| P_retain | 0 | 211 |
| P_irrelevant | 0 | 883 |

直接解释：

```text
P_forget 没有 projector / vision_text path。
所有 projector path 都被归入 P_shared。
Step7 实际主要编辑 text/vision down_proj，不是真正 projector identity path。
```

---

## 2. 总体实现目标

下一轮目标不是简单提高遗忘强度，而是让每个阶段都对齐 target name token。

需要新增或扩展的核心信号：

```text
NameNec
NameSuf
NameRet
name_forget_effect
name_retain_impact
```

其中：

```text
NameNec = clean_target_name_logprob - ablated_target_name_logprob
NameSuf = restored_target_name_logprob - corrupt_target_name_logprob
NameRet = retain_clean_name_logprob - retain_ablated_name_logprob

name_forget_effect = max(0, NameNec) + alpha_name * max(0, NameSuf)
name_retain_impact = max(0, NameRet)
```

直觉：

```text
如果 ablate 某条 path 会降低目标姓名概率，说明这条 path 对目标姓名必要。
如果从 corrupt 激活恢复这条 path 会恢复目标姓名概率，说明这条 path 对目标姓名充分。
如果 ablate 这条 path 会伤害 retain 姓名能力，说明这条 path 需要 guard。
```

---

## 3. 实现阶段

## 3.1 Phase 0: 只做 name-token diagnostic，不训练

目标：

```text
先确认当前候选 path 中是否存在 target-name sensitive path。
不要直接进入 Step7。
```

建议新增脚本：

```text
causal_mip/causal_scores/build_name_token_scores.py
```

或者在现有 Step5 脚本中增加参数：

```bash
--score_scope answer,name
```

输入：

```text
causal_pairs_train.jsonl
P_cand_bound_train_step9_bound_projector_0526_214028.jsonl
```

输出：

```text
mip_workspace/outputs/scores/path_name_scores_<RUN_ID>.jsonl
```

每条记录至少包含：

```json
{
  "pair_id": "...",
  "path_id": "...",
  "path_modality": "text|vision|vision_text",
  "contains_projector": true,
  "NameNec": 0.0,
  "NameSuf": 0.0,
  "NameRet": 0.0,
  "name_forget_effect": 0.0,
  "name_retain_impact": 0.0,
  "target_name": "...",
  "target_name_token_positions": [93, 94],
  "name_match_status": "matched|name_not_in_target_text",
  "status": "ok"
}
```

验收：

```text
至少要看到一批 path 的 NameSuf > 0。
至少要看到 vision_text/projector path 中存在 name_forget_effect 高的记录。
如果 projector path 的 NameSuf 仍然很低，需要先修 projector path 定位或 corrupt 对照构造。
```

---

## 3.2 Phase 1: Step5 增加 name-token scoring

现有入口：

```text
causal_mip/causal_scores/build_scores.py
causal_mip/causal_scores/metrics.py
```

可复用逻辑：

```text
causal_mip/evaluation/step8_probability_diagnostic.py
```

其中已有：

```text
target name token 定位
name token CE / logprob 统计
target_vs_generated_name_logprob_margin
```

建议实现方式：

1. 抽取通用工具函数：

```text
find target name token positions
compute scoped logprob / CE
score target text by answer tokens
score target text by name tokens
```

建议放到：

```text
causal_mip/causal_scores/name_token_metrics.py
```

2. 在 `compute_path_causal_score_record()` 中新增 name scope：

```text
answer_scope: 原 Nec/Suf/Ret
name_scope: 新 NameNec/NameSuf/NameRet
```

3. 输出保持向后兼容：

```text
Nec / Suf / Ret 继续保留
NameNec / NameSuf / NameRet 新增
```

注意：

```text
不要替换原 answer-level score。
下一步需要 answer score 和 name score 同时存在，方便比较路径是否只是影响描述风格，还是影响身份姓名。
```

---

## 3.3 Phase 2: Step6 name-aware classification

现有入口：

```text
causal_mip/causal_scores/classify_paths.py
```

当前逻辑：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

建议新增参数：

```bash
--name_aware_forget
--alpha_name 1.0
--name_forget_quantile 0.75
--name_retain_quantile 0.75
--min_name_sufficiency 0.0
--require_positive_name_sufficiency
--max_forget_projector_paths 8
--projector_name_effect_ratio_threshold 1.2
```

新增聚合字段：

```text
NameNec
NameSuf
NameRet
name_forget_effect
name_retain_impact
```

建议分类规则：

```text
P_forget requires:
  NameSuf > min_name_sufficiency
  name_forget_effect >= name_forget_threshold
  all_score_records_fully_patchable = true

P_shared if:
  name_retain_impact >= name_retain_threshold
  AND name_forget_effect is not clearly higher than retain impact

P_retain if:
  name_retain_impact >= name_retain_threshold
  AND name_forget_effect < name_forget_threshold

P_irrelevant otherwise
```

projector 特殊处理：

```text
如果 contains_projector=true 且 name_forget_effect 很高：
  不要默认放进 P_shared。
  可允许进入 P_forget，但限制数量和比例。
```

最小策略：

```text
允许 top-k projector / vision_text name-sensitive paths 进入 P_forget。
例如 max_forget_projector_paths = 8。
```

原因：

```text
当前 Step9 的最大问题是所有 projector path 被 retain guard 保护。
如果不允许一部分 projector path 进入 P_forget，bound projector 路线无法真正验证。
```

---

## 3.4 Phase 3: Step7 使用 name-token scoped objective

现有入口：

```text
causal_mip/editing/masked_rmisu.py
```

已有能力：

```text
name_ce_ascent
activation_random_name_ce
forget_ce_scope = name
name_token_positions mask
```

建议首轮不要用纯 `name_ce_ascent`，而是：

```text
forget_objective = activation_random_name_ce
target_ce_scope = name
forget_ce_alpha = 0.05 / 0.1 / 0.2
```

原因：

```text
纯 name_ce_ascent 风险较高，可能破坏全局姓名能力。
activation_random_name_ce 保留 activation_random 的路径扰动，同时只在 name token 上做 CE ascent。
```

Step7 保留：

```text
retain objective unchanged
shared objective unchanged
masked parameter editing unchanged
```

新增检查：

```text
forget_ce_scope 必须等于 name
forget_ce_token_count 必须 > 0
P_forget 中必须存在 NameSuf > 0 的 path
如果配置为 projector route，P_forget 中必须存在 contains_projector=true 的 path
```

---

## 3.5 Phase 4: 先 cheap eval，再 Full CLEAR

不要每次都直接跑完整 Full CLEAR。

建议顺序：

```text
1. Step8 pair eval
2. Step8 probability diagnostic
3. Step8 protocol quick check
4. 满足 name CE 条件后再跑 Full CLEAR remote
```

cheap eval 的硬门槛：

```text
forget_clean target_name_ce delta > 0
forget_clean target_name_mean_logprob delta < 0
forget_clean name_hit_rate <= baseline
hard_retain name_hit drop <= 0.05
counterfactual_retain name_hit drop <= 0.05
```

只有满足这些条件，才进入 Full CLEAR。

---

## 4. 推荐 run 设计

建议新 run 命名：

```text
step10_name_token_aligned_<timestamp>
```

第一轮只做一个小网格：

| run | Step6 mode | Step7 objective | forget_ce_alpha | projector policy |
| --- | --- | --- | ---: | --- |
| A | name-aware | activation_random_name_ce | 0.05 | max 4 projector paths |
| B | name-aware | activation_random_name_ce | 0.10 | max 4 projector paths |
| C | name-aware | activation_random_name_ce | 0.10 | max 8 projector paths |
| D | name-aware | activation_random_name_ce | 0.20 | max 8 projector paths |

优先级：

```text
先跑 A/B。
只有 target_name_ce 有明确上升，再跑 C/D。
```

不建议第一轮直接做：

```text
large forget_ce_alpha
pure name_ce_ascent
无 projector 限制的 P_forget
取消 retain/shared guard
```

---

## 5. 需要新增或修改的文件

建议新增：

```text
causal_mip/causal_scores/name_token_metrics.py
causal_mip/causal_scores/build_name_token_scores.py
causal_mip/test_name_token_metrics.py
causal_mip/test_step6_name_aware_classify_paths.py
```

建议修改：

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/build_scores.py
causal_mip/causal_scores/classify_paths.py
causal_mip/editing/masked_rmisu.py
```

其中 Step7 修改应尽量少，因为已有 name CE 支持。重点是增加检查和 run-time summary：

```text
forget_config.target_ce_scope
forget_config.forget_objective
losses[*].forget_ce_scope
losses[*].forget_ce_token_count
mask_summary contains projector counts
```

---

## 6. 最小实现顺序

推荐按下面顺序推进：

1. 抽取 name token 定位和 scoped CE/logprob 函数。
2. 写 name-token metric 单元测试。
3. 在 Step5 score record 中增加 NameNec / NameSuf / NameRet。
4. 写一个只统计、不训练的 name score diagnostic。
5. 在 Step6 增加 `--name_aware_forget` 分类模式。
6. 检查新 Step6 输出中 P_forget 的模态和 projector 分布。
7. 用现有 Step7 的 `activation_random_name_ce` 跑小网格。
8. 先跑 Step8 pair + probability diagnostic。
9. 满足 gate 后再跑 Full CLEAR。

---

## 7. 验收标准

路径选择验收：

```text
P_forget 中 NameSuf > 0 的 path 占多数。
P_forget 中至少有一部分 vision_text/projector path。
P_shared 不应吞掉所有 projector path。
```

训练验收：

```text
forget_ce_scope = name
forget_ce_token_count > 0
loss 正常下降或稳定
retain/shared loss 不爆炸
```

Step8 验收：

```text
forget_clean target_name_ce delta > 0
forget_clean name_hit_rate < 0.6111
hard_retain name_hit_rate >= 0.6444
counterfactual_retain name_hit_rate >= 0.6722
```

Full CLEAR 验收：

```text
forget classification remote acc <= baseline 0.6862
forget generation remote acc <= baseline 0.4628
retain classification remote acc >= baseline - 0.05
retain generation remote acc >= baseline - 0.05
```

强成功标准：

```text
forget_clean target_name_ce 明显上升；
forget_clean name_hit_rate 低于 baseline；
Full CLEAR forget classification 和 generation 同时下降；
retain 两项不显著下降。
```

---

## 8. 风险与注意事项

## 8.1 name token 定位失败

如果目标 answer 中没有原始 name，name token positions 可能为空。

处理策略：

```text
name_match_status = name_not_in_target_text
该样本不参与 NameNec / NameSuf 阈值统计
Step7 name CE token count 为 0 时直接 fail-fast
```

## 8.2 projector path retain impact 过高

projector path 可能同时承载 forget identity 和 retain identity。

处理策略：

```text
不要简单放弃所有 high-retain projector path。
允许 top-k name-sensitive projector path 进入 P_forget，
同时通过 retain/shared loss 控制损伤。
```

## 8.3 CE ascent 过强

name CE ascent 可能导致模型整体姓名识别能力下降。

处理策略：

```text
小 alpha 起步；
先看 counterfactual_retain 和 hard_retain；
不要第一轮取消 retain/shared guard。
```

## 8.4 generation 指标误导

forget generation acc 下降不等于姓名遗忘成功。

本路线中优先级应为：

```text
target_name_ce > pair name_hit > Full CLEAR classification > generation
```

---

## 9. 预期结果解释

如果下一轮出现：

```text
target_name_ce 上升
forget_clean name_hit 下降
retain 稳定
```

说明 name-token aligned route 有效。

如果出现：

```text
target_name_ce 不变
forget generation 下降
```

说明仍然只是在扰动描述/生成风格，没有真正遗忘身份姓名。

如果出现：

```text
target_name_ce 上升
retain name_hit 大幅下降
```

说明 name objective 有效但选择性不足，需要加强 retain guard 或减少 projector top-k / CE alpha。

如果出现：

```text
P_forget 仍无 projector / vision_text path
```

说明 Step6 仍没有解决 bound projector 路线的核心问题，需要回到 path localization / retain guard。

---

## 10. 一句话总结

```text
Step10 的目标不是更强编辑，而是更准编辑：
用 name-token causal score 选路径，
用 name-token CE ascent 训练，
用 target_name_ce 和 pair name_hit 先验收，
通过后再跑 Full CLEAR。
```
