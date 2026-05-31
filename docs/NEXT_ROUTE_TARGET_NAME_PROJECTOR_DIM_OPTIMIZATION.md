# 下一轮 target-name projector-dim 优化路线

更新时间：2026-05-31

本文档用于承接当前 `causal_mip` 阶段性结果，给出下一轮集中解决方案。

当前判断：

```text
target-name signal 已经找到；
projector-dim 编辑链路已经跑通；
但 objective 和编辑强度还没有把最终行为推过 cheap gate。
```

因此下一轮不应直接扩大 Full CLEAR，而应围绕两个可验证假设做小网格：

```text
1. objective 不够对准：当前压低 name token，但没有明确训练 redacted / unknown-person 替代输出。
2. 编辑强度不够：当前只编辑 16 个 projector dims，可能没有覆盖足够强的身份子空间。
```

---

## 1. 总体路线

下一轮主线：

```text
Step A: 固定数据、baseline 和 cheap gate
Step B: 继续 projector-dim，扩大 retain-aware candidate
Step C: 先测试 objective 强度
Step D: 再测试 dim 数量扩容
Step E: 实现真正的 redacted-name preference
Step F: cheap gate 通过后才进入 Full CLEAR
```

核心原则：

```text
不要回到 whole-vector projector；
不要只用 answer-level Nec / Suf / Ret；
不要用训练 loss 代替行为诊断；
cheap gate 未通过不跑 Full CLEAR。
```

---

## 2. 固定评估协议

所有下一轮实验必须使用同一批 pair、同一套 baseline、同一套 Step8 cheap diagnostic。

推荐继续使用当前 gate：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= baseline 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

处理理由：

```text
当前效果仍很弱。
如果每轮样本、baseline 或 gate 变动，就无法判断改进来自方法本身还是评估波动。
cheap gate 是进入 Full CLEAR 前的低成本过滤器，应继续作为硬前置条件。
```

优先观察指标：

```text
1. forget_clean target_name_ce delta
2. forget_clean name_hit
3. counterfactual_retain name_hit
4. hard_retain name_hit
```

其中优先级为：

```text
target_name_ce > pair name_hit > Full CLEAR classification > generation text overlap
```

---

## 3. projector-dim candidate 策略

下一轮仍然使用 projector-dim candidate，不建议回到 whole-vector projector。

候选池准备三档：

| candidate 档位 | 规模 | 用途 |
| --- | ---: | --- |
| small | top 4 paths / 16 dims | 复现当前设置，作为对照 |
| medium | top 8 paths / 32 dims | 主力实验 |
| large | top 12 paths / 48 dims | 强编辑上限测试 |

排序指标继续使用：

```text
name_editable_score = name_forget_effect / (1e-6 + name_retain_impact)
```

处理理由：

```text
projector / vision_text path 已经被证明包含明显 target-name signal。
但是 whole-vector projector 会带来 retain 风险。
dim-level projector 可以扩大编辑强度，同时用 retain-aware 排序控制副作用。
```

Step6 后必须检查：

```text
P_forget contains_projector > 0
P_forget projector dim 数量 > 0
P_shared 没有粗粒度吞掉同一 projector
NameSuf positive 存在
NameRet 不高
```

如果检查失败：

```text
不要进入 Step7；
回到 Step5/Step6 调整 name-aware 阈值、projector_topk_metric 或 candidate 生成。
```

---

## 4. Step7 preflight 检查

每个 Step7 run 开始前或训练 summary 生成后，必须检查：

```text
projector_modules > 0
projector_editable_neurons > 0
whole-vector projector = false
forget_ce_token_count > 0
preference_positive_token_count > 0  # 仅 preference objective 必查
```

处理理由：

```text
历史 A/B run 的核心问题是 projector placeholder 同时进入 P_forget 和 P_shared，
导致 editable = forget - shared 后 projector 被抵消。
因此不能只看 Step6 分类结果，必须确认 Step7 mask 里 projector dims 真的可编辑。
```

如果任一关键项为 0：

```text
该 run 不应计入有效实验；
优先检查 P_forget/P_shared、projector edit module 映射、name_token_positions 和 pair binding。
```

---

## 5. Objective 强度小网格

第一阶段先固定 16 dims，测试 objective 是否足够打动模型。

推荐实验：

| run | dims | objective | 参数 | 目的 |
| --- | ---: | --- | --- | --- |
| C1 | 16 | `name_ce_ascent` | ce=0.30 | 测试纯姓名压制是否有效 |
| C2 | 16 | `name_ce_ascent` | ce=0.50 | 测试更强压制上限 |
| C3 | 16 | `name_preference_unlearning` | ce=0.30, pos=0.05 | 减少 positive preserve 抵消 |

处理理由：

```text
当前 projector-dim name-pref run 的 forget_clean target_name_ce delta 只有 +0.000038。
这说明当前 objective 对行为影响过弱。
先在 16 dims 上测试 objective，可以避免一开始就把 candidate 规模和 objective 同时改动，
从而无法判断失败来源。
```

判断方式：

| 结果 | 解释 | 下一步 |
| --- | --- | --- |
| `target_name_ce delta` 明显上升 | objective 有效，当前可能是 dim 覆盖不足 | 进入 dim 扩容 |
| `target_name_ce delta` 仍低于 0.002 | objective 或 mask 仍不足 | 优先尝试 redacted preference 或检查 mask |
| CE 上升但 name_hit 不降 | 模型知道姓名变差，但生成仍回到原名 | 加 redacted positive target |
| name_hit 降但 retain 也降 | 压制太粗或 retain guard 不够 | 收紧 NameRet，提高 retain preserve |

---

## 6. Dim 数量扩容小网格

第二阶段选择 Step5 中表现最好的 objective，测试 16 / 32 / 48 dims。

推荐实验：

| run | dims | objective | 目的 |
| --- | ---: | --- | --- |
| D1 | 16 | 最佳 objective | 对照 |
| D2 | 32 | 最佳 objective | 主力配置 |
| D3 | 48 | 最佳 objective | 强编辑上限 |

处理理由：

```text
如果 16 dims 不足以覆盖身份子空间，32 或 48 dims 应该提高 forget_clean target_name_ce delta。
如果 CE delta 上升但 retain 明显下降，说明编辑强度到达上限，需要收紧 retain-aware selection。
如果 CE delta 仍不上升，说明当前 objective 或 target construction 仍不对。
```

决策规则：

| 观察 | 决策 |
| --- | --- |
| 32 dims 明显优于 16 dims，retain 仍过 gate | 以 32 dims 作为下一轮主配置 |
| 48 dims forget 更强但 retain 失败 | 退回 32 dims，或提高 retain loss |
| 16/32/48 都无明显 forget | 优先推进 redacted-name preference |
| CE delta 上升但 name_hit 不降 | 保留该 dims 配置，改 objective 为 redacted preference |

---

## 7. 实现 redacted-name preference

当前 `name_preference_unlearning` 只是最小版本：

```text
负向：压低 target name token
正向：保留 answer_without_name token
```

下一步应实现真正的：

```text
redacted_name_preference
```

目标构造：

```text
y_neg = 原答案，包含 target name
y_pos = redacted / unknown-person / generic identity answer
```

可选正样本模板：

```text
The person is unknown.
I cannot identify this person.
The identity is redacted.
This appears to be an unidentified person.
```

推荐 loss：

```text
L = alpha * target_name_ce_ascent
  + beta  * positive_redacted_ce
  + gamma * retain_preserve_loss
```

推荐初始参数：

```text
alpha = 0.30
beta = 0.10
gamma = 沿用当前 retain loss 设置
```

处理理由：

```text
只压低 target name token 会让模型缺少替代输出方向。
answer_without_name positive 可能仍然保留原答案结构，稀释 target name suppression。
redacted positive target 可以明确训练模型从“输出目标姓名”转向“输出 unknown / redacted identity”。
```

实现后新增检查：

```text
redacted_positive_token_count > 0
redacted_positive_loss 有记录
target_name_ce_ascent loss 有记录
retain_loss 有记录
```

---

## 8. Checkpoint 诊断

每个 run 建议保存并诊断多个 checkpoint：

```text
50 steps
100 steps
200 steps
400 steps
```

每个 checkpoint 都跑：

```text
Step8 probability diagnostic
pair generation name_hit eval
retain name_hit eval
```

处理理由：

```text
unlearning 可能出现早期 forget 有效、后期 retain 损伤变大的情况。
也可能出现后期被 positive preserve 拉回，导致 target name suppression 变弱。
多 checkpoint 诊断可以找到最佳停止点，而不是只看最后一个 checkpoint。
```

---

## 9. 推荐最小实验矩阵

下一轮优先跑 6 个 run：

| run | dims | objective | 参数 | 目的 |
| --- | ---: | --- | --- | --- |
| R1 | 16 | `name_ce_ascent` | ce=0.30 | 纯姓名压制基础强度 |
| R2 | 16 | `name_ce_ascent` | ce=0.50 | 纯姓名压制上限 |
| R3 | 16 | `name_preference_unlearning` | ce=0.30, pos=0.05 | 减少 positive preserve 抵消 |
| R4 | 32 | `name_ce_ascent` | ce=0.30 | 测试 dim 覆盖是否不足 |
| R5 | 32 | `name_preference_unlearning` | ce=0.30, pos=0.05 | 测试中等 dims + preference |
| R6 | 32 | `redacted_name_preference` | alpha=0.30, beta=0.10 | 测试真正替代身份目标 |

执行顺序建议：

```text
1. 先跑 R1/R2/R3，判断 objective 是否有效。
2. 如果 R1/R2 有 CE 提升，跑 R4。
3. 如果 preference 有优势，跑 R5。
4. 实现 redacted_name_preference 后跑 R6。
5. 只有 cheap gate 通过，才进入 Full CLEAR。
```

---

## 10. 失败分支处理

| 失败现象 | 优先解释 | 处理动作 |
| --- | --- | --- |
| `target_name_ce delta < 0.002` | objective 没打到行为，或 dims 覆盖不足 | 先检查 Step7 mask，再试 32 dims / redacted preference |
| CE delta 上升但 `name_hit` 不降 | 模型仍缺少替代生成目标 | 实现 redacted-name positive target |
| `name_hit` 降但 retain 也降 | 编辑过强或 NameRet guard 不够 | 降 dims、提高 retain loss、收紧 NameRet |
| 32 dims 有效但 48 dims retain 失败 | 48 dims 超过选择性编辑上限 | 以 32 dims 为主配置 |
| Step7 loss 正常但行为无变化 | loss 只能说明接入，不说明遗忘 | 继续以 Step8 cheap diagnostic 为准 |
| Full CLEAR 前 cheap gate 未通过 | 当前 run 不具备扩大评估价值 | 不跑 Full CLEAR，回到 candidate/objective |

---

## 11. 进入 Full CLEAR 的条件

只有同时满足以下条件，才进入 Full CLEAR：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= baseline 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

并且 Step7 summary 必须满足：

```text
projector_modules > 0
projector_editable_neurons > 0
whole-vector projector = false
forget_ce_token_count > 0
```

如果使用 preference objective，还必须满足：

```text
positive_token_count > 0
positive_loss 有记录
```

---

## 12. 一句话执行策略

```text
先用 name_ce_ascent 判断“压姓名能不能打动模型”，
再用 32/48 projector dims 判断“是不是编辑覆盖不足”，
最后用真正的 redacted_name_preference 解决“模型不知道该输出什么替代身份”的问题。
Full CLEAR 只在 cheap gate 通过后执行。
```
