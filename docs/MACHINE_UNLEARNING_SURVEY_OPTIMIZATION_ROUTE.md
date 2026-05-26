# 基于 Machine Unlearning Survey 的 CHIP-Editor 优化路线

本文档基于当前 CHIP-Editor / MIP-Editor 实验结果，讨论如何从 Machine Unlearning survey 与 `awesome-machine-unlearning` 中借鉴方法思想，改进当前 causal masked RMisU 实验。

参考对象：

- 当前 CHIP-Editor 实验结果：`EXPERIMENT_RESULT_CHIP_FULL_0526_1348.md`
- 原 MIP-Editor baseline 结果：`EXPERIMENT_RESULT_0430_141446.md`
- 当前技术路线：`CHIP_EDITOR_TECHNICAL_ROADMAP.md`
- Machine Unlearning survey 索引：<https://github.com/tamlhp/awesome-machine-unlearning>

## 1. 当前实验现象

当前 `chip_full_train_0526_1348` 的工程链路已经完整跑通：

```text
Step2 candidate paths
-> Step3 causal pairs
-> Step4 activation cache / patching
-> Step5 Nec / Suf / Ret causal scores
-> Step6 path classification
-> Step7 masked RMisU editing
-> Step8 final evaluation
```

但实验效果弱于原始 MIP-Editor baseline。

主要现象如下：

| 指标 | 当前 CHIP full 0526 | 原 0430 baseline |
|---|---:|---:|
| forget classification | 远程评分中仍较高 | 明显下降 |
| forget generation | 远程评分中仍较高 | 大幅下降 |
| retain performance | 基本保持甚至更强 | 基本保持 |
| Step8 forget name_hit_rate | 未下降 | 原协议中表现为明显遗忘 |

当前结果说明：

```text
模型整体能力没有明显损坏，但目标 forget-set 知识没有被有效擦除。
```

也就是说，当前失败不是工程失败，而是 unlearning 信号没有真正压低目标身份 / 图文关联知识。

## 2. 问题诊断

### 2.1 编辑范围偏小且偏 text

当前 `step6_v1_quantile75` 的最终 mask 较保守：

```text
P_forget = 8 paths
editable neurons = 45
mask modules = 18
```

并且这些 `P_forget` 主要来自 text path，vision / cross-modal path 基本没有进入最终编辑集合。

这与 CLEAR 的多模态身份遗忘目标不完全匹配。对于图像人物身份、图文绑定关系、视觉实体识别等知识，仅编辑少量 text FFN neuron 很可能不够。

### 2.2 当前 masked RMisU 的 forget 信号不直接对齐评估指标

当前 masked RMisU 的核心 forget loss 是：

```text
forget activation -> random activation target
```

但 Step8 和 Full CLEAR 远程评分主要评估：

```text
模型是否还能答出目标姓名 / 身份 / 目标知识
```

因此 hidden activation MSE 不一定能稳定传导到输出层的目标答案概率下降。

### 2.3 forget CE 项方向存在风险

当前 loss 中 forget CE 以负号加入：

```text
loss = beta * unlearn_loss
     + alpha * retain_loss
     + shared_alpha * shared_loss
     - forget_ce_alpha * forget_ce_loss
```

最小化该 loss 时，`- forget_ce_alpha * forget_ce_loss` 会鼓励模型增大 forget CE。这个方向理论上可以用于遗忘，但权重较小：

```text
0.5 * unlearn_loss ~= 35
0.1 * forget_ce_loss ~= 0.44
```

所以输出层面的遗忘信号相对很弱，主导训练的仍是 activation RMU。

### 2.4 原 baseline 的强干预没有被充分继承

原 MIP-Editor baseline 中有更直接的 neuron prune / zeroing / adaptive RMU 流程。当前 causal masked RMisU 更精细，但干预强度明显降低。

这解释了为什么当前模型 retain 很稳，但 forget 没有明显下降。

## 3. 可借鉴的 Machine Unlearning 思想

从 survey 和 `awesome-machine-unlearning` 中，最适合当前项目的不是完整照搬某个方法，而是吸收以下几类思想。

### 3.1 SalUn / gradient saliency：用梯度显著性辅助选 mask

相关思想：

```text
Forget-set 上梯度显著、retain-set 上梯度不显著的参数，更适合作为编辑目标。
```

当前 Step6 主要依赖 causal patching 得到的 `Nec / Suf / Ret`。建议加入 gradient saliency，构造新的 path / neuron 选择分数：

```text
edit_score =
  causal_forget_effect
  - lambda * retain_impact
  + gamma * forget_gradient_saliency
  - eta * retain_gradient_saliency
  + mu * mip_score
```

为什么可能提升：

- Causal score 说明 path 对行为有影响。
- Gradient saliency 说明当前训练目标真正会更新这些参数。
- 二者结合可以避免只选到 causal patching 上有影响、但实际训练中难以推动的 neuron。
- 对 retain gradient 做惩罚，可以降低误伤 shared utility path 的概率。

### 3.2 SSD / ASSD：先做 selective synaptic dampening

相关思想：

```text
对 forget-set 重要、retain-set 不重要的参数做选择性衰减。
```

可以在 masked RMisU 之前加入 warm start：

```text
damp_factor_i = f(forget_importance_i / retain_importance_i)
theta_i <- damp_factor_i * theta_i
```

仅作用于 `P_forget` 对应的参数切片，例如：

```text
up_proj / gate_proj selected columns
down_proj selected rows or related projected directions
```

为什么可能提升：

- 当前 masked RMisU 只开放 45 个 neuron 训练，1 epoch 内不足以制造明显遗忘。
- Selective dampening 能先产生可观的忘却效果。
- 后续再用 retain KD / RMU 修复 retain，整体更接近原 baseline 的强干预逻辑。

### 3.3 SCRUB / KGA / knowledge adaptation：retain 蒸馏 + forget 反蒸馏

相关思想：

```text
retain-set 上模仿原模型，forget-set 上远离原模型或对齐替代目标。
```

当前可以改为：

```text
L_retain_KD = KL(updated(retain), frozen(retain))
L_forget_KD = - KL(updated(forget_clean), frozen(forget_clean))
```

或者更稳定地使用 corrupt / counterfactual target：

```text
L_forget_pair = KL(updated(forget_clean), frozen(forget_corrupt))
```

为什么可能提升：

- 仅对 hidden activation 做随机扰动，不保证输出层不再说出目标姓名。
- 使用 corrupt / counterfactual target 可以把模型行为直接推向“非目标身份”或“无法识别目标身份”。
- retain KD 可以保持 hard retain / counterfactual retain 的能力。

### 3.4 DPO / AltPO：把遗忘写成偏好优化

相关思想：

```text
preferred response: 不泄露目标知识的回答
rejected response: 原始 forget answer
```

对每个 forget sample 构造：

```text
positive: corrupt answer / unknown answer / counterfactual answer
negative: original target answer
```

优化目标可写为：

```text
L_forget_pref =
  - log sigmoid(
      beta * [
        logp(positive) - logp(negative)
      ]
    )
```

为什么可能提升：

- Step8 的 `name_hit_rate` 和远程 judge 关注的是输出内容。
- 偏好优化直接降低目标答案相对概率，比 hidden MSE 更贴近评价协议。
- 对身份遗忘任务，pairwise preference 通常比单纯 CE ascent 更稳定。

### 3.5 CaMU / causal disentanglement：保留因果路径分离，但提高判定质量

当前 CHIP-Editor 已经有 CaMU 风格的因果思想：

```text
P_cand -> Nec/Suf/Ret -> P_forget/P_shared
```

建议保留这条主线，但改进 Step6：

```text
旧版：
path mean score -> quantile threshold -> category

新版：
path x pair score matrix
-> sample coverage
-> stability under corruption
-> gradient saliency
-> retain impact penalty
-> category
```

为什么可能提升：

- 当前 simple mean 会稀释 sample-specific identity path。
- 身份知识往往不是单一路径全样本共享，而是样本簇 / 模态簇相关。
- coverage 和 stability 能更好地区分“目标身份路径”和“通用视觉主题路径”。

## 4. 推荐优化路线

### Phase 1：重做 Step6 mask，得到 `P_forget_v2`

目标：

```text
让最终编辑集合覆盖真正的 text + vision + cross-modal forget path。
```

具体修改：

1. 合并现有候选来源：

```text
step6_v1
step6_v1_quantile75
step6_vision_stable_0522
step6_vision_text_5x5
```

2. 对每个 path 保留如下统计：

```text
mean_forget_effect
mean_retain_impact
forget_effect_coverage
retain_impact_coverage
num_effective_pairs
path_modality
mip_score
forget_gradient_saliency
retain_gradient_saliency
```

3. 新的选择标准：

```text
P_forget_v2 =
  high forget_effect
  high forget coverage
  low retain impact
  high forget gradient saliency
  low retain gradient saliency
```

4. 单独保留三组 mask：

```text
P_forget_text_v2
P_forget_vision_v2
P_forget_cross_modal_v2
```

预期提升：

- 提高多模态身份路径覆盖率。
- 避免只编辑 text FFN 导致图像身份关联仍然存在。
- 降低误伤 retain 的风险。

### Phase 2：加入 selective dampening warm start

目标：

```text
在 masked RMisU 前先制造真实遗忘效果。
```

建议形式：

```text
for neuron in P_forget_v2:
    importance_ratio = forget_importance / (retain_importance + eps)
    if importance_ratio > threshold:
        damp selected weights
```

可先做保守版本：

```text
damp_factor = 0.7
```

然后网格搜索：

```text
damp_factor in {0.9, 0.7, 0.5, 0.3}
```

预期提升：

- 比单纯 masked RMisU 更快压低 forget-set 表现。
- 与原 MIP-Editor 的 prune / zeroing 强干预更接近。
- dampening 比 hard zero 更平滑，retain 损伤可控。

### Phase 3：把 forget loss 改成 pair-aware objective

目标：

```text
让训练目标直接压低目标答案 / 目标姓名。
```

建议从三个 loss 中选一个 MVP：

#### 方案 A：corrupt activation target

```text
L_forget = MSE(
  activation(updated, forget_clean),
  activation(frozen, forget_corrupt)
)
```

优点：

- 与现有 Step4 activation cache / patching 兼容。
- 比 random target 更有语义。

#### 方案 B：output-level corrupt KD

```text
L_forget = KL(
  logits(updated, forget_clean),
  logits(frozen, forget_corrupt)
)
```

优点：

- 直接改变输出分布。
- 更贴近 Step8 / remote judge。

#### 方案 C：preference unlearning

```text
positive = corrupt / unknown / counterfactual answer
negative = original forget answer

L_forget_pref =
  -log sigmoid(logp(positive) - logp(negative))
```

优点：

- 直接降低目标答案相对概率。
- 对 name_hit_rate 更有针对性。

推荐顺序：

```text
先做 B，再做 C。
```

原因：

- B 的工程成本较低。
- C 的行为最贴近最终评估，但需要构造稳定 positive answer。

### Phase 4：retain 侧从 activation preserve 升级为 KD preserve

当前 retain preserve 主要在 selected activation 上约束。建议增加输出层 KD：

```text
L_retain_KD =
  KL(logits(updated, retain), logits(frozen, retain))
```

retain 数据应包括：

```text
retain_loader
hard_retain::same_topic
hard_retain::same_reasoning
counterfactual_retain
```

预期提升：

- 更好保护最终生成行为，而不是只保护局部 activation。
- 降低 counterfactual retain name_hit_rate 下降的问题。

### Phase 5：residual failure mining

目标：

```text
针对仍然泄露目标姓名的样本做第二轮编辑。
```

流程：

```text
Step8 evaluation
-> collect forget_clean samples with name_hit = 1
-> recompute causal / gradient saliency on these samples
-> expand P_forget_v2
-> dampening + pair-aware unlearning round 2
```

预期提升：

- 解决 concept revival / bypass path 问题。
- 把 unlearning 从一次性静态 mask 变成迭代修正。

## 5. 推荐实验矩阵

### Experiment 1：mask v2 ablation

目的：

```text
验证是否是 mask 选得太少 / 太偏 text。
```

设置：

| Run | Mask | Editing |
|---|---|---|
| E1.1 | 当前 quantile75 | 当前 masked RMisU |
| E1.2 | step6_v1 较宽 mask | 当前 masked RMisU |
| E1.3 | vision_stable + text | 当前 masked RMisU |
| E1.4 | causal + gradient saliency v2 | 当前 masked RMisU |

判断标准：

```text
forget_clean name_hit_rate 是否下降
Full CLEAR forget acc 是否下降
counterfactual retain 是否保持
```

### Experiment 2：selective dampening strength

目的：

```text
验证强干预是否能恢复原 baseline 的遗忘效果。
```

设置：

| Run | damp factor | 后续训练 |
|---|---:|---|
| E2.1 | none | masked RMisU |
| E2.2 | 0.9 | masked RMisU |
| E2.3 | 0.7 | masked RMisU |
| E2.4 | 0.5 | masked RMisU |
| E2.5 | 0.3 | masked RMisU |

判断标准：

```text
找到 forget 明显下降且 retain 未 collapse 的最小干预强度。
```

### Experiment 3：forget objective ablation

目的：

```text
验证输出级目标是否比 random activation target 更有效。
```

设置：

| Run | Forget loss | Retain loss |
|---|---|---|
| E3.1 | random activation RMU | activation preserve |
| E3.2 | corrupt activation target | activation preserve |
| E3.3 | corrupt output KD | output retain KD |
| E3.4 | preference unlearning | output retain KD |

判断标准：

```text
name_hit_rate 和 remote forget acc 是否下降。
```

### Experiment 4：residual mining

目的：

```text
验证多轮修正是否能继续降低泄露样本。
```

设置：

```text
Round 1: P_forget_v2 + dampening + output KD
Round 2: only residual name-hit samples
Round 3: optional
```

判断标准：

```text
forget_clean name_hit_rate 是否逐轮下降
hard_retain / counterfactual_retain 是否稳定
```

## 6. 最推荐的第一版实现

如果只做一版最小可行优化，建议如下：

```text
1. 生成 P_forget_v2:
   causal score + gradient saliency + retain penalty

2. masked RMisU 前加入 selective dampening:
   damp_factor = 0.7

3. forget loss 从 random activation target 改为 corrupt output KD:
   updated(forget_clean) -> frozen(forget_corrupt)

4. retain loss 增加 output KD:
   updated(retain/hard_retain/counterfactual_retain) -> frozen(...)

5. 保留 Step8 + Full CLEAR remote protocol 做对比。
```

推荐总 loss：

```text
L =
  beta_forget * L_forget_corrupt_KD
  + beta_activation * L_forget_activation_RMU
  + alpha_retain * L_retain_KD
  + alpha_shared * L_shared_activation_preserve
```

建议初始权重：

```text
beta_forget = 1.0
beta_activation = 0.2
alpha_retain = 1.0
alpha_shared = 0.5
```

不建议继续使用很弱的 `-0.1 * forget_ce_loss` 作为主要遗忘信号。

## 7. 为什么这条路线比当前方案更可能提升

### 7.1 更贴近目标知识所在位置

当前只编辑少量 text FFN neuron，可能没有覆盖多模态身份关联。mask v2 会显式纳入 vision / cross-modal path。

### 7.2 更贴近最终评估指标

当前 hidden random RMU 不直接约束输出答案。output KD / preference unlearning 会直接压低目标答案概率。

### 7.3 更接近原 baseline 的成功机制

原 baseline 有更强的 neuron-level destructive edit。selective dampening 可以恢复这种强干预，但比 hard zero 更可控。

### 7.4 更好控制 retain 损伤

retain KD 比局部 activation preserve 更接近最终行为表现，适合保护 hard retain 和 counterfactual retain。

### 7.5 能处理旁路恢复

residual failure mining 可以处理第一次编辑后仍然泄露的样本，降低 concept revival 风险。

## 8. 风险与注意事项

### 8.1 Dampening 过强可能损伤 retain

需要用小网格搜索确定强度。优先从 `0.9 / 0.7 / 0.5` 开始，不建议第一轮直接 hard zero。

### 8.2 Corrupt target 质量会影响 forget 行为

如果 `forget_corrupt` 不是合理替代目标，模型可能学到奇怪输出。需要检查 corrupt answer 是否稳定。

### 8.3 Preference unlearning 需要可靠 positive answer

如果使用 unknown / refusal answer，应保证格式与评估协议兼容，否则 remote judge 可能误判。

### 8.4 Step8 val 样本较少

Step8 val 只有 18 个 forget_clean 样本，容易波动。最终仍需 Full CLEAR remote protocol 验证。

## 9. 结论

当前 CHIP-Editor 的因果链路是有价值的，但 Step7 的编辑机制过于保守，且 forget objective 没有直接对齐输出评估。

最值得优先尝试的优化是：

```text
Causal-Saliency Mask v2
+ Selective Dampening Warm Start
+ Pair-aware Output KD / Preference Unlearning
+ Retain KD Preserve
+ Residual Failure Mining
```

这条路线的核心不是放弃 CHIP-Editor，而是把 Machine Unlearning survey 中成熟的 saliency、selective dampening、knowledge adaptation、preference optimization 思想接入现有 causal path 框架，让 causal localization 负责“选哪里”，unlearning objective 负责“怎么真正忘掉”。
