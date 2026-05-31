# Step10 下一步优化：借鉴 Machine Unlearning 思想的实现路线

更新时间：2026-05-30

本文基于：

```text
docs/STEP10_STAGE_SUCCESS_AND_LESSONS_0530.md
docs/PROJECT_STRUCTURE.md
https://github.com/tamlhp/awesome-machine-unlearning
```

目标是回答：

```text
下一步如何优化当前 Step10？
awesome-machine-unlearning 中有哪些思想可以借鉴到当前 MIP-Editor 问题？
```

结论先行：

```text
有可以借鉴的思想，但不应直接照搬完整方法。

当前最值得借鉴的是：
1. PII token-level unlearning
2. preference / positive-target unlearning
3. MLLM projector 稳定化
4. retain-anchor guided mask
5. Fisher / saliency specificity 的参数选择思想
```

这些思想应该被落到当前 Step10 的具体断点上：

```text
projector 粒度从 whole-vector 改为 projector-dim；
训练目标从单纯 target name CE ascent 改为 name preference / replacement；
retain 保护从 coarse P_shared 改成 dim-level retain-anchor mask；
评估从 name_hit 扩展到 PII-style adversarial leakage。
```

---

## 1. 当前问题复盘

Step10 已经完成的阶段性成功：

```text
1. 找到了 target-name sensitive projector / vision_text path。
2. Step6 name-aware 分类能把 projector path 从 P_shared 拉回 P_forget。
3. Step7 strict name-token CE scope 已经跑通。
4. projector whole-vector mask 语义已经修正。
5. cheap diagnostic 能提前否决不合格 run。
```

但 Step7 small grid 没有达到行为成功：

| run | forget_clean target_name_ce delta | forget_clean name_hit | 主要问题 |
| --- | ---: | ---: | --- |
| A maxproj4 ce0.05 | -0.003828 | 0.6667 | forget 方向错误，且 projector 未实际编辑 |
| B maxproj4 ce0.10 | +0.000429 | 0.7222 | CE 正向太弱，generation 失败 |
| C maxproj8 ce0.10 | +0.000421 | 0.6667 | projector 进入训练，但 retain 风险上升 |

核心判断：

```text
链路已经通了，但当前 objective 和 projector 粒度还不够准。
继续增加 steps 或继续 whole-vector projector，不是优先方向。
```

---

## 2. 外部思想一：PII token-level unlearning

参考方向：

```text
Machine Unlearning of Personally Identifiable Information in Large Language Models
PERMU / PERMUtok
```

该类工作强调：

```text
PII unlearning 不应把整段回答等价处理；
需要针对 PII token 做更细粒度的干预；
评估也要覆盖 direct / paraphrased / indirect leakage。
```

对当前项目的启发：

```text
target name 本质上就是 PII-like identity token。
Step10 不应该再主要优化整段 answer；
应围绕 name_token_positions 做 token-level unlearning。
```

建议新增 Step7 objective：

```text
pii_name_token_noise
```

实现含义：

```text
只在 forget 样本的 target name token hidden states / logits 上注入扰动；
不扰动非姓名描述 token；
retain 样本使用 retain/shared loss 约束。
```

建议优先做轻量版本：

```text
1. 找到 name_token_positions。
2. 在这些 token 的 projector / LM hidden states 上加入小幅 noise target。
3. loss 使用 KL / MSE 让 forget name token 的表示远离 clean 表示。
4. 保留 retain name token 的 clean KL。
```

对应当前代码落点：

| 位置 | 修改 |
| --- | --- |
| `causal_mip/causal_scores/name_token_metrics.py` | 复用 name token 定位 |
| `causal_mip/editing/masked_rmisu.py` | 新增 `pii_name_token_noise` objective |
| `causal_mip/evaluation/step8_probability_diagnostic.py` | 增加 PII-style leakage prompt |

---

## 3. 外部思想二：Preference / positive-target unlearning

参考方向：

```text
Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models
AltPO
```

该类方法的关键判断：

```text
只给 negative feedback 压制 forget answer，
容易导致输出不稳定、无意义或 retain 退化。
更合理的是同时提供 in-domain positive feedback。
```

当前 Step10 的问题正好类似：

```text
name_ce_ascent / activation_random_name_ce 都偏 negative feedback；
它们能让 target_name_ce 轻微上升，
但没有稳定转化为 name_hit 下降。
```

建议新增 Step7 objective：

```text
name_preference_unlearning
```

训练样本构造：

```text
x = 原图 + 原问题
y_neg = 原始 target answer，包含 target name
y_pos = name-redacted / unknown-person / generic-identity answer，不包含 target name
```

示例：

```text
y_neg: "This is Bao Nguyen at a street food stall in Hanoi."
y_pos: "The image shows a person at a street food stall in Hanoi."
```

目标：

```text
降低 y_neg preference；
提高 y_pos preference；
retain 样本保持原答案 preference。
```

这比单纯让姓名 token CE 上升更贴近最终评估：

```text
forget_clean name_hit 下降；
回答仍然自然；
不强迫模型输出随机文本。
```

对应当前代码落点：

| 位置 | 修改 |
| --- | --- |
| `causal_mip/data_pairs/` | 增加 name-redacted positive target 构造 |
| `causal_mip/editing/masked_rmisu.py` | 新增 preference-style loss |
| `causal_mip/evaluation/step8_final_eval.py` | 记录 positive target hit / target name hit |

建议 loss 形式：

```text
L_forget_pref = -log sigmoid(beta * [
  log pi_theta(y_pos | x) - log pi_ref(y_pos | x)
  - log pi_theta(y_neg | x) + log pi_ref(y_neg | x)
])
```

retain loss 继续保留：

```text
L_retain = KL(pi_theta(retain) || pi_ref(retain))
```

---

## 4. 外部思想三：MLLM projector 稳定化

参考方向：

```text
SineProject: Machine Unlearning for Stable Vision-Language Alignment
```

该类 MLLM unlearning 工作指出：

```text
MLLM unlearning 容易破坏 vision-language alignment；
问题与 projector 网络的优化不稳定有关；
需要受控地编辑 projector，而不是粗暴更新。
```

这和当前 C run 的现象一致：

```text
whole-vector projector 编辑确实让 forget_clean target_name_ce 有微弱正向变化；
但 counterfactual_retain / hard_retain name CE 也上升。
```

因此下一步不应继续：

```text
whole-vector visual.merger.mlp.0 5120 维全量编辑。
```

应改为：

```text
projector-dim candidate
+ bounded projector adapter
+ dim-level retain guard
```

建议实现方式：

```text
1. 使用已有 projector-dim candidates。
2. Step5 计算每个 dim 的 NameNec / NameSuf / NameRet。
3. Step6 只让 high NameSuf / low NameRet 的 dim 进入 P_forget。
4. Step7 只更新这些 dim 对应的 adapter / delta。
5. 对 delta 使用 bounded parameterization。
```

bounded update 可先用轻量版本：

```text
delta = scale * tanh(raw_delta)
W_eff = W_base + mask * delta
```

优势：

```text
避免 projector whole-vector 大幅漂移；
让 forget 修改可控、可回滚；
保留现有 masked RMisU 结构。
```

对应当前代码落点：

| 位置 | 修改 |
| --- | --- |
| `causal_mip/path_localization/projector_dim_export.py` | projector-dim candidate 主入口 |
| `causal_mip/causal_scores/build_scores.py` | 支持 projector-dim name score |
| `causal_mip/causal_scores/classify_paths.py` | dim-level P_forget / P_shared |
| `causal_mip/editing/masked_rmisu.py` | bounded projector adapter / delta |

---

## 5. 外部思想四：Retain-anchor guided mask

参考方向：

```text
SMFA: Sculpted Memory Forgetting Adapter
```

该类 MLLM selective unlearning 方法强调：

```text
先构造 memory forgetting adapter；
再用 retaining anchor-guided masking 限制对无关视觉理解能力的影响。
```

映射到当前项目：

```text
P_forget 负责姓名遗忘；
P_shared / P_retain 不能只是粗粒度 path guard；
应成为 retain-anchor mask。
```

建议改法：

```text
retain_anchor_score = max(0, NameRet) + lambda * retain_grad_norm
forget_score = max(0, NameSuf) + alpha * max(0, NameNec)

editable_score = forget_score / (epsilon + retain_anchor_score)
```

进入 Step7 的 dim 应满足：

```text
NameSuf > 0
editable_score 高
NameRet 不在 top retain-sensitive bucket
跨 pair 稳定
```

这比当前的 coarse 规则更适合 C run 暴露的问题：

```text
projector 有 forget signal；
但同一 projector whole-vector 也承担 retain signal。
```

---

## 6. 外部思想五：Fisher / saliency specificity

参考方向：

```text
SalUn
SSD: Selective Synaptic Dampening
```

相关工作共同强调：

```text
不要全参数或粗粒度更新；
先找对 forget set disproportionately important 的参数；
再做选择性 dampening / editing。
```

当前项目已经有类似基础：

```text
causal_mip/causal_scores/saliency_specificity.py
```

建议下一步把它接到 name-token 和 projector-dim：

```text
forget_name_grad = grad(target_name_loss, projector_dim)
retain_name_grad = grad(retain_name_loss, projector_dim)
name_specificity = forget_name_grad / (epsilon + retain_name_grad)
```

dim-level 排序不只看 NameSuf，还看：

```text
NameSuf
NameNec
NameRet
name_specificity
pair_stability
```

---

## 7. 推荐的下一步实验矩阵

不建议直接跑 Full CLEAR。

先跑 20-pair / val cheap grid：

| run | candidate | objective | projector policy | 目标 |
| --- | --- | --- | --- | --- |
| D | projector-dim name-aware | `activation_random_name_ce` alpha 0.20 | dim-level | 检查 CE 是否能明显上升 |
| E | projector-dim name-aware | `name_ce_ascent` alpha 0.05 | dim-level | 测试纯 output-level 小剂量 |
| F | projector-dim name-aware | `name_preference_unlearning` | dim-level | 直接优化 name_hit 相关行为 |
| G | projector-dim name-aware | `name_preference_unlearning` | bounded adapter | 检查 retain 是否更稳 |
| H | projector-dim name-aware | `pii_name_token_noise` | bounded adapter | 测试 token-level PII 风格扰动 |

推荐优先级：

```text
F > G > D > E > H
```

原因：

```text
F 最直接对准 generation-level name_hit；
G 进一步控制 projector retain 风险；
D/E 是用于测量 output-level 强度的 baseline；
H 有价值，但需要更仔细调 noise 强度。
```

---

## 8. Cheap gate

每个 run 先过 cheap gate，再考虑 Full CLEAR：

```text
forget_clean target_name_ce delta > 0.01
forget_clean target_name_mean_logprob delta < 0
forget_clean name_hit <= 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

新增 PII-style leakage gate：

```text
direct_name_question target name hit
paraphrased_question target name hit
first_name_only_prompt target name hit
surname_only_prompt target name hit
indirect_identity_question target name hit
```

通过标准建议：

```text
forget-side adversarial leakage 不高于 forget_clean name_hit；
retain-side name_hit drop <= 0.05；
answer naturalness 不明显崩坏。
```

---

## 9. 实现顺序

### Phase 1: projector-dim name-aware scoring

输入：

```text
../mip_workspace/outputs/paths/P_cand_projector_dim_retainaware_top4_dedup_20pairs_0529.jsonl
../mip_workspace/outputs/paths/P_cand_retainaware_node_projdim_20pairs_0529.jsonl
../mip_workspace/outputs/paths/P_cand_stable_retainaware_node_projdim_20pairs_0529.jsonl
```

输出：

```text
../mip_workspace/outputs/scores/path_name_scores_projector_dim_<RUN_ID>.jsonl
```

新增字段：

```text
projector_dim
NameNec
NameSuf
NameRet
name_specificity
pair_stability
editable_score
```

### Phase 2: dim-level Step6

输出：

```text
../mip_workspace/outputs/paths/step6_name_aware_projector_dim_<RUN_ID>/
```

要求：

```text
P_forget 中 projector path 必须是具体 dim；
P_shared 中 projector guard 也必须是具体 dim；
禁止 coarse mm_projector placeholder 抵消整个 projector。
```

### Phase 3: Step7 objective 扩展

新增：

```text
name_preference_unlearning
pii_name_token_noise
bounded_projector_adapter
```

保留现有：

```text
activation_random_name_ce
name_ce_ascent
target_ce_scope=name
```

### Phase 4: probability + leakage diagnostic

扩展：

```text
step8_probability_diagnostic.py
```

增加：

```text
paraphrased target-name prompts
partial-name prompts
indirect identity prompts
name-redacted answer preference check
```

---

## 10. 不建议做的事

当前不建议：

```text
1. 继续 whole-vector projector 编辑。
2. 只把 Step7 steps 从 200 加到更大。
3. 只调 forget_ce_alpha。
4. cheap gate 未通过就跑 Full CLEAR。
5. 用训练 loss 下降替代 target_name_ce / name_hit 判断。
```

原因：

```text
这些方向不能解决当前真正的问题：
projector 粒度太粗；
negative-only objective 太弱；
retain guard 不是 dim-level；
评估还没有覆盖 PII-style leakage。
```

---

## 11. 最小可执行版本

如果只做一轮最小有效优化，建议如下：

```text
Step A: 用 projector-dim candidates 重新跑 NameNec / NameSuf / NameRet。
Step B: 用 editable_score 选 top projector dims。
Step C: Step7 跑 name_preference_unlearning，先不加复杂 adapter。
Step D: 如果 retain 风险仍高，再加入 bounded projector adapter。
Step E: 过 cheap gate 后才跑 Full CLEAR。
```

最小 run：

```text
RUN_ID=step10_projector_dim_name_pref_0530
candidate=projector_dim_name_aware
objective=name_preference_unlearning
target_ce_scope=name
steps=200
batch_size=2
learning_rate=1e-5
retain_guard=dim_level
```

预期成功信号：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= 0.6111
retain name_hit drop <= 0.05
projector_modules > 0
projector editable dims > 0
whole-vector projector dims = 0
```

---

## 12. 参考链接

```text
awesome-machine-unlearning:
https://github.com/tamlhp/awesome-machine-unlearning

Machine Unlearning of Personally Identifiable Information in Large Language Models:
https://aclanthology.org/2025.nllp-1.6/

Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models:
https://huggingface.co/papers/2409.13474

SineProject: Machine Unlearning for Stable Vision-Language Alignment:
https://arxiv.org/abs/2511.18444

Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning:
https://arxiv.org/abs/2511.20196

SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models:
https://arxiv.org/abs/2504.02883

Selective Synaptic Dampening:
https://huggingface.co/papers/2308.07707
```

---

## 13. 一句话结论

```text
下一步不要再扩大 whole-vector Step7；
应把 Step10 升级为 projector-dim + name preference + retain-anchor mask。

这条路线最贴合当前证据：
target name signal 确实在 projector / vision_text path 中；
但必须用 dim-level、bounded、preference-style 的方式编辑，
否则 forget 信号太弱或 retain 风险太高。
```
