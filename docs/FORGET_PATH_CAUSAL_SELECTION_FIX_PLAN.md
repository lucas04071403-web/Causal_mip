# Forget Path 因果选择质量修正方案

更新时间：2026-05-27

本文档整理针对 `step9_bound_projector_0526_214028` 实验失败点的讨论，目标是说明如何修正 forget path 的因果选择质量，尤其让 projector / cross-modal path 真正进入编辑集合，并结合 machine unlearning survey 中可借鉴的思想给出理由。

参考背景：

- 当前项目文档：`docs/PROJECT_STRUCTURE.md`
- 最新实验结果：`docs/EXPERIMENT_RESULT_STEP9_BOUND_PROJECTOR_0526_214028.md`
- 原 baseline 实验：`docs/EXPERIMENT_RESULT_0430_141446.md`
- Survey repo：`https://github.com/tamlhp/awesome-machine-unlearning`

## 1. 当前问题判断

`step9_bound_projector_0526_214028` 已经跑通完整工程链路：

```text
Step2/3 bound candidate paths
-> Step5 causal scores
-> Step6 P_forget/P_shared/P_retain/P_irrelevant
-> Step7 masked RMisU checkpoint
-> Step8 pair screen + Full CLEAR
```

但实验结果显示：

```text
Full CLEAR main: passed
Step8 total protocol: failed
failed reason: pair_screen:pair_forget_clean_name_hit_rate
```

关键失败点：

| 指标 | candidate | PEFT baseline | delta |
| --- | ---: | ---: | ---: |
| `forget_clean name_hit_rate` | 0.777778 | 0.611111 | +0.166667 |

这说明候选模型不但没有降低目标身份命中，反而比 baseline 更容易命中 forget_clean 身份。

Full CLEAR forget 侧虽然形式上下降：

| 指标 | candidate | PEFT baseline | delta |
| --- | ---: | ---: | ---: |
| forget classification remote acc | 0.675532 | 0.686170 | -0.010638 |
| forget generation remote acc | 0.457447 | 0.462766 | -0.005319 |

但换成样本数仅约为：

```text
forget classification: 129/188 -> 127/188
forget generation:     87/188  -> 86/188
```

因此这只是非常弱的下降，不能说明遗忘成功。

另一个关键现象是：

```text
Step7 summary: num_projector_modules = 0
```

虽然本轮启用了 bound projector 路线，且代码中已有 projector 编辑入口，但实际进入编辑集合的模块仍主要是 LLM / vision `down_proj`，projector 没有真正被编辑。

结论：

```text
当前问题不是单纯训练不够，而是 forget path 的因果选择质量不足；
尤其是 cross-modal / projector path 没有作为有效编辑对象进入 Step7。
```

## 2. 总体修正原则

修正目标不是直接替换成某个通用 machine unlearning 方法，而是把 survey 中的思想嵌入当前 CHIP-Editor / causal selective editing 流程：

1. 用 saliency / Fisher / gradient ratio 改善 projector path 选择。
2. 用 causal necessity + sufficiency + retain impact 共同筛 path。
3. 用 modality-aware ranking 防止 `vision_text` 被 text / vision 大量路径淹没。
4. 用 negative preference / name-token suppression 改造 forget objective。
5. 用 retain anchor / KL preserve 保护跨模态对齐和 retain 能力。

## 3. 修正方案一：把 projector 进入编辑集合变成硬性可观测指标

当前出现了 silent failure：

```text
run name 包含 projector
Step7 代码支持 projector
但最终 num_projector_modules = 0
```

因此需要在每个阶段增加强制诊断。

### 3.1 Step2/3 诊断

必须统计：

```text
num_candidate_paths
num_text_paths
num_vision_paths
num_vision_text_paths
num_paths_containing_mm_projector
```

并检查每条 `vision_text` path 是否包含：

```text
vision node
mm_projector node
LLM image-token node
LLM answer-token node
```

### 3.2 Step5 诊断

必须统计：

```text
vision_text.num_patchable_nodes == num_nodes 的比例
projector node 是否被 activation cache
projector node 是否被 ablation
projector node 是否被 restoration
```

建议 Step5 score record 增加字段：

```json
{
  "resolved_nodes": [],
  "skipped_nodes": [],
  "num_nodes": 4,
  "num_patchable_nodes": 4,
  "contains_projector": true,
  "projector_patchable": true,
  "clean_score": 0.0,
  "corrupt_score": 0.0,
  "restored_score": 0.0,
  "target_answer_text": "...",
  "answer_token_ids": [],
  "answer_token_positions": [],
  "restore_token_positions": []
}
```

### 3.3 Step6 诊断

每个 category 都要统计：

```text
P_forget: vision_text count, projector count
P_shared: vision_text count, projector count
P_retain: vision_text count, projector count
P_irrelevant: vision_text count, projector count
```

如果 `P_forget` 和 `P_shared` 中没有 projector path，应直接视为本轮分类不合格，而不是继续跑 Step7。

### 3.4 Step7 诊断

Step7 summary 必须显式输出：

```text
num_projector_modules
num_projector_editable_neurons
projector_edit_modules
projector_skipped_modules
projector_skip_reasons
```

验收标准：

```text
num_projector_modules > 0
```

理由：

当前失败不是 loss 层面的问题，而是路径筛选和编辑对象对不上。没有这些硬性诊断，继续调训练超参可能只是更充分地编辑错误路径。

## 4. 修正方案二：重做 projector / cross-modal candidate path 生成

当前 cross-modal builder 仍存在占位逻辑：

```text
mm_projector neuron = placeholder
或固定 neuron = 0
```

这会导致 Step5/Step7 即使看见 projector 节点，也不一定对应真实敏感维度。

### 4.1 用 forget/retain saliency ratio 选择 projector dim

建议为 projector hidden dimension 计算：

```text
projector_specificity(dim)
  = saliency_forget(dim) / (saliency_retain(dim) + eps)
```

或：

```text
projector_specificity(dim)
  = saliency_forget(dim) - gamma * saliency_retain(dim)
```

其中：

```text
saliency_forget:
  forget_clean 上目标 identity answer logprob / name token logprob 对 projector dim 的梯度、Fisher、IG 或 activation delta

saliency_retain:
  hard_retain / counterfactual_retain 上相同 projector dim 的梯度、Fisher、IG 或 activation delta
```

候选 projector dim 选择：

```text
top-k dims by projector_specificity
```

然后构造 cross-modal path：

```text
vision high-saliency node
-> projector top-k specificity dim
-> LLM image-token node
-> LLM answer-token / name-token node
```

### 4.2 从 sample-level 绑定生成 cross-modal path

不要只全局生成少量 `vision_text` path。应按 forget sample 生成：

```text
each forget sample:
  top-k vision nodes
  top-k projector dims
  top-k LLM identity/name-token nodes
  build bounded number of cross-modal paths
```

并保留：

```json
{
  "sample_id": "...",
  "pair_id": "...",
  "source_row_idx": 0,
  "vision_path_id": "...",
  "text_path_id": "...",
  "projector_dim_score": 0.0
}
```

理由：

Survey 中 SalUn / SSD 一类方法的核心思想是：先找对 forget 样本 disproportionally important 的参数，再做选择性更新。当前 projector dim 不是随便选一个 bridge 节点，而应该是 forget identity 相关、retain 不敏感的 cross-modal alignment dimension。

## 5. 修正方案三：Step5 必须修复 sufficiency，不能让 Suf 全为 0 后继续分类

当前 Step6 公式是：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

但历史审计中已经出现：

```text
positive_Suf = 0 across text / vision / vision_text
```

如果 `Suf=0`，分类会退化成：

```text
forget_effect ~= max(0, Nec)
```

Nec 只能说明：

```text
ablate 这条 path 会影响 clean 输出
```

但不能说明：

```text
这条 path 足以承载目标身份信息
```

因此很多 shared reasoning path 可能被误选为 forget path。这样会出现当前现象：

```text
retain 没明显坏
forget 也没真正降
```

### 5.1 新的 P_forget eligibility

建议 `P_forget` 至少满足：

```text
Nec > tau_nec
Suf > tau_suf
Ret < tau_ret
num_patchable_nodes == num_nodes
```

对 `vision_text` path 额外要求：

```text
contains_projector = true
projector_patchable = true
```

如果 `Suf` 仍然长期为 0，则不应声明 path 是 causal forget path。

### 5.2 Cross-modal sufficiency 的 restoration 位置

对 `vision_text` path，restoration 不应只看 answer token MLP。应重点验证：

```text
projector output / image token hidden state
-> answer name logprob
```

可尝试的 restore token selector：

```text
image_tokens
last_prompt_token
answer_tokens
```

并分别记录：

```text
clean_score
corrupt_score
restored_score
Suf = restored_score - corrupt_score
```

验收标准：

```text
至少一部分 vision_text/projector path 出现 positive Suf，
且 restored_score 明显高于 corrupt_score。
```

理由：

跨模态身份知识往往不是纯语言层知识，而是图像特征经过 projector 对齐后触发语言空间中的身份 token。如果 projector restoration 不提升 corrupt prompt 下的 clean answer logprob，就不能证明这条 path 是目标跨模态身份通路。

## 6. 修正方案四：Step6 改成 modality-aware + specificity ranking

新实验中：

```text
total classified paths = 1368
text = 680
vision = 680
vision_text = 8
```

如果使用全局 quantile，`vision_text` path 很容易被数量巨大的 text / vision path 淹没。

### 6.1 分模态阈值

不要只用全局：

```text
forget_quantile = 0.75
retain_quantile = 0.75
```

建议改为按 modality 分桶：

```text
text:        quantile / top-k
vision:      quantile / top-k
vision_text: separate quantile / top-k
```

### 6.2 用 specificity 排名

建议增加：

```text
forget_specificity = forget_effect - gamma * retain_impact
```

或：

```text
forget_specificity = forget_effect / (retain_impact + eps)
```

`P_forget` 优先选择：

```text
forget_effect 高
retain_impact 低
Suf 为正
projector patchable
```

### 6.3 设置 cross-modal 最低覆盖

可以先采用工程约束：

```text
P_forget must include at least K validated vision_text/projector paths
```

如果没有满足，就不进入 Step7，回到 Step5/Step6 修正。

理由：

当前任务的核心是跨模态身份遗忘。若 `vision_text` 只占 8/1368，用全局阈值几乎无法保证 projector path 进入编辑集合。分桶和最低覆盖不是为了造假，而是为了避免候选空间规模不均导致关键模态系统性缺席。

## 7. 修正方案五：projector 编辑先用 whole-vector / adapter MVP，再做精细 neuron editing

当前 projector neuron 选择还不可靠。建议不要一开始就强行 sparse projector neuron editing。

### 7.1 第一阶段：projector whole-vector 或 low-rank adapter

对 validated projector paths，先使用：

```text
projector whole-vector activation objective
```

或：

```text
projector low-rank adapter editing
```

同时加 retain anchor preserve：

```text
hard_retain KL / CE
counterfactual_retain KL / CE
retain image-token projector activation preserve
```

### 7.2 第二阶段：hidden-dim sparse projector editing

当 projector dim saliency 变稳定后，再切换为：

```text
projector hidden-dim sparse editing
```

理由：

MLLM unlearning 的 projector 很敏感。SineProject / SMFA 一类 multimodal unlearning 思路强调：遗忘时要避免破坏 vision-language alignment。当前实验 retain 损伤很小，但 forget 不动，说明可以适度提高 projector 编辑强度，同时用 retain anchor 保护对齐。

## 8. 修正方案六：Step7 forget objective 从 random activation 改成身份输出目标

当前 Step8 失败点是：

```text
forget_clean name_hit_rate 没降，反而升高。
```

因此 forget objective 应直接压制身份输出，而不是只做 random-direction activation。

### 8.1 建议的 forget loss

```text
L_forget =
  NPO / DPO-style negative preference on target identity answer
  + target name token probability suppression
  + optional unknown/refusal target CE
```

其中 name token suppression 可定义为：

```text
L_name_suppress = mean(log p(target_name_tokens | prompt, image))
```

训练时最小化：

```text
- L_name_suppress
```

或使用 margin：

```text
max(0, log p(target_name) - log p(unknown/refusal) + margin)
```

### 8.2 建议的 retain loss

```text
L_retain =
  retain KL to frozen model
  + hard_retain CE/KL
  + counterfactual_retain CE/KL
  + P_shared activation preserve
  + projector activation preserve on retain image tokens
```

### 8.3 总 loss

```text
L_total =
  L_forget
  + alpha * L_retain
  + beta * L_shared_preserve
  + lambda_projector * L_projector_retain_preserve
```

理由：

NPO / negative preference optimization 的思想是把“不再偏好 forget answer”直接写进目标，而不是依赖普通 gradient ascent 或随机 activation 扰动。当前问题不是模型 collapse，而是 forget 太弱；因此需要更直接地压低 target identity answer。

## 9. 从 survey 可借鉴的思想

`awesome-machine-unlearning` 的价值不是提供一个可以直接替换当前流程的单一算法，而是提供若干方法类别。对当前问题最有用的是以下几类。

### 9.1 SalUn / gradient-based saliency

可借鉴点：

```text
用 forget 样本的梯度或 saliency 找真正需要更新的权重/神经元。
```

对应到当前项目：

```text
projector dim selection
LLM answer-token neuron selection
vision node selection
```

理由：

当前 projector neuron 是 placeholder 或固定值，缺乏 forget-specific 依据。Saliency 可以把 cross-modal path 从“结构上存在”变成“对目标身份输出有实际贡献”。

### 9.2 SSD / Fisher ratio

可借鉴点：

```text
比较 forget 与 retain 上的 parameter importance，
优先 dampen 对 forget 重要、对 retain 不重要的参数。
```

对应到当前项目：

```text
projector_specificity = Fisher_forget / (Fisher_retain + eps)
```

理由：

这正好对应 `P_forget` 的科学定义：对 forget 高影响，对 retain 低影响。

### 9.3 NPO / negative preference optimization

可借鉴点：

```text
把 forget answer 作为 negative preference，
降低模型对目标答案的偏好。
```

对应到当前项目：

```text
target name token probability suppression
identity answer negative preference
unknown/refusal alternative preference
```

理由：

Step8 评估的是 name hit rate。优化目标应直接对齐这个指标。

### 9.4 SCRUB / distillation-style retain preserve

可借鉴点：

```text
forget 侧拉开，retain 侧蒸馏回 frozen model。
```

对应到当前项目：

```text
retain KL to frozen model
hard_retain / counterfactual_retain preserve
P_shared activation preserve
```

理由：

projector 编辑容易破坏跨模态对齐，因此必须配 retain distillation。

### 9.5 Multimodal unlearning: projector / alignment stability

可借鉴点：

```text
MLLM unlearning 不能只看 language model 层；
projector 和 vision-language alignment 是关键风险点。
```

对应到当前项目：

```text
projector patching
projector editing
retain image-token activation preserve
cross-modal pair screen
```

理由：

当前实验已经证明：只编辑 LLM/vision `down_proj`，并不能可靠降低跨模态身份命中。

## 10. 推荐实施优先级

### P0: 增加 projector/cross-modal invariant

必须先让以下指标可见：

```text
Step5 vision_text full patchable ratio
Step6 P_forget projector count
Step7 num_projector_modules
```

### P1: 修复 Step5 sufficiency

目标：

```text
positive Suf > 0 for some validated vision_text/projector paths
```

否则 Step6 分类没有足够因果含义。

### P2: 用 saliency ratio 替换 projector placeholder

目标：

```text
projector dim 来自 forget/retain specificity，而不是固定 neuron=0。
```

### P3: Step6 改为 modality-aware specificity ranking

目标：

```text
validated projector paths can enter P_forget/P_shared。
```

### P4: Step7 增加 projector whole-vector / adapter MVP

目标：

```text
num_projector_modules > 0
projector parameters receive gradient
retain anchor protects alignment
```

### P5: Step7 forget objective 改为 name-token / NPO objective

目标：

```text
forget_clean name_hit_rate < PEFT baseline 0.611111
```

### P6: 快速 Step8 pair screen 后再跑 Full CLEAR

只有当 pair screen 先过：

```text
forget_clean name_hit_rate 下降
hard_retain / counterfactual_retain 不明显下降
```

再跑 Full CLEAR remote scoring，避免远程评分成本浪费在明显失败的候选模型上。

## 11. 最终判断

当前实验不是证明 causal selective editing 路线失败，而是暴露出三个具体缺口：

```text
1. cross-modal / projector path 选择不准
2. Step5 sufficiency 证据不足
3. Step7 forget objective 与 name_hit_rate 指标不直接对齐
```

最值得优先修的是：

```text
projector saliency ratio
vision_text sufficiency restoration
modality-aware Step6
name-token / NPO forget objective
retain anchor preserve
```

这些修正都能从 machine unlearning survey 中找到对应思想，但应作为当前 causal path pipeline 的局部增强，而不是直接替换整个 CHIP-Editor 路线。
