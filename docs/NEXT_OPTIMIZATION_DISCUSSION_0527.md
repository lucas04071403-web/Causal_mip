# CHIP-Editor 下一步优化讨论：基于 `step9_bound_projector_0526_214028` 复跑结果

更新时间：2026-05-27

本文基于当前复跑结果、`CHIP_EDITOR_TECHNICAL_ROADMAP.md` 以及近期 Step7 probe 实验，整理当前存在的问题，并讨论下一步优化应该先围绕科学问题 1、科学问题 2、科学问题 3 逐步推进，还是做总体优化。

结论先行：

```text
当前不建议直接做总体超参优化。
主线应先修科学问题 1 的因果路径识别与分类，
再专门补强科学问题 2 的跨模态/projector 路径定位，
最后再系统优化科学问题 3 的选择性编辑目标函数。
```

原因是：当前失败的核心不是训练强度不足，而是“编辑对象不够可信”。如果没有先证明哪些路径是真正承载目标身份知识的因果路径，继续调 Step7 loss、probe 强度或训练轮数，很可能只是更强地编辑错误路径。

---

## 1. 本次实验复跑结论

复跑产物：

```text
mip_workspace/outputs/step8_eval_step9_bound_projector_0526_214028_rerun_0527_val.json
mip_workspace/outputs/full_clear_step9_bound_projector_0526_214028_rerun_0527.json
mip_workspace/outputs/step8_protocol_step9_bound_projector_0526_214028_rerun_0527_fullrerun.json
```

固定 Step8 主协议结果：

```text
status = do_not_claim_success
can_claim_success = false
failed_reasons = ["pair_screen:pair_forget_clean_name_hit_rate"]
```

关键指标：

| 指标 | candidate | baseline | delta | 方向 |
| --- | ---: | ---: | ---: | --- |
| Step8 forget_clean name_hit_rate | 0.7222 | 0.6111 | +0.1111 | 失败，越低越好 |
| Step8 hard_retain name_hit_rate | 0.7222 | 0.6944 | +0.0278 | 通过，越高越好 |
| Step8 counterfactual_retain name_hit_rate | 0.7778 | 0.7222 | +0.0556 | 通过，越高越好 |
| Full CLEAR forget clf remote acc | 0.6755 | 0.6862 | -0.0106 | 形式通过，但很弱 |
| Full CLEAR forget gen remote acc | 0.4574 | 0.4628 | -0.0053 | 形式通过，但很弱 |
| Full CLEAR retain clf remote acc | 0.7034 | 0.7036 | -0.0003 | retain 基本稳定 |
| Full CLEAR retain gen remote acc | 0.4293 | 0.4321 | -0.0028 | retain 基本稳定 |

解释：

- Full CLEAR main 在当前宽松门槛下通过。
- 但 forget 侧下降非常小，只是约 1.06 pp 和 0.53 pp。
- Step8 pair screen 直接失败，因为 `forget_clean name_hit_rate` 高于 PEFT baseline。
- 因此不能声明成功遗忘。

---

## 2. 当前存在的主要问题

## 2.1 Step5 sufficiency 失效

本轮 Step5 统计：

```text
records = 2720
ok = 2720
positive Suf = 0
Suf 全部为 0
```

这意味着 Step6 中：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
```

实际退化为：

```text
forget_effect ~= max(0, Nec)
```

问题在于：

- `Nec` 只能说明 ablate 某条 path 后 clean 输出发生变化。
- 它不能证明这条 path 足以承载目标身份知识。
- 如果没有 positive `Suf`，就不能严谨地说该路径是目标知识的充分因果路径。

这会导致 Step6 误把 shared reasoning path 或 benign topic path 当成 forget path。

当前现象也符合这个判断：

```text
retain 没明显坏，
forget 也没有真正降。
```

这说明当前编辑可能没有命中真正的目标身份因果路径。

## 2.2 projector / vision_text 没进入真正 P_forget 编辑集合

Step6 分类统计：

```text
P_forget = 100
P_shared = 110
P_retain = 221
P_irrelevant = 937

text = 680
vision = 680
vision_text = 8
```

更关键的是：

```text
P_forget 中没有 vision_text path
vision_text path 全部进入 P_shared
Step7 最终 projector modules = 0
```

也就是说，虽然 run id 中包含 `bound_projector`，代码也支持 projector 编辑，但本轮真正被编辑的仍然主要是 LLM/vision `down_proj`，不是 projector/cross-modal identity path。

这使得实验名和实际编辑对象不一致：

```text
名义上是 projector 路线；
实际没有编辑 projector。
```

## 2.3 Step8 pair screen 否定了目标遗忘

复跑结果：

```text
candidate forget_clean name_hit_rate = 0.7222
baseline forget_clean name_hit_rate  = 0.6111
delta = +0.1111
```

目标是降低 name hit rate，但候选模型反而更容易命中目标姓名。

因此当前问题不是“遗忘不够强”，而是“遗忘方向没有对齐”。

## 2.4 Step7 objective 不够对齐，但不是第一优先级

近期 Step7 probe 实验证明：

- projector path 可以进入 Step7 编辑集合。
- `visual.merger.mlp.0` 可以作为 Qwen projector 编辑模块。
- name token scoped CE 可以正确定位姓名 token。

但实验结果也说明：

- 单纯增强 projector probe 无效。
- `activation_random + name CE` 无效。
- 单纯 `name_ce_ascent` 仍不能优于 baseline。
- 继续加大 `probe_beta`、训练步数、`name_ce_alpha` 风险很大。

因此 Step7 objective 确实需要改，但它不是当前第一断点。

当前第一断点是：

```text
Step5/Step6 还没有稳定证明哪些 path 是真正 target identity causal path。
```

---

## 3. 三个科学问题的推进关系

`CHIP_EDITOR_TECHNICAL_ROADMAP.md` 中三个科学问题是：

1. 在多模态大模型中，哪些路径是真正承载目标知识的因果路径，如何区分“有害意图路径”和“良性主题路径”？
2. 如何定位跨模态因果路径中的有害知识？
3. 如何在删除目标知识时避免破坏 retain knowledge？

这三个问题不应并列推进，而应按因果依赖关系推进。

推荐顺序：

```text
科学问题 1：先证明哪些路径是因果路径，并区分 P_forget / P_shared
        ↓
科学问题 2：在问题 1 框架内专门强化 cross-modal / projector path
        ↓
科学问题 3：在可信 P_forget / P_shared 上优化 selective editing
```

原因：

- 问题 3 依赖问题 1 的路径分类结果。
- 问题 2 是问题 1 在跨模态路径上的特化。
- 如果问题 1 没解决，问题 3 的编辑目标再强，也可能编辑错路径。

---

## 4. 第一阶段：优先修科学问题 1

科学问题 1 的目标是：

```text
证明哪些 path 是目标知识的因果路径，
并区分 harmful identity path 和 benign/shared topic path。
```

当前要优先修的是 Step5/Step6。

## 4.1 修复 Step5 sufficiency

当前 `Suf=0` 不能继续接受为正常结果。

需要检查：

```text
clean_score
corrupt_score
restored_score
target_answer_text
answer_token_positions
restore token positions
resolved_nodes
skipped_nodes
```

新的最低验收：

```text
至少一部分 path 出现 positive Suf
restored_score > corrupt_score
Suf 诊断字段能解释 restoration 是否真的发生
```

如果 `positive Suf = 0`，则不能进入正式 Step6/Step7 主实验。

## 4.2 改造 P_forget eligibility

当前 `P_forget` 不应只依赖 `Nec`。

建议新的基本条件：

```text
Nec > tau_nec
Suf > tau_suf
Ret < tau_ret
num_patchable_nodes == num_nodes
```

也就是说：

- 必要性高。
- 充分性为正。
- retain impact 低。
- path 节点可以完整 patch。

只有满足这些条件，才有资格进入 `P_forget`。

## 4.3 明确 P_shared 的科学定义

`P_shared` 应该是：

```text
forget_effect 高
retain_impact 也高
```

它不是“编辑失败集合”，而是：

```text
与目标知识相关，但也承载 retain/shared 能力的路径。
```

因此 `P_shared` 应该保护，而不是直接编辑。

## 4.4 第一阶段验收标准

第一阶段不以 Step8 成功为唯一目标，而以因果路径识别是否可信为目标。

建议验收：

```text
Step5:
  positive Suf > 0
  clean/corrupt/restored 三个 score 有可解释差异
  resolved/skipped node 诊断完整

Step6:
  P_forget 中每条 path 都满足 Nec 高、Suf 正、Ret 低
  P_shared 能解释为 forget 高且 retain 高
  没有 positive Suf 时禁止输出正式 P_forget
```

---

## 5. 第二阶段：再补科学问题 2

科学问题 2 是：

```text
如何定位跨模态因果路径中的有害知识？
```

它应在科学问题 1 的因果验证框架内推进，而不是单独绕过 Step5/Step6。

## 5.1 重做 projector / cross-modal candidate

当前 projector dim 选择不够可靠，不能继续依赖 placeholder 或固定 neuron。

建议为 projector hidden dimension 计算：

```text
projector_specificity(dim)
  = saliency_forget(dim) - gamma * saliency_retain(dim)
```

或：

```text
projector_specificity(dim)
  = saliency_forget(dim) / (saliency_retain(dim) + eps)
```

其中：

```text
saliency_forget:
  forget_clean 上 target identity/name token logprob 对 projector dim 的梯度/Fisher/IG

saliency_retain:
  hard_retain/counterfactual_retain 上相同 dim 的梯度/Fisher/IG
```

目标是找到：

```text
对 forget identity 高影响、对 retain 低影响的 projector dim。
```

## 5.2 sample-level 生成 cross-modal path

不要只全局生成少量 `vision_text` path。

应按 forget sample 绑定生成：

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
  "vision_path_id": "...",
  "projector_dim_score": 0.0,
  "text_path_id": "..."
}
```

## 5.3 Step6 改成 modality-aware

当前全局 quantile 会让 `vision_text` 被 text/vision 大量路径淹没。

建议按 modality 分桶：

```text
text: separate threshold/top-k
vision: separate threshold/top-k
vision_text: separate threshold/top-k
```

同时增加 cross-modal 最低覆盖：

```text
validated vision_text/projector P_forget paths >= K
```

如果没有满足，不进入正式 Step7。

## 5.4 第二阶段验收标准

```text
vision_text path 数量不再只有个位数
vision_text/projector path 至少部分出现 positive Suf
P_forget 中包含 validated projector/cross-modal path
Step7 summary 中 num_projector_modules > 0
```

---

## 6. 第三阶段：最后优化科学问题 3

科学问题 3 是：

```text
如何在删除目标知识时避免破坏 retain knowledge？
```

它依赖：

```text
可信的 P_forget
可信的 P_shared
validated cross-modal/projector path
```

在这些条件满足前，不建议把主要精力放在 Step7 超参上。

## 6.1 Step7 forget objective 需要换方向

当前不建议继续：

```text
单纯加大 probe_beta
单纯加大 probe_steering_coeff
单纯延长 activation_random 训练
单纯提高 name_ce_alpha
```

更合理的方向是：

```text
NPO / negative preference objective
replacement / refusal objective
KL-constrained preference editing
```

建议 forget loss：

```text
L_forget =
  NPO / DPO-style negative preference on target identity answer
  + target name token probability suppression
  + optional unknown/refusal target CE
```

例如 margin 形式：

```text
max(0, log p(target_name) - log p(unknown/refusal) + margin)
```

## 6.2 retain preserve 需要更强

建议 retain loss：

```text
L_retain =
  retain KL to frozen model
  + hard_retain CE/KL
  + counterfactual_retain CE/KL
  + P_shared activation preserve
  + projector activation preserve on retain image tokens
```

总 loss：

```text
L_total =
  L_forget
  + alpha * L_retain
  + beta * L_shared_preserve
  + lambda_projector * L_projector_retain_preserve
```

## 6.3 第三阶段验收标准

```text
Step8 pair screen:
  forget_clean name_hit_rate < PEFT baseline
  hard_retain name_hit_rate >= baseline - 0.05
  counterfactual_retain name_hit_rate >= baseline - 0.05

Full CLEAR:
  forget remote acc 低于 baseline
  retain remote acc 下降不超过 0.05

总协议:
  step8_protocol can_claim_success = true
```

---

## 7. 推荐的下一步执行顺序

## 7.1 不建议做的事

短期不建议继续：

```text
直接加大 Step7 训练轮数
直接加大 activation random 强度
直接加大 projector probe 强度
直接提高 name_ce_alpha
直接跑 Full CLEAR 大评测筛选大量 Step7 超参
```

这些操作会消耗算力，但很可能不能解决根因。

## 7.2 建议优先做的事

建议下一轮按以下顺序推进：

1. 修 Step5 sufficiency 诊断，让 `Suf` 不再全为 0。
2. 增加 Step5 resolved/skipped/projector patching 诊断字段。
3. 改 Step6 eligibility：`P_forget` 必须要求 positive Suf。
4. 改 Step6 modality-aware ranking，给 `vision_text/projector` 单独通道。
5. 重新生成 projector saliency-based cross-modal candidates。
6. 只有当 validated projector/cross-modal path 进入 `P_forget` 后，再跑 Step7。
7. Step7 目标函数切换到 NPO/replacement-style，而不是 activation random。

---

## 8. 最终判断

当前阶段最合理的判断是：

```text
工程闭环已经打通；
Full CLEAR 主评测在宽松阈值下可通过；
但科学闭环没有打通。
```

核心原因：

```text
当前还没有可靠证明被编辑的路径是真正 target identity causal path。
```

因此下一步不是做总体优化，而是按科学问题的依赖关系推进：

```text
先完成科学问题 1：
  修复 Nec/Suf/Ret 因果验证和 P_forget/P_shared 分类。

再完成科学问题 2：
  用 projector saliency 和 modality-aware ranking 定位跨模态有害路径。

最后推进科学问题 3：
  在可信路径集合上做 NPO/replacement-style selective editing。
```

一句话总结：

```text
先找对路径，再编辑路径；
不要在路径证据不足时优化编辑强度。
```
