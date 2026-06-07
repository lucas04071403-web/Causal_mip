# Causal MIP 项目阶段性总结合辑

更新时间：2026-06-07

本文基于当前 `MIP-Editor` 工作区、已有阶段性成功/失败记录，以及最新 H5 Full CLEAR / 更大 split 评估结果，对当前项目做一次重新整理。

当前最重要的结论是：

```text
H5 是当前项目中第一个同时通过 Step8 cheap gate、在更大 split 上保持 CE 增益稳定、
并在 Full CLEAR 远程评估中与 baseline 基本持平且 forget 小幅改善的候选。
```

但也需要明确：

```text
这不是“大幅碾压 baseline”的结果。
这次成功更准确地说是：在 retain 几乎不掉的前提下，取得了可复现的小幅 forget 增强。
```

---

## 1. 当前项目一句话总结

本项目是在 `MIP-Editor` 上扩展的一条因果选择性编辑路线，目标是在多模态大模型 `Qwen2.5-VL-3B-Instruct` 上实现更精细的 PII / target-name unlearning。

核心思想不是全模型粗暴遗忘，而是：

```text
先定位与目标姓名强相关的 causal path / projector dim，
再只在这些局部子空间上做 masked RMisU 编辑，
同时用 hard retain / same_topic / same_reasoning / counterfactual retain 约束副作用。
```

当前项目已经从早期的“工程链路是否能跑通”推进到：

```text
如何在有限 projector-dim 子空间内取得更好的 forget-retain trade-off。
```

---

## 2. 当前代码和实验位置

代码库：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
```

主实验 workspace：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

当前主线代码目录：

```text
MIP-Editor/causal_mip/
```

当前最重要的文档：

```text
MIP-Editor/README.md
MIP-Editor/docs/PROJECT_STRUCTURE.md
MIP-Editor/docs/CAUSAL_MIP_PROJECT_SUMMARY_0607.md
```

当前最重要的 H5 checkpoint：

```text
../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603/
```

---

## 3. 当前成功候选：H5

H5 的配置可以概括为：

```text
H5 = H1 retain-safe base + p000006 top15 小幅回加
objective = pii_name_token_noise
forget_ce_alpha = 0.40
pii_noise_alpha = 0.08
max_steps = 200
projector_edit_mode = qwen_merger_mlp
target_ce_scope = name
```

H5 的设计来源是：

```text
1. Step14 top16 证明扩大 projector-dim 子空间能提高 forget CE。
2. H1 证明替换高 retain 风险 path 的部分 dims 可以修复 retain 边界。
3. H3 证明 p000006 不能直接删除，因为它承担关键 forget CE drive。
4. H4/H5 证明对 p000006 做小幅 top14/top15 回加，比纯 SSD 扩容更稳。
```

最终 H5 选择：

```text
以 H1 作为 retain-safe base，
对 p000006 做 top15 小幅回加，
不继续做纯 SSD 扩容。
```

这一步非常关键，因为它说明当前有效路线不是“无脑扩大可编辑维度”，而是：

```text
保留 CE-driving core，
移除或替换 retain-risk dim，
再对关键 path 做小剂量回加。
```

---

## 4. 当前 H5 的核心实验结果

### 4.1 Step8 val cheap gate

H5 首先在 val cheap gate 上通过。

| gate 项 | 阈值 | H5 结果 | 判定 |
| --- | ---: | ---: | --- |
| forget_clean target_name_ce delta | > +0.01 | +0.011417 | PASS |
| forget_clean name_hit | <= 0.6111 | 0.5556 | PASS |
| hard_retain name_hit | >= 0.6444 | 0.6944 | PASS |
| counterfactual_retain name_hit | >= 0.6722 | 0.7222 | PASS |

这说明：

```text
H5 第一次在本地 cheap gate 上同时满足 forget 概率侧、forget 生成侧和 retain 生成侧条件。
```

### 4.2 更大 split 概率诊断

随后在更大 train split 上复测 probability diagnostic。

| 分组 | 指标 | delta | 解释 |
| --- | --- | ---: | --- |
| forget_clean | target_name_ce | +0.010743 | 目标姓名 CE 上升，遗忘方向稳定 |
| forget_clean | target_name_logprob | -0.010743 | 目标姓名 logprob 下降 |
| hard_retain::same_reasoning | target_name_ce | +0.000426 | 极小变化，retain 风险很低 |
| hard_retain::same_topic | target_answer_ce | -0.000108 | 没有扩大 same_topic 损失 |
| counterfactual_retain | target_name_ce | -0.000462 | counterfactual 概率侧略好 |
| counterfactual_retain | target_answer_ce | -0.000730 | counterfactual answer CE 略好 |

核心判断：

```text
之前 val 上的 +0.011417 CE 增益不是偶然。
在更大 split 上仍能保持约 +0.0107 的 target-name CE 增益。
```

### 4.3 更大 split generation 对比

与 PEFT baseline 相比：

| eval set | baseline name_hit | H5 name_hit | delta | 解释 |
| --- | ---: | ---: | ---: | --- |
| forget_clean | 0.658824 | 0.629412 | -0.029412 | forget 目标姓名命中下降，正向 |
| hard_retain | 0.602941 | 0.602941 | 0.000000 | hard retain 未下降 |
| counterfactual_retain | 0.588235 | 0.570588 | -0.017647 | counterfactual 有小幅生成侧损失 |
| hard_retain::same_reasoning | 0.600000 | 0.600000 | 0.000000 | same_reasoning 稳定 |
| hard_retain::same_topic | 0.605882 | 0.605882 | 0.000000 | same_topic 稳定 |

BLEU / ROUGE-L 侧：

| eval set | BLEU delta | ROUGE-L delta | 判断 |
| --- | ---: | ---: | --- |
| forget_clean | +0.002605 | +0.001723 | 文本重叠没有下降，但 name_hit 已下降 |
| hard_retain | -0.000583 | -0.002653 | 可忽略小幅波动 |
| counterfactual_retain | -0.010329 | -0.010201 | 有小幅损失，需要下一步关注 |

核心判断：

```text
H5 在更大 split 上确实压低了 forget 目标姓名命中。
hard retain / same_topic 没有明显损失。
counterfactual retain 在概率侧稳定，但 generation 表面指标有小幅下降。
```

---

## 5. Full CLEAR 远程评估结果

最新完整 Full CLEAR 结果文件：

```text
../mip_workspace/outputs/full_clear_remote_eval/hybrid_h5_top16_p000006_to_top15_full_remote_full_0607/full_clear_remote_protocol_summary.json
```

baseline 结果文件：

```text
../mip_workspace/outputs/full_clear_remote_eval/peft_baseline_full_remote_step9_bound_projector_0526_214028/full_clear_remote_protocol_summary.json
```

Full CLEAR 任务规模：

| task | split | examples | scored |
| --- | --- | ---: | ---: |
| classification | forget | 188 | 188 |
| classification | retain | 3580 | 3580 |
| generation | forget | 188 | 188 |
| generation | retain | 3580 | 3580 |

Full CLEAR 指标对比：

| 指标 | baseline | H5 | delta | 方向 |
| --- | ---: | ---: | ---: | --- |
| forget_classification_remote_acc | 0.686170 | 0.680851 | -0.005319 | forget 小幅更好 |
| forget_generation_remote_acc | 0.462766 | 0.446809 | -0.015957 | forget 更好 |
| forget_generation_bleu | 0.236700 | 0.228762 | -0.007938 | forget 文本重叠下降 |
| forget_generation_rougeL | 0.425851 | 0.419917 | -0.005934 | forget 文本重叠下降 |
| retain_classification_remote_acc | 0.703631 | 0.705587 | +0.001955 | retain 略好 |
| retain_generation_remote_acc | 0.432123 | 0.431564 | -0.000559 | 基本持平 |
| retain_generation_bleu | 0.239301 | 0.239647 | +0.000345 | 基本持平 |
| retain_generation_rougeL | 0.423476 | 0.424192 | +0.000716 | 基本持平 |

Full CLEAR 结论：

```text
H5 与 baseline 的整体结果非常接近。
retain 基本不掉，classification 还略升。
forget classification / generation 都朝遗忘方向小幅下降。
因此 H5 是当前有效的 trade-off 改进。
```

需要避免过度解读：

```text
H5 不是大幅超过 baseline。
H5 的价值在于保住 retain 的同时，稳定获得小幅 forget 改善。
```

---

## 6. 当前项目已经突破的瓶颈

### 6.1 突破了 projector path 找不到的问题

早期失败中，projector path 虽然存在，但 Step6 answer-level guard 会把它们归到 `P_shared`，导致真正身份相关的 cross-modal path 不能进入编辑。

后续通过 target-name diagnostic 证明：

```text
target-name sensitive path 主要集中在 projector / vision_text path。
```

这说明身份遗忘任务不能只看 answer-level causal score，必须看：

```text
NameNec
NameSuf
NameRet
target_name_ce
target_name_logprob
```

### 6.2 突破了 Step6 分类误杀 projector 的问题

旧分类方式：

```text
projector path retain risk 高 -> 进入 P_shared -> 不编辑
```

修正后：

```text
如果 path 对 target-name 有强因果作用，即使 answer-level retain risk 高，
也应进入 P_forget 候选，再由 retain-aware dim 选择和 loss 控制副作用。
```

这让项目从粗粒度 path 分类推进到 name-aware path selection。

### 6.3 突破了 projector whole-vector 编辑过粗的问题

whole-vector projector 编辑会带来较大 retain 风险。

当前主线已经改成：

```text
只编辑 visual.merger.mlp.0 中少量具体 projector dims。
```

这带来两个好处：

```text
1. 编辑强度可以精确控制。
2. 不需要把整个 projector 作为可编辑对象，从而降低 retain 损伤。
```

### 6.4 突破了 name-token objective 不对齐的问题

当前 `forget_ce_scope = name` 时，只对 target name token 做 CE ascent，不再 fallback 到整句 answer token。

这解决了一个关键问题：

```text
模型可能仍然回答相似描述，但目标姓名概率必须下降。
```

因此当前评估优先级已经更清楚：

```text
target_name_ce / name_hit
> Full CLEAR remote acc
> BLEU / ROUGE-L
```

### 6.5 突破了 cheap gate 无法筛选的问题

项目已经形成低成本筛选闭环：

```text
Step8 probability diagnostic
+ Step8 generation eval
+ Full CLEAR remote eval
```

cheap gate 的价值已经被多次验证：

```text
失败 run 不必直接跑 Full CLEAR；
只有 Step8 同时满足 forget CE、forget name_hit、retain name_hit 后，才值得扩大评估。
```

### 6.6 突破了 top16 retain 边界问题

Step14 top16 提高了 forget CE，但 counterfactual retain 边界不稳。

H1/H4/H5 的实验说明：

```text
不能直接删除 p000006；
也不能继续纯 SSD 扩容；
应该保留 H1 retain-safe base，并对 p000006 小幅回加。
```

最终 H5 证明：

```text
top15 小幅回加足以把 forget CE 推过 +0.01，
同时 hard retain / counterfactual retain 在 val cheap gate 上保持稳定。
```

---

## 7. 当前失败经验和被否定的路线

### 7.1 只看 answer-level causal score 不够

失败原因：

```text
answer-level score 会把身份相关 projector path 误认为高 retain 风险，
从而排除真正需要编辑的路径。
```

经验：

```text
身份遗忘必须引入 target-name 粒度 score。
```

### 7.2 whole-vector projector 编辑风险过高

失败原因：

```text
whole-vector projector 可编辑神经元过多，
容易造成 retain / counterfactual retain 副作用。
```

经验：

```text
当前主线应保持 projector-dim 级编辑。
```

### 7.3 redacted full-answer positive 不稳定

失败原因：

```text
全答案 redacted positive token 太多，
容易错位并抵消 target-name CE ascent。
```

经验：

```text
如果使用 redacted positive，应优先只在 name span 上做约束。
```

### 7.4 纯 top20/top24 SalUn/SSD 替换失败

失败原因：

```text
SalUn/SSD 式重排作为“替换旧 top16 子空间”的策略时，
没有提高 forget_clean target_name_ce delta，甚至让概率侧接近 0 或变负。
```

经验：

```text
SSD 不能作为主扩容策略直接替换 CE-driving core。
它更适合作为 H5 之后的小剂量补剂，而且必须受 causal/name score 约束。
```

### 7.5 直接删除 p000006 失败

H3 证明：

```text
p000006 虽然有 retain 风险，但也承担关键 forget CE drive。
直接删除会让 forget CE 大幅回落。
```

经验：

```text
对关键 path 应做小幅回加或局部替换，而不是直接删除。
```

---

## 8. 当前项目的真实状态判断

当前可以宣称：

```text
1. causal_mip 工程闭环已经跑通。
2. target-name sensitive projector path 可以被定位。
3. projector-dim 编辑可以稳定执行。
4. H5 已通过 Step8 cheap gate。
5. H5 在更大 split 上保持约 +0.0107 的 target-name CE 增益。
6. H5 在 Full CLEAR 上与 baseline 基本持平，并在 forget 指标上有小幅改善。
7. retain classification / generation 没有明显损失。
```

当前不能过度宣称：

```text
1. 不能说 H5 大幅超过 baseline。
2. 不能说所有 counterfactual retain 生成行为都完全无损。
3. 不能说当前已经达到最终最优 unlearning。
4. 不能说 SSD 扩容已经被证明有效；目前有效的是 H5 的小幅回加策略。
```

最准确的项目状态：

```text
当前项目已经找到一条可工作的 causal selective editing 路线。
H5 是第一个在本地 gate、larger split 和 Full CLEAR 上都站得住的候选。
下一阶段目标不是重新找路线，而是在 H5 基础上修复 counterfactual generation 小幅损失，
并继续放大 forget 效果。
```

---

## 9. 当前推荐的下一步路线

### 9.1 不建议做的事

当前不建议：

```text
1. 不建议继续纯 SSD 扩容。
2. 不建议直接扩大 top20/top24 替换 H5 子空间。
3. 不建议重新回到 whole-vector projector。
4. 不建议只看 Full CLEAR acc 而忽略 target_name_ce。
5. 不建议在 cheap gate 未过的 run 上反复跑完整 Full CLEAR。
```

理由：

```text
H5 已经证明主要瓶颈不是“维度越多越好”，而是“CE-driving dim 和 retain-risk dim 的边界选择”。
```

### 9.2 建议优先做 H5 + counterfactual retain anchor

下一步最合理的实验是：

```text
H5 base
+ 小剂量 counterfactual retain anchor
+ 保持 forget CE drive
```

目标：

```text
1. 保持 forget_clean target_name_ce_delta >= +0.010。
2. 保持 Full CLEAR forget generation acc 低于 baseline。
3. 把 counterfactual_retain generation 的 name_hit / BLEU / ROUGE-L 小幅损失拉回。
4. 不损伤 hard_retain::same_topic。
```

原因：

```text
当前 H5 的主要问题不是 forget 不够，而是 counterfactual generation 有轻微表面指标下降。
因此下一步应该补 retain anchor，而不是继续增强 forget。
```

### 9.3 如果 H5 + retain anchor 仍不够，再做受约束 SSD 小剂量补充

备用路线：

```text
H5 base
+ causal/name score 约束的 SSD 小剂量 dims
+ 严格 retain guard
```

SSD 补充必须满足：

```text
1. 不替换 H5 CE-driving core。
2. 只追加少量 high target-name effect / low retain impact dims。
3. 每次追加后先跑 Step8 probability diagnostic。
4. cheap gate 不过，不进入 Full CLEAR。
```

---

## 10. 当前推荐验收标准

下一轮实验建议继续使用三层验收。

第一层：Step8 probability diagnostic

```text
forget_clean target_name_ce_delta >= +0.010
forget_clean target_name_logprob_delta <= -0.010
hard_retain / counterfactual_retain target CE 不明显恶化
same_topic target_answer_ce 不上升
```

第二层：Step8 generation eval

```text
forget_clean name_hit 不高于 baseline
hard_retain name_hit 不低于 baseline - 0.05
counterfactual_retain name_hit 不低于 baseline - 0.05
same_topic / same_reasoning 不出现新增明显损失
```

第三层：Full CLEAR remote eval

```text
forget classification / generation acc 相对 baseline 下降
retain classification / generation acc 基本持平
retain BLEU / ROUGE-L 不明显下降
```

当前 H5 在第三层的表现是：

```text
forget 小幅改善；
retain 基本持平；
因此可作为下一轮优化的 base。
```

---

## 11. 当前项目关键词

当前项目的技术关键词可以整理为：

```text
Causal MIP
multimodal unlearning
Qwen2.5-VL
target-name token alignment
name-aware path selection
projector-dim localization
masked RMisU
pii_name_token_noise
retain-safe hybrid candidate
H5 top15 p000006 add-back
Step8 cheap gate
Full CLEAR remote evaluation
```

---

## 12. 最终阶段性结论

H5 这次实验是成功的。

成功不是因为它大幅超过 baseline，而是因为它第一次同时满足了以下条件：

```text
1. val cheap gate 通过；
2. 更大 split 上 target-name CE 增益稳定；
3. forget generation name_hit 相对 baseline 下降；
4. hard retain / same_topic 没有扩大损失；
5. Full CLEAR 上 forget 指标小幅改善；
6. Full CLEAR 上 retain 基本与 baseline 持平。
```

因此当前项目的主线应从：

```text
继续寻找是否存在有效 causal editing 路线
```

转为：

```text
以 H5 为 base，围绕 counterfactual retain anchor 和受约束小剂量补充继续优化 trade-off。
```
