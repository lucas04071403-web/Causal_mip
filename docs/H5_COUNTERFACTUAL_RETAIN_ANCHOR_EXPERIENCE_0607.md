# H5 Counterfactual Retain Anchor 成功经验记录 0607

## 1. 实验目标

本次目标是在 H5 checkpoint 上继续做一个小剂量修复实验：

- 保持 forget 强度：`forget_clean target_name_ce delta >= +0.010`
- 优先修复 H5 在 `counterfactual_retain` generation 侧的轻微损失
- 控制 retain 副作用，特别是 `hard_retain` name-hit 与 probability CE 风险

最终候选：

```text
masked_rmisu_h5_cf_anchor_name_a002_80steps_0607
```

结论：该候选满足 forget CE 硬门槛，并对 counterfactual generation 做到了稳定小幅修复。

## 2. 核心经验

H5 本身已经在 forget CE 上接近可用边界，但 counterfactual generation 有轻微损失。直接增强 retain 或减弱 forget 目标容易损害 forget CE，因此本次采用更局部的办法：

```text
在 H5 base 上加入小剂量 counterfactual retain anchor CE
```

具体做法是只从 Step3 pair JSONL 中抽取 `counterfactual_retain` 样本，对其目标 token 加正向 CE 约束：

```text
loss += counterfactual_anchor_alpha * CE(counterfactual target)
```

同时保留 H5 的 forget objective：

```text
loss -= forget_ce_alpha * forget_ce_loss
loss -= pii_noise_alpha * pii_noise_loss
```

这相当于给 counterfactual retain 子分布加一个很轻的 anchor，避免继续训练时把反事实生成能力一起拉坏。

## 3. 实现摘要

主要改动：

- `causal_mip/editing/masked_rmisu.py`
  - 新增 `counterfactual_anchor_alpha`
  - 新增 `counterfactual_anchor_scope`
  - 新增可选 `counterfactual_anchor_loader`
  - 当 `alpha != 0` 时，循环读取 anchor batch 并加入正向 CE
  - 记录 `counterfactual_anchor_loss` 和 token count

- `causal_mip/data_pairs/pair_sample_dataset.py`
  - 新增 `PairSampleDataset`
  - 从 Step3 pair JSONL 中暴露 `counterfactual_retain` 样本
  - 复用现有 CLEAR collate 流程

- `main.py`
  - 新增 anchor CLI 参数
  - 支持从 H5 full checkpoint 继续训练：`--initial_full_checkpoint`

- `load_model.py`
  - 新增 full checkpoint loader

- `causal_mip/project_paths.py`
  - 新增 workspace dataset path relocation
  - 解决旧 JSONL 中 stale absolute dataset path 的问题

默认行为保持兼容：

```text
counterfactual_anchor_alpha = 0.0
```

因此不显式开启 anchor 时，原 H5 masked RMisU 行为不变。

## 4. 成功配置

H5 base：

```text
../mip_workspace/outputs/checkpoints/masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603
```

关键参数：

```text
counterfactual_anchor_alpha = 0.02
counterfactual_anchor_scope = name
max_steps = 80
forget_objective = pii_name_token_noise
forget_ce_alpha = 0.40
pii_noise_alpha = 0.08
projector_edit_mode = qwen_merger_mlp
```

路径配置：

```text
P_forget:
../mip_workspace/outputs/paths/step6_hybrid_h5_top16_p000006_to_top15_0603/P_forget.jsonl

candidate paths:
../mip_workspace/outputs/paths/P_cand_projector_dim_hybrid_h5_top16_p000006_to_top15_0603.jsonl

anchor pairs:
../mip_workspace/outputs/causal_pairs_train.jsonl
```

训练产物：

```text
../mip_workspace/outputs/masked_rmisu_h5_cf_anchor_name_a002_80steps_0607.json
../mip_workspace/outputs/checkpoints/masked_rmisu_h5_cf_anchor_name_a002_80steps_0607
```

## 5. 关键实验结果

### 5.1 Probability Diagnostic

Train split：

| 指标 | H5 | H5 + anchor | 结论 |
|---|---:|---:|---|
| `forget_clean target_name_ce delta` | `+0.010743` | `+0.019995` | 明显增强，超过 `+0.010` |
| `counterfactual_retain target_name_ce delta` | `-0.000462` | `-0.000259` | 仍为负，counterfactual 概率侧保持修复 |
| `counterfactual_retain target_answer_ce delta` | `-0.000730` | `-0.000308` | 仍为负 |
| `hard_retain target_name_ce delta` | `+0.000426` | `+0.000699` | 小幅 retain 风险，可接受 |

Val split：

| 指标 | H5 | H5 + anchor | 结论 |
|---|---:|---:|---|
| `forget_clean target_name_ce delta` | `+0.011417` | `+0.013719` | 仍超过 `+0.010` |
| `counterfactual_retain target_name_ce delta` | `-0.002297` | `+0.000793` | name CE 有轻微回升 |
| `counterfactual_retain target_answer_ce delta` | `-0.001128` | `-0.001200` | answer CE 更好 |
| `hard_retain target_name_ce delta` | `+0.000150` | `+0.001860` | 小风险，需要继续观察 |

结论：

```text
forget CE 硬门槛通过。
counterfactual probability train 侧修复，val 侧 name CE 轻微回升但 answer CE 改善。
hard retain 风险存在但幅度小。
```

### 5.2 Generation Evaluation

Train split：

| 指标 | PEFT baseline | H5 | H5 + anchor | 结论 |
|---|---:|---:|---:|---|
| `counterfactual_retain name_hit` | `0.5882` | `0.5706` | `0.5824` | 修复大部分 H5 损失 |
| `counterfactual_retain BLEU` | `0.2203` | `0.2100` | `0.2131` | 小幅修复，未完全回到 baseline |
| `counterfactual_retain ROUGE-L` | `0.4059` | `0.3957` | `0.4013` | 稳定修复 |
| `hard_retain name_hit` | `0.6029` | `0.6029` | `0.5941` | 小幅下降 |
| `forget_clean name_hit` | `0.6588` | `0.6294` | `0.6000` | 忘记更强 |

Val split：

| 指标 | PEFT baseline | H5 | H5 + anchor | 结论 |
|---|---:|---:|---:|---|
| `counterfactual_retain name_hit` | `0.7222` | `0.7222` | `0.7222` | 不变 |
| `counterfactual_retain BLEU` | `0.2167` | `0.2249` | `0.2245` | 基本持平 |
| `counterfactual_retain ROUGE-L` | `0.3858` | `0.3709` | `0.3755` | 相比 H5 小幅修复 |
| `hard_retain name_hit` | `0.6944` | `0.6944` | `0.6944` | 不变 |
| `forget_clean name_hit` | `0.6111` | `0.5556` | `0.5556` | 保持 H5 忘记表现 |

结论：

```text
counterfactual generation 的 H5 小幅损失得到修复。
修复不是完全恢复到 PEFT baseline，但方向稳定。
forget generation 没有退回，probability forget CE 反而增强。
```

## 6. 为什么这次有效

本次有效的关键不是大幅改 H5，而是控制干预剂量：

1. 从 H5 full checkpoint 继续训练，而不是重新从 PEFT baseline 开始。
2. anchor 只绑定 counterfactual retain 子分布，避免泛化成过强 retain preserve。
3. `alpha=0.02` 足够小，不会抵消 H5 的 forget CE ascent。
4. `scope=name` 对准 H5 损失最敏感的 identity token，而不是全 answer 强约束。
5. `80 steps` 是短程修复，不做长时间漂移。

经验上，这类 anchor 更适合用作局部修复项，而不是主目标。

## 7. 推荐门槛

以后类似实验可以用下面的 quick gate：

```text
Hard gate:
forget_clean target_name_ce delta >= +0.010

Counterfactual repair:
counterfactual_retain generation ROUGE-L >= H5
counterfactual_retain generation name_hit >= H5

Retain risk:
hard_retain name_hit 不应明显低于 H5
hard_retain target_name_ce delta 最好控制在 +0.002 内
```

本次 candidate：

```text
forget CE: pass
counterfactual generation repair: pass
hard retain risk: acceptable but needs Full CLEAR confirmation
```

## 8. 复现实验入口

### 8.1 Train Generation Eval

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mip-editor

PYTHONPATH=. python causal_mip/evaluation/step8_final_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_h5_cf_anchor_name_a002_80steps_0607 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_train.jsonl \
  --output ../mip_workspace/outputs/step8_eval_h5_cf_anchor_name_a002_80steps_0607_train.json \
  --split train \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80
```

### 8.2 Val Generation Eval

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mip-editor

PYTHONPATH=. python causal_mip/evaluation/step8_final_eval.py \
  --model_path ../mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint ../mip_workspace/outputs/checkpoints/masked_rmisu_h5_cf_anchor_name_a002_80steps_0607 \
  --pair_jsonl ../mip_workspace/outputs/causal_pairs_val.jsonl \
  --output ../mip_workspace/outputs/step8_eval_h5_cf_anchor_name_a002_80steps_0607_val.json \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80
```

### 8.3 快速抽指标

Generation：

```bash
jq -r '"set\tname_hit\tbleu\trougeL", (.summary.by_eval_set | to_entries[] | [.key, .value.name_hit_rate, .value.bleu, .value.rougeL] | @tsv)' \
  ../mip_workspace/outputs/step8_eval_h5_cf_anchor_name_a002_80steps_0607_train.json
```

Probability：

```bash
jq -r '.comparison.groups[] | [.group, (.target_name_ce.delta // ""), (.target_answer_ce.delta // ""), (.target_vs_generated_name_logprob_margin.delta // "")] | @tsv' \
  ../mip_workspace/outputs/diagnostics/step8_probability_diag_h5_cf_anchor_name_a002_80steps_0607_train.json
```

## 9. 下一步建议

当前候选可以作为 H5 的局部改进版进入更完整评估：

```text
h5_cf_anchor_name_a002_80steps_full_remote_0607
```

如果 Full CLEAR 暴露出 counterfactual 仍未完全恢复，可尝试两个低风险变体：

| 变体 | 预期 | 风险 |
|---|---|---|
| `alpha=0.03, scope=name, 80 steps` | 更强 counterfactual name 修复 | retain 风险略升，forget CE 需复查 |
| `alpha=0.02, scope=answer, 80 steps` | 更偏生成内容修复 | name-hit 修复可能弱于 name scope |

优先级建议：

```text
先 Full CLEAR 当前 candidate。
如果远端 generation 仍显示 counterfactual 损失，再试 alpha=0.03。
不要优先大幅增加 steps，因为容易产生 retain/forget 共同漂移。
```

## 10. 经验总结

本次最有价值的经验是：

```text
当 H5 已经满足 forget 方向但 counterfactual generation 有小损失时，
不要先减弱 forget objective，
而是用小剂量、子分布定向的 counterfactual retain anchor 做局部修复。
```

推荐默认起点：

```text
base = H5 full checkpoint
counterfactual_anchor_alpha = 0.02
counterfactual_anchor_scope = name
max_steps = 80
forget_clean target_name_ce delta gate = +0.010
```

这组设置在本次实验中达到了较好的 forget/retain/counterfactual trade-off。
