# Step7 Probe 强度与目标函数优化实验结果

本文档记录 Step6 specificity probe 之后，对 Step7 projector probe 强度和 forget 目标函数进行的小规模优化实验。实验目标是验证：

1. `projector / cross-modal path` 是否能真正进入 Step7 编辑集合。
2. 单纯增强 projector probe 是否能改善 forget_clean 遗忘。
3. 将 forget CE 收窄到 `answer/name token` 后，是否能提升因果编辑质量。

## 1. 背景结论

前序 Step6 specificity probe 显示：

- `vision_text` 路径在 `specificity_margin = forget_effect - gamma * retain_impact` 排名中位于全局前 8。
- 这些路径主要没有进入 `P_forget`，而是进入了 `P_shared`。
- 因此不应立即回到 Step2 重做 projector saliency，而是先建立 `P_projector_probe` 做弱编辑实验。

本轮使用的 projector probe 文件：

```text
mip_workspace/outputs/paths/step6_specificity_probe_suf_fix_step9/P_projector_probe.jsonl
```

## 2. 本轮代码改造

### 2.1 Step7 新增目标函数

在 `causal_mip/editing/masked_rmisu.py` 中新增：

| objective | 含义 |
| --- | --- |
| `answer_ce_ascent` | 只对答案 token 做 CE ascent |
| `name_ce_ascent` | 只对姓名 token 做 CE ascent |
| `activation_random_answer_ce` | activation random + answer scoped CE ascent |
| `activation_random_name_ce` | activation random + name scoped CE ascent |

新增配置项：

```text
target_ce_scope = all | answer | name
```

### 2.2 CLEAR collate 增加 token span metadata

在 `clear.py` 中为 batch item 增加：

```text
answer_token_positions
name_token_positions
```

真实 CLEAR/Qwen processor sanity check 结果：

| 样本 | name | answer tokens | name tokens |
| --- | --- | ---: | ---: |
| 0 | Ursula Schmidt | 30 | 4 |
| 1 | Giorgi Meladze | 32 | 6 |
| 2 | Gabriela Carrasco | 24 | 5 |
| 3 | Jaime Vasquez | 31 | 4 |

说明 `name token` 能正确定位到答案中的姓名片段。

### 2.3 显存修正

Step7 preserve forward 原本会传入 `labels`，导致模型内部计算完整 CE loss，占用额外显存。已改为默认不传 `labels`，只在 scoped CE 中用 logits 手动计算目标 token loss。

验证命令：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile causal_mip/editing/masked_rmisu.py clear.py ours.py main.py causal_mip/test_step7_masked_rmisu.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step7_masked_rmisu.py
```

结果：

```text
Step 7 masked RMisU tests passed.
```

## 3. 实验配置

公共配置：

| 项目 | 配置 |
| --- | --- |
| 模型 | `Qwen2.5-VL-3B-Instruct` |
| 数据集 | `CLEAR` |
| forget ratio | `5%` |
| batch size | `2` |
| ptm checkpoint batch size | `2` |
| learning rate | `1e-5` |
| max steps | `120` 或 `300` |
| Step6 path dir | `mip_workspace/outputs/paths/step6_suf_fix_step9` |
| candidate paths | `mip_workspace/outputs/paths/P_cand.jsonl` |
| projector probe | `mip_workspace/outputs/paths/step6_specificity_probe_suf_fix_step9/P_projector_probe.jsonl` |

Step8 pair eval 配置：

```text
pair_jsonl = mip_workspace/outputs/causal_pairs_val.jsonl
split = val
max_new_tokens = 80
image_resize = 224
```

指标说明：

- `forget_clean name_hit_rate` 越低越好。
- `hard_retain / counterfactual_retain name_hit_rate` 越高越好。

## 4. Step7 运行结果

### 4.1 weak projector probe

run id：

```text
step6_probe_weak_ptm2_0527_152213
```

配置：

| 项目 | 值 |
| --- | ---: |
| max steps | 120 |
| `probe_beta` | 0.03 |
| `forget_objective` | `activation_random_ce` |
| `forget_ce_alpha` | 0.05 |
| learning rate | 1e-5 |

关键现象：

- projector path 成功进入编辑集合。
- projector edit module 为 `visual.merger.mlp.0`。
- `probe_neurons = [0]`。

Step8：

```text
mip_workspace/outputs/step8_eval_step6_probe_weak_ptm2_0527_152213_val.json
```

### 4.2 增强 projector probe

run id：

```text
step7_probe_b005_ce01_s300_0527_154024
```

配置：

| 项目 | 值 |
| --- | ---: |
| max steps | 300 |
| `probe_beta` | 0.05 |
| `forget_objective` | `activation_random_ce` |
| `forget_ce_alpha` | 0.1 |
| learning rate | 1e-5 |

Step7 summary：

```text
mip_workspace/outputs/masked_rmisu_step7_probe_b005_ce01_s300_0527_154024.json
```

loss 统计：

| loss | first | last | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `probe_loss` | 189 | 121 | 143.852 | 112 | 206 |
| `unlearn_loss` | 556 | 840 | 641.253 | 314 | 1272 |
| `retain_loss` | - | - | 1.440 | - | 17.125 |
| `shared_loss` | - | - | 1.009 | - | 12.000 |

结论：

- projector probe 的随机扰动增强后，没有改善 forget。
- `forget_clean name_hit_rate` 从 baseline 的 `0.611` 升到 `0.722`，方向变差。

### 4.3 activation random + name scoped CE

run id：

```text
step7_probe_namece_b003_ce02_s120_0527_155759
```

配置：

| 项目 | 值 |
| --- | ---: |
| max steps | 120 |
| `forget_objective` | `activation_random_name_ce` |
| `forget_ce_alpha` | 0.2 |
| `probe_beta` | 0.03 |
| `rmu_beta` | 0.5 |

Step7 summary：

```text
mip_workspace/outputs/masked_rmisu_step7_probe_namece_b003_ce02_s120_0527_155759.json
```

关键统计：

| 项目 | 值 |
| --- | ---: |
| `num_modules` | 68 |
| `num_editable_neurons` | 1040 |
| `num_probe_neurons` | 14 |
| `forget_ce_scope` | `name` |
| `forget_ce_token_count` mean | 15.05 |
| `forget_ce_loss` mean | 0.3415 |
| `probe_loss` mean | 158.6417 |
| `unlearn_loss` mean | 640.7333 |

结论：

- name CE 确实只覆盖少量姓名 token。
- 但 activation random 项仍然远大于 name CE，主导了更新方向。
- Step8 结果退化明显。

### 4.4 name CE only + weak projector probe

run id：

```text
step7_namece_only_probe001_ce2_s120_0527_160135
```

配置：

| 项目 | 值 |
| --- | ---: |
| max steps | 120 |
| `forget_objective` | `name_ce_ascent` |
| `forget_ce_alpha` | 2.0 |
| `probe_beta` | 0.01 |
| `rmu_beta` | 0.0 |

Step7 summary：

```text
mip_workspace/outputs/masked_rmisu_step7_namece_only_probe001_ce2_s120_0527_160135.json
```

关键统计：

| 项目 | 值 |
| --- | ---: |
| `forget_ce_scope` | `name` |
| `forget_ce_token_count` mean | 15.05 |
| `forget_ce_loss` mean | 0.3418 |
| `probe_loss` mean | 161.2833 |
| `unlearn_loss` mean | 0.0 |
| `retain_loss` mean | 1.1903 |
| `shared_loss` mean | 0.8734 |

结论：

- 移除 activation random 后，结果比 mixed objective 好。
- 但 `forget_clean` 仍未低于 PEFT baseline。

### 4.5 name CE only + probe loss off

run id：

```text
step7_namece_probe0_ce4_s120_0527_160447
```

配置：

| 项目 | 值 |
| --- | ---: |
| max steps | 120 |
| `forget_objective` | `name_ce_ascent` |
| `forget_ce_alpha` | 4.0 |
| `probe_beta` | 0.0 |
| `rmu_beta` | 0.0 |

Step7 summary：

```text
mip_workspace/outputs/masked_rmisu_step7_namece_probe0_ce4_s120_0527_160447.json
```

关键统计：

| 项目 | 值 |
| --- | ---: |
| `forget_ce_scope` | `name` |
| `forget_ce_token_count` mean | 15.05 |
| `forget_ce_loss` mean | 0.3413 |
| `probe_loss` mean | 0.0 |
| `unlearn_loss` mean | 0.0 |
| `retain_loss` mean | 0.9926 |
| `shared_loss` mean | 0.7299 |

结论：

- 关闭 random probe loss 后，projector neuron 仍在 trainable set 中，但只由 name CE 和 preserve loss 传递梯度。
- `forget_clean` 没有进一步改善。
- counterfactual retain 下降，说明继续加大 name CE alpha 会伤害 retain specificity。

## 5. Step8 Pair Eval 对比

| run | forget_clean | hard_retain | counterfactual_retain | 判断 |
| --- | ---: | ---: | ---: | --- |
| PEFT baseline | 0.611 | 0.694 | 0.722 | baseline |
| old Step9 bound projector | 0.778 | 0.722 | 0.778 | forget 退化 |
| weak probe 120 | 0.611 | 0.722 | 0.722 | 与 baseline 持平，retain 略好 |
| probe 300 | 0.722 | 0.694 | 0.722 | probe 加强后 forget 变差 |
| activation_random + name CE | 0.778 | 0.722 | 0.667 | 明显无效，retain specificity 变差 |
| name CE only + weak probe | 0.667 | 0.694 | 0.722 | 比 mixed objective 好，但未超过 baseline |
| name CE only + probe off | 0.667 | 0.722 | 0.667 | forget 未改善，counterfactual 下降 |

## 6. 实验结论

### 6.1 有效部分

1. projector / cross-modal path 已经能真正进入 Step7 编辑集合。
2. `P_projector_probe` 中的 projector neuron 能被加入 trainable set。
3. `visual.merger.mlp.0` 可作为 Qwen projector 编辑模块。
4. answer/name token scoped CE 的实现正确，真实 CLEAR 样本中能定位姓名 token。
5. Step7 forward 显存问题已修复。

### 6.2 无效部分

1. 单纯增强 projector probe 强度无效。
   - 300-step probe 将 `forget_clean` 从 baseline `0.611` 推高到 `0.722`。
2. `activation_random + name CE` 无效。
   - activation random loss 量级远大于 name CE，主导了错误更新方向。
3. 单纯 `name_ce_ascent` 不足以超过 baseline。
   - 最好结果为 `forget_clean = 0.667`，仍差于 baseline `0.611`。
4. 关闭 probe loss 后没有改善。
   - `forget_clean` 仍为 `0.667`，且 counterfactual retain 降到 `0.667`。

## 7. 路线判断

本轮实验说明：

```text
projector path selection/editing plumbing 已经打通，
但当前 Step7 的 activation_random / CE ascent 目标函数不够对齐。
```

因此下一步不建议继续：

- 单纯加大 `probe_beta`
- 单纯加大 `probe_steering_coeff`
- 单纯延长 activation random 训练步数
- 单纯提高 `name_ce_alpha`

更合理的下一步是转向更贴近 machine unlearning 的目标函数：

1. **NPO / negative preference objective**
   - 降低 forget 样本真实姓名答案相对于替代答案或拒答答案的偏好。
   - 比 CE ascent 更接近“不要再生成这个人名”。

2. **replacement / refusal objective**
   - 为 forget 样本构造 `unknown`、`cannot identify` 或 counterfactual replacement target。
   - 用 retain KL 或 preserve loss 保住 retain。

3. **KL-constrained preference editing**
   - forget 侧做 preference suppression。
   - retain 侧做 KL / activation preserve。

## 8. 文件清理记录

已删除无效的大 checkpoint，以释放磁盘空间：

```text
mip_workspace/outputs/checkpoints/masked_rmisu_step6_probe_weak_0527_151624
mip_workspace/outputs/checkpoints/masked_rmisu_step7_probe_b005_ce01_s300_0527_154024
mip_workspace/outputs/checkpoints/masked_rmisu_step7_probe_namece_b003_ce02_s120_0527_155759
mip_workspace/outputs/checkpoints/masked_rmisu_step7_namece_only_probe001_ce2_s120_0527_160135
mip_workspace/outputs/checkpoints/masked_rmisu_step7_namece_probe0_ce4_s120_0527_160447
```

保留了小体积记录文件：

```text
mip_workspace/outputs/masked_rmisu_*.json
mip_workspace/outputs/step8_eval_*.json
mip_workspace/outputs/logs/step7_*.log
mip_workspace/outputs/logs/step8_pair_eval_*.log
```

清理后磁盘可用空间从约 `63G` 恢复到约 `98G`。

## 9. 最终结论

本轮优化在工程上有效，但在遗忘效果上没有达到目标：

- projector probe 路径确实进入编辑集合。
- name scoped CE 确实只作用于姓名 token。
- 但 Step8 指标没有优于 PEFT baseline。

因此当前实验结论是：

```text
Step7 的路径编辑机制可用，
但 forget objective 需要从 activation random / CE ascent 转向 NPO 或 replacement-style objective。
```
