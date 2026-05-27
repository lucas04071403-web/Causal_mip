# CHIP-Editor Full CLEAR 实验结果：`step9_bound_projector_0526_214028`

本文档记录按当前已修改方法完成的一次完整 CLEAR 实验，目标是检查技术路线从 candidate path 到 selective editing 的闭环效果，并用固定 Step8 主协议判断是否可以声明遗忘成功。

## 1. 实验配置

| 项目 | 配置 |
| --- | --- |
| 训练模型 | 本地 `Qwen2.5-VL-3B-Instruct` |
| 评分模型 | 远程 `Qwen2.5-VL-7B-Instruct` |
| 远程评分地址 | `http://210.40.56.85:21936/v1` |
| 数据集 | `CLEAR` |
| forget ratio | `5%` |
| batch size | `2` |
| epochs | `1` |
| finetune_epochs | `1` |
| 遗忘方法 | 当前修改后的 causal selective editing / masked RMisU 方法 |
| run id | `step9_bound_projector_0526_214028` |

被评估候选 checkpoint：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step9_bound_projector_0526_214028
```

对照 baseline：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/llms/Qwen2.5-VL-3B-Instruct
```

## 2. 路线链路产物

### Step 2/3: 样本绑定 candidate path

本轮使用样本绑定后的 candidate path 文件：

```text
mip_workspace/outputs/paths/P_cand_bound_train_step9_bound_projector_0526_214028.jsonl
mip_workspace/outputs/paths/P_cand_bound_val_step9_bound_projector_0526_214028.jsonl
```

### Step 5/6: 因果评分与路径四分类

Step5 score 文件：

```text
mip_workspace/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl
```

Step6 分类目录：

```text
mip_workspace/outputs/paths/step6_bound_suf_projector_step9_bound_projector_0526_214028
```

Step6 分类统计：

| category | count |
| --- | ---: |
| `P_forget` | 100 |
| `P_shared` | 110 |
| `P_retain` | 221 |
| `P_irrelevant` | 937 |
| total classified paths | 1368 |

按模态统计：

| modality | count |
| --- | ---: |
| text | 680 |
| vision | 680 |
| vision_text | 8 |

阈值：

| threshold | value |
| --- | ---: |
| forget threshold | 0.0078125 |
| retain threshold | 0.005208333333333333 |
| forget quantile | 0.75 |
| retain quantile | 0.75 |

### Step 7: selective editing

Step7 summary：

```text
mip_workspace/outputs/masked_rmisu_step9_bound_projector_0526_214028.json
```

Step7 编辑统计：

| 项目 | 数值 |
| --- | ---: |
| mask modules | 65 |
| editable neurons | 615 |
| merged PartialLinear modules | 130 |
| checkpoint saved | yes |

注意：本轮虽然启用了 projector 相关路线，但最终 Step7 summary 中 `num_projector_modules=0`，也就是实际进入 `P_forget/P_shared` 并被编辑的模块仍主要是 LLM/vision `down_proj`，没有真正编辑 projector 模块。

## 3. Full CLEAR 远程评分完整性

候选模型与 PEFT baseline 的四个 remote-scored 文件均检查通过：

| model | task | partial | num_examples | num_scored | preds |
| --- | --- | --- | ---: | ---: | ---: |
| candidate | `clf_forget` | false | 188 | 188 | 188 |
| candidate | `clf_retain` | false | 3580 | 3580 | 3580 |
| candidate | `gen_forget` | false | 188 | 188 | 188 |
| candidate | `gen_retain` | false | 3580 | 3580 | 3580 |
| PEFT baseline | `clf_forget` | false | 188 | 188 | 188 |
| PEFT baseline | `clf_retain` | false | 3580 | 3580 | 3580 |
| PEFT baseline | `gen_forget` | false | 188 | 188 | 188 |
| PEFT baseline | `gen_retain` | false | 3580 | 3580 | 3580 |

Full CLEAR summary 文件：

```text
mip_workspace/outputs/full_clear_step9_bound_projector_0526_214028.json
mip_workspace/outputs/full_clear_peft_baseline_step9_bound_projector_0526_214028.json
```

## 4. Full CLEAR 指标

`delta = candidate - PEFT baseline`。forget 侧 remote acc 越低越好，retain 侧 remote acc 越高越好。

| metric | candidate | PEFT baseline | delta |
| --- | ---: | ---: | ---: |
| forget classification remote acc | 0.675532 | 0.686170 | -0.010638 |
| forget generation remote acc | 0.457447 | 0.462766 | -0.005319 |
| forget generation BLEU | 0.230842 | 0.236700 | -0.005858 |
| forget generation ROUGE-L | 0.419460 | 0.425851 | -0.006391 |
| retain classification remote acc | 0.703352 | 0.703631 | -0.000279 |
| retain generation remote acc | 0.429330 | 0.432123 | -0.002793 |
| retain generation BLEU | 0.238751 | 0.239301 | -0.000550 |
| retain generation ROUGE-L | 0.423723 | 0.423476 | +0.000246 |

Full CLEAR 主评测结论：

```text
full_clear_main: passed
```

原因：

- forget classification / generation remote acc 均低于 PEFT baseline，满足当前协议中 `min_full_forget_remote_acc_drop=0.0` 的降低要求。
- retain classification / generation remote acc 的下降远小于 `max_full_retain_remote_acc_drop=0.05`，满足保留能力容忍阈值。
- 四个 Full CLEAR 任务均为完整 scored 文件，不是 partial 结果。

但要注意：forget 降幅很小，classification 约 `-1.06 pp`，generation 约 `-0.53 pp`，只能说明在当前协议阈值下通过，不能说明遗忘效果强。

## 5. Step8 主协议

Step8 protocol 输出：

```text
mip_workspace/outputs/step8_protocol_step9_bound_projector_0526_214028.json
```

协议规则：

```text
pair_screen_must_pass_and_full_clear_remote_main_must_pass
```

### Pair screen

| eval set | candidate name_hit_rate | PEFT baseline name_hit_rate | delta | check |
| --- | ---: | ---: | ---: | --- |
| forget_clean | 0.777778 | 0.611111 | +0.166667 | failed |
| hard_retain | 0.722222 | 0.694444 | +0.027778 | passed |
| counterfactual_retain | 0.777778 | 0.722222 | +0.055556 | passed |

Pair screen 结论：

```text
pair_screen: failed
failed reason: pair_screen:pair_forget_clean_name_hit_rate
```

解释：`forget_clean` 的 name hit rate 应该低于 baseline，但本轮候选模型反而更高，说明在 Step8 val pair 的身份命中指标上没有形成有效遗忘。

### Full CLEAR main

| check | candidate | PEFT baseline | delta | result |
| --- | ---: | ---: | ---: | --- |
| forget classification remote acc | 0.675532 | 0.686170 | -0.010638 | passed |
| forget generation remote acc | 0.457447 | 0.462766 | -0.005319 | passed |
| retain classification remote acc | 0.703352 | 0.703631 | -0.000279 | passed |
| retain generation remote acc | 0.429330 | 0.432123 | -0.002793 | passed |

Full CLEAR main 结论：

```text
full_clear_main: passed
```

### 总决策

固定 Step8 主协议最终决策：

```text
status: do_not_claim_success
can_claim_success: false
failed_reasons:
  - pair_screen:pair_forget_clean_name_hit_rate
```

也就是说，本轮实验不能声明技术路线已经在科学问题意义上成功闭环。

## 6. 对三个科学问题的回答

### Q1: 能否找到与目标知识相关的候选路径？

工程上已经可以。当前链路产出了样本绑定后的 candidate path，并进入 Step5/6 评分与分类。

但 `vision_text=8` 仍然偏少，跨模态路径覆盖有限。

### Q2: 候选路径是否被因果验证为 forget/shared/retain 等角色？

部分成立。Step5 score 与 Step6 四分类都已产出，且本轮分类得到 `P_forget=100`、`P_shared=110`、`P_retain=221`。

但 projector 相关路径没有真正进入最终编辑集合，且 Step8 pair screen 中 forget_clean 未下降，说明当前因果分类证据还不足以支撑强结论。

### Q3: 选择性编辑是否能遗忘目标知识并保护 retain？

工程上已经完成选择性编辑并保存 checkpoint。Full CLEAR 主评测在当前协议阈值下通过，retain 损伤非常小。

但 Step8 主协议未通过，核心失败点是 `forget_clean name_hit_rate` 从 PEFT baseline 的 `0.611111` 升到候选模型的 `0.777778`。因此不能声明已经实现可靠选择性遗忘。

## 7. 验证命令与结果

已运行 Step8 主协议：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.evaluation.step8_protocol \
  --pair_eval /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/step8_eval_step9_bound_projector_0526_214028_val.json \
  --pair_baseline /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/step8_eval_peft_baseline_val_full_0522.json \
  --full_clear_summary /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/full_clear_step9_bound_projector_0526_214028.json \
  --full_clear_baseline /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/full_clear_peft_baseline_step9_bound_projector_0526_214028.json \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace/outputs/step8_protocol_step9_bound_projector_0526_214028.json
```

输出：

```text
status = do_not_claim_success
can_claim_success = false
failed_reasons = ["pair_screen:pair_forget_clean_name_hit_rate"]
```

已运行协议单元测试：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step8_protocol.py
```

结果：

```text
Step 8 protocol tests passed.
```

## 8. 最终结论

本轮实验已经完整跑完候选模型与 PEFT baseline 的 Full CLEAR 远程评分，并生成固定 Step8 主协议报告。

结论是：

```text
工程链路完整跑通；
Full CLEAR 主评测通过当前协议阈值；
但 Step8 总协议未通过，不能声明成功遗忘。
```

下一步最优先的问题不是继续扩大训练轮数，而是修正 forget path 的因果选择质量和 projector path 的实际编辑覆盖。当前结果显示 retain 保护较稳，但 forget 侧的身份命中没有被 Step8 pair screen 认可。
