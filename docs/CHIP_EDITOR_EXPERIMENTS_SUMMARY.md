# CHIP-Editor Experiments Summary

更新时间：2026-05-29

本文合并旧实验结果文档，记录 CHIP-Editor 相关主实验、Step7 probe 和最终判断。路线状态见 `CHIP_EDITOR_ROUTE_STATUS.md`，实现审计见 `CHIP_EDITOR_IMPLEMENTATION_AUDIT.md`。

## 1. 总体实验结论

按当前证据：

```text
0430 原始 our baseline 在原协议下表现为强遗忘、retain 基本稳定；
0526/0527 causal masked RMisU 路线工程上跑通，但没有稳定超过 PEFT baseline；
Step9 bound projector 完整跑通 Full CLEAR 和 Step8 protocol，但 pair screen 失败，不能声明成功。
```

最重要的负结论：

```text
当前 causal selective editing 的主要失败点不是训练没跑通，
而是 forget path 因果选择质量、projector 实际编辑覆盖和 Step7 objective 尚不足。
```

## 2. 0430 baseline: `0430_141446`

配置：

| item | value |
| --- | --- |
| model | local `Qwen2.5-VL-3B-Instruct` |
| judge | remote `Qwen2.5-VL-7B-Instruct` |
| dataset | CLEAR |
| method | `our` |
| forget ratio | 5% |
| batch size | 2 |
| epochs / finetune_epochs | 1 / 1 |
| seed | 42 |

结果：

| task | before | after | delta |
| --- | ---: | ---: | ---: |
| forget multi classification | 22.34% | 10.11% | -12.23 pp |
| retain multi classification | 23.38% | 21.40% | -1.98 pp |
| forget multi generation acc | 56.38% | 10.64% | -45.74 pp |
| retain multi generation acc | 55.00% | 57.37% | +2.37 pp |

forget generation metrics：

| metric | before | after |
| --- | ---: | ---: |
| Rouge1 | 24.10% | 5.43% |
| Rouge2 | 8.63% | 1.52% |
| RougeL | 18.96% | 4.29% |
| Bleu | 4.11% | 1.95% |

训练：

```text
1790 steps
mean loss ~= 24.64
training completed normally
```

限制：

1. CLEAR text 模态没有有效参与评估。
2. 单 seed 单次 run。
3. 远程 judge 有波动。
4. 与后续 restored Full CLEAR 协议不完全等价。

结论：在原协议下是正向 baseline，但不能直接作为后续固定 Step8/Full CLEAR 协议的成功证明。

## 3. CHIP full causal masked RMisU: `chip_full_train_0526_1348`

配置：

| item | value |
| --- | --- |
| base | `Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth` |
| dataset | CLEAR |
| method | `our --use_masked_rmisu` |
| Step6 mask | `step6_v1_quantile75` |
| forget CE alpha | 0.1 |
| `rmu_alpha / rmu_beta` | 0.5 / 0.5 |
| shared alpha | 1.0 |
| learning rate | 1e-5 |

产物：

```text
mip_workspace/outputs/masked_rmisu_chip_full_train_0526_1348.json
mip_workspace/outputs/checkpoints/masked_rmisu_chip_full_train_0526_1348
mip_workspace/outputs/step8_eval_chip_full_train_0526_1348_val_full.json
mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/full_clear_remote_protocol_summary.json
```

训练结果：

| metric | value |
| --- | ---: |
| steps | 1790 / 1790 |
| checkpoint size | 7.1G |
| mask modules | 18 |
| editable neurons | 45 |
| merged PartialLinear modules | 36 |
| loss mean | 35.0355 |
| forget CE mean | 4.3962 |

Step8 val：

| model | eval_set | n | name_hit_rate | BLEU | ROUGE-L |
| --- | --- | ---: | ---: | ---: | ---: |
| PEFT baseline | forget_clean | 18 | 0.6111 | 0.2090 | 0.4015 |
| CHIP full 0526 | forget_clean | 18 | 0.6111 | 0.2025 | 0.4003 |
| PEFT baseline | hard_retain | 36 | 0.6944 | 0.1591 | 0.3519 |
| CHIP full 0526 | hard_retain | 36 | 0.7222 | 0.1686 | 0.3547 |
| PEFT baseline | counterfactual_retain | 18 | 0.7222 | 0.2167 | 0.3858 |
| CHIP full 0526 | counterfactual_retain | 18 | 0.6667 | 0.2089 | 0.3641 |

Full CLEAR remote vs restored full baseline：

| metric | CHIP full 0526 | restored full baseline | delta |
| --- | ---: | ---: | ---: |
| forget classification remote acc | 69.68% | 55.32% | +14.36 pp |
| retain classification remote acc | 70.36% | 56.79% | +13.58 pp |
| forget generation remote acc | 44.68% | 1.06% | +43.62 pp |
| retain generation remote acc | 43.02% | 0.11% | +42.91 pp |
| forget generation BLEU | 24.03% | 5.27% | +18.76 pp |
| forget generation ROUGE-L | 42.81% | 19.62% | +23.19 pp |

结论：

```text
训练链路完整跑通；
但 Step8 forget_clean 没有低于 PEFT baseline；
Full CLEAR forget-set remote acc 反而偏高；
counterfactual retain 有轻微损伤。
```

因此该 run 不是成功遗忘。

## 4. Step9 bound projector: `step9_bound_projector_0526_214028`

配置：

| item | value |
| --- | --- |
| model | local `Qwen2.5-VL-3B-Instruct` |
| judge | remote `Qwen2.5-VL-7B-Instruct` |
| dataset | CLEAR |
| method | causal selective editing / masked RMisU |
| forget ratio | 5% |
| batch size | 2 |
| epochs / finetune_epochs | 1 / 1 |

checkpoint：

```text
mip_workspace/outputs/checkpoints/masked_rmisu_step9_bound_projector_0526_214028
```

Step2/3:

```text
mip_workspace/outputs/paths/P_cand_bound_train_step9_bound_projector_0526_214028.jsonl
mip_workspace/outputs/paths/P_cand_bound_val_step9_bound_projector_0526_214028.jsonl
```

Step5/6:

```text
mip_workspace/outputs/scores/path_scores_bound_projector_step9_bound_projector_0526_214028.jsonl
mip_workspace/outputs/paths/step6_bound_suf_projector_step9_bound_projector_0526_214028
```

Step6 classification:

| category | count |
| --- | ---: |
| `P_forget` | 100 |
| `P_shared` | 110 |
| `P_retain` | 221 |
| `P_irrelevant` | 937 |
| total | 1368 |

Modality:

| modality | count |
| --- | ---: |
| text | 680 |
| vision | 680 |
| vision_text | 8 |

Step7 summary:

```text
mip_workspace/outputs/masked_rmisu_step9_bound_projector_0526_214028.json
```

Step7 stats:

| item | value |
| --- | ---: |
| mask modules | 65 |
| editable neurons | 615 |
| merged PartialLinear modules | 130 |
| checkpoint saved | yes |
| projector modules | 0 |

Important caveat:

```text
虽然本轮启用了 projector 相关路线，
但最终 Step7 summary 中 num_projector_modules=0；
实际编辑仍主要是 LLM/vision down_proj，没有真正编辑 projector。
```

Full CLEAR remote:

| metric | candidate | PEFT baseline | delta |
| --- | ---: | ---: | ---: |
| forget classification remote acc | 0.675532 | 0.686170 | -0.010638 |
| forget generation remote acc | 0.457447 | 0.462766 | -0.005319 |
| forget generation BLEU | 0.230842 | 0.236700 | -0.005858 |
| forget generation ROUGE-L | 0.419460 | 0.425851 | -0.006391 |
| retain classification remote acc | 0.703352 | 0.703631 | -0.000279 |
| retain generation remote acc | 0.429330 | 0.432123 | -0.002793 |

Full CLEAR main:

```text
passed
```

但 forget 降幅很小：

```text
classification about -1.06 pp
generation about -0.53 pp
```

Step8 pair screen:

| eval set | candidate | PEFT baseline | delta | check |
| --- | ---: | ---: | ---: | --- |
| forget_clean | 0.777778 | 0.611111 | +0.166667 | failed |
| hard_retain | 0.722222 | 0.694444 | +0.027778 | passed |
| counterfactual_retain | 0.777778 | 0.722222 | +0.055556 | passed |

Step8 protocol:

```text
status: do_not_claim_success
can_claim_success: false
failed_reasons:
  - pair_screen:pair_forget_clean_name_hit_rate
```

结论：

```text
工程链路完整跑通；
Full CLEAR 主评测通过当前阈值；
但 Step8 总协议未通过，不能声明成功遗忘。
```

后续 Science Q1 重新审计发现，该 run 的旧 `P_forget=100` 缺少 positive `Suf` 支撑，因此不能继续作为可靠因果路径集合。

## 5. Step7 projector probe / objective optimization

目的：

1. 验证 projector / cross-modal path 是否能真正进入 Step7。
2. 验证增强 projector probe 是否改善 forget_clean。
3. 验证 answer/name scoped CE 是否提升因果编辑质量。

公共配置：

```text
dataset = CLEAR
forget ratio = 5%
batch size = 2
learning_rate = 1e-5
pair_jsonl = causal_pairs_val.jsonl
max_new_tokens = 80
```

Step8 pair eval:

| run | forget_clean | hard_retain | counterfactual_retain | 判断 |
| --- | ---: | ---: | ---: | --- |
| PEFT baseline | 0.611 | 0.694 | 0.722 | baseline |
| old Step9 bound projector | 0.778 | 0.722 | 0.778 | forget 退化 |
| weak probe 120 | 0.611 | 0.722 | 0.722 | 与 baseline 持平，retain 略好 |
| probe 300 | 0.722 | 0.694 | 0.722 | probe 加强后 forget 变差 |
| activation_random + name CE | 0.778 | 0.722 | 0.667 | 明显无效，retain specificity 变差 |
| name CE only + weak probe | 0.667 | 0.694 | 0.722 | 比 mixed objective 好，但未超过 baseline |
| name CE only + probe off | 0.667 | 0.722 | 0.667 | forget 未改善，counterfactual 下降 |

有效发现：

1. projector / cross-modal path 已经能真正进入 Step7 编辑集合。
2. projector edit module 为 `visual.merger.mlp.0`。
3. answer/name token scoped CE 实现正确。
4. preserve forward 显存问题已修复。

无效发现：

1. 单纯增强 projector probe 强度无效。
2. `activation_random + name CE` 无效，activation random loss 主导更新。
3. 单纯 `name_ce_ascent` 不足以超过 baseline。
4. 关闭 probe loss 后没有改善，并可能伤害 counterfactual retain。

路线判断：

```text
Step7 路径编辑机制可用；
forget objective 需要从 activation random / CE ascent 转向 NPO 或 replacement-style objective。
```

## 6. 0529 formal stable single-candidate smoke

该结果已归入 Science Q1 新文档，但对 CHIP-Editor 路线也有意义。

run id：

```text
step7_single_stable_smoke_retainaware_0529_152017
```

性质：

```text
formal stable single-candidate smoke
```

结果：

| field | value |
| --- | ---: |
| `num_loss_records` | 5 |
| `mask_summary.num_modules` | 1 |
| `mask_summary.num_editable_neurons` | 4 |
| skipped modules | 0 |

module:

| field | value |
| --- | --- |
| logical module | `mm_projector` |
| edit module | `visual.merger.mlp.0` |
| editable dims | `[318, 387, 886, 984]` |

Step8 diagnostic:

```text
retain-side 出现一个正向信号；
forget-side 无可判定信号，因为 baseline 已不命中 pair_000089 目标姓名。
```

结论：这是 plumbing + minimal edit check，不是正式 Step7 成功。

## 7. 总结判断

当前所有 CHIP-Editor 实验合起来说明：

```text
1. 原始 our baseline 可以在旧协议下表现很好。
2. causal selective editing 管线已经能完整跑通并保存 checkpoint。
3. projector edit plumbing 已打通到 visual.merger.mlp.0。
4. 固定 Step8 protocol 能阻止“Full CLEAR 微弱通过但 pair forget 失败”的误报。
5. 当前 masked RMisU objective 尚不能稳定降低 forget_clean name-hit。
6. 下一步应优先改 path selection 与 forget objective，而不是盲目扩大训练。
```

下一步实验方向：

```text
exact target probability / scoped CE diagnostic
NPO / replacement / refusal objective
retain KL / CE preserve
pair-level / node-level / dim-level stable candidate selection
small stable multi-candidate diagnostic
```

