# CHIP-Editor 实验结果总结：`chip_full_train_0526_1348`

## 1. 实验配置

本次实验参考原 baseline 配置，并启用 Step 2-Step 8 的 causal masked RMisU 路线：

- 训练模型：本地 `Qwen2.5-VL-3B-Instruct`
- 原始 PEFT 起点：`Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth`
- 数据集：`CLEAR`
- forget ratio：`5%`
- batch size：`2`
- epochs：`1`
- finetune_epochs：`1`
- unlearning：`our`
- causal edit：`--use_masked_rmisu`
- Step6 mask：`step6_v1_quantile75`
- `masked_rmisu_forget_ce_alpha=0.1`
- `rmu_alpha=0.5`
- `rmu_beta=0.5`
- `masked_rmisu_shared_alpha=1.0`
- `learning_rate=1e-5`

说明：尝试过用 `main.py --eval_flag --use_remote_scoring` 直接跑完整训练前/训练后全量远程评分，但进程卡在训练前 retain95 远程 judge 阶段超过 20 分钟且未继续落盘，因此改为可控流程：

```text
完整训练并保存 checkpoint
→ Step8 fixed val protocol 评估
→ 与 PEFT baseline / 0430 baseline 结论对比
```

## 2. 关键产物

训练 summary：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_chip_full_train_0526_1348.json
```

checkpoint：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_chip_full_train_0526_1348
```

Step8 评估：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_chip_full_train_0526_1348_val_full.json
```

Full CLEAR 远程评分：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/full_clear_remote_protocol_summary.json
```

## 3. 训练结果

- 训练步数：`1790 / 1790`
- checkpoint 大小：`7.1G`
- mask modules：`18`
- editable neurons：`45`
- merged PartialLinear modules：`36`
- loss mean：`35.0355`
- forget CE mean：`4.3962`

首步：

```text
loss=34.3166
unlearn_loss=69.5
forget_ce_loss=4.3337
retain_loss=0.00473
shared_loss=0.00479
```

末步：

```text
loss=36.8247
unlearn_loss=74.5
forget_ce_loss=4.2531
retain_loss=0.02356
shared_loss=0.00543
```

训练正常完成，没有 OOM 或异常中断。

## 4. Step8 Val 结果

评估集：

```text
causal_pairs_val.jsonl = 18 pairs
total examples = 72
forget_clean = 18
hard_retain = 36
counterfactual_retain = 18
```

### 4.1 by eval_set

| model | eval_set | n | name_hit_rate | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| PEFT baseline | forget_clean | 18 | 0.6111 | 0.2090 | 0.4015 |
| CHIP full 0526 | forget_clean | 18 | 0.6111 | 0.2025 | 0.4003 |
| quantile75 CE 0522 | forget_clean | 18 | 0.6111 | 0.2104 | 0.4096 |
| PEFT baseline | hard_retain | 36 | 0.6944 | 0.1591 | 0.3519 |
| CHIP full 0526 | hard_retain | 36 | 0.7222 | 0.1686 | 0.3547 |
| quantile75 CE 0522 | hard_retain | 36 | 0.7222 | 0.1574 | 0.3473 |
| PEFT baseline | counterfactual_retain | 18 | 0.7222 | 0.2167 | 0.3858 |
| CHIP full 0526 | counterfactual_retain | 18 | 0.6667 | 0.2089 | 0.3641 |
| quantile75 CE 0522 | counterfactual_retain | 18 | 0.7222 | 0.2241 | 0.3743 |

### 4.2 by sample_type

| model | sample_type | n | name_hit_rate | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| CHIP full 0526 | hard_retain::same_topic | 18 | 0.6667 | 0.1044 | 0.3109 |
| CHIP full 0526 | hard_retain::same_reasoning | 18 | 0.7778 | 0.2008 | 0.3930 |
| CHIP full 0526 | counterfactual_retain | 18 | 0.6667 | 0.2089 | 0.3641 |

## 5. 与原 baseline 的对比结论

### 5.1 Full CLEAR 远程评分结果

评分配置：

- 训练/被评模型：`masked_rmisu_chip_full_train_0526_1348`
- 评分模型：远程 `Qwen2.5-VL-7B-Instruct`
- 数据集：`CLEAR`
- forget ratio：`5%`
- 任务：`clf_forget`, `clf_retain`, `gen_forget`, `gen_retain`
- 评分样本：forget `188`，retain `3580`
- 远程评分失败/解析失败：`0`

| metric | CHIP full 0526 | restored full baseline | delta |
|---|---:|---:|---:|
| forget classification remote acc | 69.68% | 55.32% | +14.36 pp |
| retain classification remote acc | 70.36% | 56.79% | +13.58 pp |
| forget generation remote acc | 44.68% | 1.06% | +43.62 pp |
| retain generation remote acc | 43.02% | 0.11% | +42.91 pp |
| forget generation BLEU | 24.03% | 5.27% | +18.76 pp |
| forget generation ROUGE-L | 42.81% | 19.62% | +23.19 pp |
| retain generation BLEU | 24.09% | 4.78% | +19.31 pp |
| retain generation ROUGE-L | 42.52% | 18.96% | +23.56 pp |

远程评分文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/clf_forgetset_multi_preds_remote_scored.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/clf_retainset_multi_preds_remote_scored.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/gen_forgetset_multi_preds_remote_scored.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/full_clear_remote_eval/chip_full_train_0526_1348_full_remote/gen_retainset_multi_preds_remote_scored.json
```

Full CLEAR 远程评分结论：

- forget classification 和 forget generation 都显著高于 restored full baseline。
- 对遗忘任务来说，accuracy / BLEU / ROUGE-L 越高，说明模型越能继续答出目标知识；因此这不是成功遗忘，而是明显保留了 forget-set 知识。
- retain 指标也同步变高，说明模型整体生成/识别能力较强，但没有形成 forget/retain 的有效分离。
- 这与 Step8 的 `forget_clean name_hit_rate` 未下降一致：当前 causal masked RMisU 配置没有把目标人物身份/图文关联有效擦除。

### 5.2 与原 0430 结果的关系

原 `0430_141446` baseline 的主评估结论是：

- forget multi classification：`22.34% -> 10.11%`
- forget multi generation acc：`56.38% -> 10.64%`
- retain multi classification：`23.38% -> 21.40%`
- retain multi generation acc：`55.00% -> 57.37%`

该 baseline 在原主评估协议下表现为明显遗忘且 retain 基本稳定。

需要注意：上面 `0430_141446` 文档中的百分比来自原始实验日志；本次补充的 Full CLEAR 结果使用可恢复脚本重新跑远程 judge，并以 `0525_compare_0430_remote_score_restored_full_protocol.json` 作为直接可比基线。因此结论优先看 `5.1` 的 restored full protocol 对比。

### 5.3 Step8 固定协议结果

本次 CHIP full 0526 的 Step8 固定协议结果显示：

- `forget_clean name_hit_rate` 没有低于 PEFT baseline：`0.6111 -> 0.6111`
- `hard_retain name_hit_rate` 略高：`0.6944 -> 0.7222`
- `counterfactual_retain name_hit_rate` 下降：`0.7222 -> 0.6667`
- BLEU / ROUGE-L 在 forget_clean 上基本持平，未体现明显遗忘

因此，本次 causal masked RMisU 完整训练可以认为：

```text
训练链路已完整跑通，但 Step8 和 Full CLEAR 远程评分都未证明成功遗忘。
```

它没有复现原 baseline 那种明显的 forget 下降；Full CLEAR 远程评分中 forget accuracy 反而偏高，同时 Step8 counterfactual retain 有轻微损伤。

## 6. 注意事项

1. `main.py --eval_flag --use_remote_scoring` 的全量远程评分不稳定。本次最终使用可恢复脚本完成全量远程评分，避免单条远程请求卡死整个流程。
2. Step8 的 `name_hit_rate` 与 Full CLEAR `acc_by_llm` 不是同一个指标，不能逐点等价比较。
3. 在 Step8 固定协议内，本次结果与 PEFT baseline 对比是直接可比的；在 Full CLEAR 协议内，本次结果与 restored full baseline 直接可比。
4. 当前最关键的问题仍是：masked RMisU 没有让 `forget_clean` 的身份泄露率下降，也没有让 Full CLEAR forget-set remote acc 下降。

## 7. 结论

本次实验完整产出了 checkpoint、Step8 评估结果和 Full CLEAR 全量远程评分结果。工程链路是成功的，但方法效果未超过 PEFT/restored baseline，也明显弱于原 `0430_141446` baseline 在主评估协议中的遗忘表现。
