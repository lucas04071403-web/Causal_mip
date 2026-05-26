# Step 8：最终评估实现与实验记录

**更新时间**: 2026-05-22
**状态**: MVP 已实现，Step7 checkpoint 与 PEFT baseline 的 val 集评估已跑通

---

## 1. 目标

Step 8 用于验证 Step7 edited model 是否同时满足：

- 删除目标知识
- 保留 retain knowledge
- 在 hard retain / counterfactual retain 上保持稳定

当前 MVP 基于 Step3 生成的 pair 文件评估：

```text
forget_clean
hard_retain
counterfactual_retain
```

其中 `hard_retain` 进一步包含：

```text
same_topic
same_reasoning
```

---

## 2. 新增实现

新增文件：

```text
causal_mip/evaluation/__init__.py
causal_mip/evaluation/step8_final_eval.py
```

评估入口支持：

```text
--model_path
--peft_checkpoint
--pair_jsonl
--output
--split
--device
--image_resize
--max_per_set
--max_new_tokens
```

设计原则：

- 直接加载 Step7 保存的 full checkpoint
- 可选加载 base model + PEFT adapter，并 merge 后作为公平 baseline
- 从 Step3 pair JSONL 的 `image_ref.dataset_path + row_idx` 回读图像
- 不默认加载额外 LLM scorer，避免 24GB GPU 上评估 OOM
- 输出 raw predictions 和 summary metrics

---

## 3. 当前指标

当前 Step8 MVP 使用轻量本地指标：

```text
name_hit_rate:
  预测中是否包含样本 name

BLEU / ROUGE:
  预测文本与 pair 中 answer/caption 的文本相似度
```

解释：

- 对 `forget_clean`，`name_hit_rate` 越低越好，表示目标身份泄露更少
- 对 `hard_retain.same_topic`，`name_hit_rate` 不一定越高越好，因为问题要求不识别人物
- 对 `hard_retain.same_reasoning` 和 `counterfactual_retain`，`name_hit_rate` 越高通常表示 retain 身份知识保留更好
- BLEU / ROUGE 用于粗略观察生成质量，不替代 LLM judge

---

## 4. 已运行命令

### 4.1 Step7 checkpoint smoke

```bash
conda run -n mip-editor python causal_mip/evaluation/step8_final_eval.py \
  --model_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521 \
  --pair_jsonl /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_per_set 1 \
  --max_new_tokens 40 \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_smoke_val_max1_0522.json
```

结果：

```text
num_examples=4
forget_clean=1
hard_retain=2
counterfactual_retain=1
```

说明：Step3 pair -> image_ref 回读 -> Step7 checkpoint 加载 -> generation -> metrics -> JSON 保存链路已跑通。

### 4.2 Step7 checkpoint val full

```bash
conda run -n mip-editor python causal_mip/evaluation/step8_final_eval.py \
  --model_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521 \
  --pair_jsonl /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80 \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_val_full_0522.json
```

### 4.3 PEFT baseline val full

公平 baseline 使用 Step7 训练前的 PEFT cache：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
```

命令：

```bash
conda run -n mip-editor python causal_mip/evaluation/step8_final_eval.py \
  --model_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/llms/Qwen2.5-VL-3B-Instruct \
  --peft_checkpoint /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth \
  --pair_jsonl /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl \
  --split val \
  --device cuda \
  --image_resize 224 \
  --max_new_tokens 80 \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_peft_baseline_val_full_0522.json
```

---

## 5. Val 集结果

评估集：

```text
causal_pairs_val.jsonl = 18 pairs
total examples = 72
forget_clean = 18
hard_retain = 36
counterfactual_retain = 18
```

### 5.1 by eval_set

| model | eval_set | n | name_hit_rate | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| PEFT baseline | forget_clean | 18 | 0.6111 | 0.2090 | 0.4015 |
| Step7 masked RMisU | forget_clean | 18 | 0.7778 | 0.2249 | 0.4383 |
| PEFT baseline | hard_retain | 36 | 0.6944 | 0.1591 | 0.3519 |
| Step7 masked RMisU | hard_retain | 36 | 0.7222 | 0.1619 | 0.3484 |
| PEFT baseline | counterfactual_retain | 18 | 0.7222 | 0.2167 | 0.3858 |
| Step7 masked RMisU | counterfactual_retain | 18 | 0.7222 | 0.2053 | 0.3662 |

### 5.2 by sample_type

| model | sample_type | n | name_hit_rate | ROUGE-L |
|---|---:|---:|---:|---:|
| PEFT baseline | hard_retain::same_topic | 18 | 0.6111 | 0.3138 |
| Step7 masked RMisU | hard_retain::same_topic | 18 | 0.6667 | 0.3151 |
| PEFT baseline | hard_retain::same_reasoning | 18 | 0.7778 | 0.3860 |
| Step7 masked RMisU | hard_retain::same_reasoning | 18 | 0.7778 | 0.3835 |

---

## 6. 当前结论

当前 Step7 checkpoint 没有达成预期遗忘。

关键证据：

```text
forget_clean name_hit_rate:
PEFT baseline       = 0.6111
Step7 masked RMisU  = 0.7778
```

也就是说，经过当前 Step7 masked RMisU 后，目标身份名字泄露率没有下降，反而上升。

retain 侧：

```text
counterfactual_retain name_hit_rate:
PEFT baseline       = 0.7222
Step7 masked RMisU  = 0.7222

hard_retain name_hit_rate:
PEFT baseline       = 0.6944
Step7 masked RMisU  = 0.7222
```

retain 没有明显崩坏，但这不能抵消 forget 失败。

---

## 7. 风险与解释

当前负结果与 Step7 训练日志一致：

```text
retain_loss: 0.00384521484375 -> 1.296875
shared_loss: 0.0030059814453125 -> 0.00482177734375
```

说明 preserve 约束有压力，但真正的问题是 `forget_clean` 上的名字生成没有被压制。

可能原因：

- 当前 Step7 只编辑语言侧 `up_proj/gate_proj` 的少量 FFN intermediate neurons
- Step5 的 `Suf(P)` 长期为 0，Step6 路径分类主要由 Nec/Ret 主导
- RMisU random-direction loss 只作用 selected activation，不直接约束输出中的目标 name token
- 当前 `P_forget` 可能没有覆盖真正决定身份输出的视觉/跨模态路径

---

## 8. 下一步建议

进入下一轮优化，而不是把当前结果当成成功版本。

优先级：

1. 跑 `step6_v1_quantile75` 的保守 mask 对照，降低误编辑和 retain 压力
2. 在 Step7 loss 中加入输出层面的 forget 约束，例如目标 name token NLL 反向项或拒答目标
3. 扩展 Step4/5/7 到 vision/projector 路径，否则身份知识很可能仍由视觉侧触发
4. 增加 LLM judge 评估，但应使用远程评分或分进程 CPU/GPU 释放，避免与 3B 被评估模型同时占满显存

---

## 9. Step6/Step7 优化后评估

基于初评负结果，继续完成两组优化实验：

```text
Step6: step6_v1_quantile75
Step7: step6_v1_quantile75 + forget CE alpha=0.1
```

新增 Step7 参数：

```text
--masked_rmisu_forget_ce_alpha
```

### 9.1 输出文件

Step7 quantile75：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_quantile75_full1epoch_0522.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_quantile75_full1epoch_0522
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_quantile75_val_full_0522.json
```

Step7 quantile75 + CE：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_quantile75_ce01_full1epoch_0522.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_quantile75_ce01_full1epoch_0522
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_quantile75_ce01_val_full_0522.json
```

### 9.2 对比结果

| model | forget name hit | hard retain name hit | counterfactual retain name hit | forget ROUGE-L |
|---|---:|---:|---:|---:|
| PEFT baseline | 0.6111 | 0.6944 | 0.7222 | 0.4015 |
| Step7 step6_v1 | 0.7778 | 0.7222 | 0.7222 | 0.4383 |
| Step7 quantile75 | 0.6667 | 0.7222 | 0.7222 | 0.4189 |
| Step7 quantile75 + CE 0.1 | 0.6111 | 0.7222 | 0.7222 | 0.4096 |

### 9.3 优化结论

`step6_v1_quantile75` 修复了一部分 Step7 过编辑问题：

```text
forget_clean name_hit_rate:
step6_v1      0.7778
quantile75    0.6667
```

加入输出层 forget CE 后进一步改善：

```text
quantile75 + CE 0.1 = 0.6111
```

但这个结果只是回到 PEFT baseline：

```text
PEFT baseline = 0.6111
```

因此当前最佳版本是：

```text
masked_rmisu_step7_quantile75_ce01_full1epoch_0522
```

但仍不能声明成功遗忘。下一轮应继续提高输出层 forget 约束或扩展视觉/projector 路径。
