# Step 7：masked RMisU MVP 实现记录

**更新时间**: 2026-05-22
**状态**: MVP 已实现；语言侧真实训练已跑通；mask / parameter wrapper 已扩展到 Qwen2.5-VL visual block，等待视觉侧 Step5/Step6 路径集合后做真实视觉训练

---

## 1. 目标

Step 7 的目标是把 Step 6 得到的路径集合接入编辑训练：

- `P_forget`：优先执行 random-direction misdirection
- `P_shared`：作为共享机制进行 preserve / freeze
- `P_retain`：第一版不编辑
- `P_irrelevant`：第一版不编辑

当前支持从 Step6 路径集合构建语言侧和 Qwen2.5-VL visual block 的 neuron mask：

```text
model.language_model.layers.{j}.mlp.down_proj
model.visual.blocks.{j}.mlp.down_proj
```

`mm_projector` 和 attention path 当前仍不纳入 masked RMisU。

---

## 2. 当前实现文件

新增模块：

```text
causal_mip/editing/__init__.py
causal_mip/editing/masked_rmisu.py
causal_mip/test_step7_masked_rmisu.py
```

核心对象：

```python
PathNeuronMask
MaskedRMisUConfig
build_path_neuron_masks(...)
apply_masked_rmisu_parameter_mask(...)
masked_rmisu_finetune(...)
```

---

## 3. 实现逻辑

### 3.1 从 Step6 路径集合构建 neuron mask

输入：

```text
P_cand.jsonl
step6_v1/P_forget.jsonl
step6_v1/P_shared.jsonl
```

流程：

1. 读取 `P_forget` 和 `P_shared` 中的 `path_id`
2. 回到 `P_cand.jsonl` 查找完整 `CandidatePath`
3. 只保留 patchable `mlp.down_proj` 节点：
   - `model.language_model.layers.{j}.mlp.down_proj`
   - `model.visual.blocks.{j}.mlp.down_proj`
4. 按 module 聚合 neuron：

```text
forget_neurons[module]
shared_neurons[module]
editable_neurons = forget_neurons - shared_neurons
preserve_neurons = forget_neurons union shared_neurons
```

这样可以保证：

- 被 `P_shared` 标记的 neuron 不会进入可训练集合
- `P_forget` 中与 `P_shared` 重叠的 neuron 会被保护

### 3.2 参数级 mask

Step 4/5 中的 neuron 是 `down_proj` 输入侧 FFN intermediate neuron。

因此 Step 7 对应的可训练参数不是 `down_proj` 输出维度，而是同一 MLP 中：

```text
up_proj output column
gate_proj output column
```

当前实现复用项目已有的：

```text
PartialLinear
```

把每层 MLP 的 `up_proj` 和 `gate_proj` 包装成只训练 `editable_neurons` 的部分线性层。

其他参数默认冻结。

### 3.3 masked RMisU loss

MVP 训练函数：

```python
masked_rmisu_finetune(...)
```

包含三类 loss：

```text
unlearn_loss:
  在 forget batch 上，把 editable neuron activation 推向 random direction

retain_loss:
  在 retain batch 上，让 updated model 的 preserve neuron activation 接近 frozen model

shared_loss:
  在 retain batch 上，额外保护 shared neuron activation
```

总 loss：

```text
loss = beta * unlearn_loss + alpha * retain_loss + shared_alpha * shared_loss
```

---

## 4. 已验证结果

### 4.1 toy 单元测试

命令：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
conda run -n mip-editor python causal_mip/test_step7_masked_rmisu.py
```

结果：

```text
Step 7 masked RMisU tests passed.
```

测试覆盖：

- `P_forget/P_shared` path id 读取
- `P_cand` 回表解析
- `editable_neurons = forget - shared`
- `PartialLinear` 包装 `up_proj/gate_proj`
- toy dataloader 上完整跑通 `masked_rmisu_finetune`

### 4.2 真实 Step6 mask 构建检查

使用真实路径集合：

```text
P_cand:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl

P_forget:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/P_forget.jsonl

P_shared:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/P_shared.jsonl
```

检查结果：

```text
modules=36
forget_neuron_refs=203
shared_neuron_refs=114
editable_neuron_refs=123
```

说明：

- `step6_v1` 可以完整映射回 Step2 的候选路径
- 36 个语言层都有可识别的 mask 信息
- 扣除 `P_shared` 重叠后，当前有 123 个 editable neuron 引用可供 Step7 训练使用

---

## 5. 当前边界

### 5.1 主训练入口接入状态

当前 Step7 已通过显式参数接入 `main.py` / `ours.py`：

```text
--use_masked_rmisu
--masked_rmisu_candidate_paths
--masked_rmisu_p_forget
--masked_rmisu_p_shared
--masked_rmisu_shared_alpha
--masked_rmisu_output
--masked_rmisu_max_steps
--skip_post_unlearning_eval
```

默认不启用 masked RMisU，因此不会影响原始 MIP-Editor baseline 复现。

### 5.2 当前支持范围

当前处理：

```text
model.language_model.layers.{j}.mlp.down_proj
model.visual.blocks.{j}.mlp.down_proj
```

不处理：

- `mm_projector`
- attention head

### 5.3 当前 preserve loss 是 activation-level MVP

当前 preserve loss 只约束 `down_proj` 输入侧 selected neurons 的 activation MSE。

后续可扩展为：

- retain KL
- full hidden-state preserve
- shared path token-level preserve

---

## 6. 下一步

当前 Step7 已完成真实训练入口 smoke test 和 1 epoch 小实验。下一步建议进入 Step8 评估，对比：

```text
baseline MIP-Editor
原始 RMisU
masked RMisU step6_v1
masked RMisU step6_v1_quantile75
```

如果 retain 损伤过大，则优先尝试 `step6_v1_quantile75`、提高 `masked_rmisu_shared_alpha`，或降低 `rmu_beta` / learning rate。

---

## 7. 主训练入口接入与真实 smoke test

### 7.1 接入位置

`main.py` 新增参数：

```text
--use_masked_rmisu
--masked_rmisu_candidate_paths
--masked_rmisu_p_forget
--masked_rmisu_p_shared
--masked_rmisu_shared_alpha
--masked_rmisu_output
--masked_rmisu_max_steps
--skip_post_unlearning_eval
```

`ours.py` 中的 `our(...)` 现在会根据：

```text
args.use_masked_rmisu
```

选择：

```text
原始 adaptive_rmu_finetune
或
Step7 masked_rmisu_finetune
```

### 7.2 真实 smoke test 命令

为了验证真实训练闭环，同时避免完整 1790 step 训练耗时过长，使用：

```text
--masked_rmisu_max_steps 2
--skip_post_unlearning_eval
--rmu_alpha 0.0
--masked_rmisu_shared_alpha 0.0
```

命令：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python main.py \
  --unlearning our \
  --use_masked_rmisu \
  --dataset clear \
  --forget_ratio 5 \
  --model Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --batch_size 2 \
  --ptm_ckpt_batch_size 2 \
  --epochs 1 \
  --finetune_epochs 1 \
  --learning_rate 1e-5 \
  --rmu_alpha 0.0 \
  --rmu_beta 1.0 \
  --masked_rmisu_shared_alpha 0.0 \
  --rmu_steering_coeff 1.0 \
  --rmu_coeffs 10.0 \
  --use_neuron_cache_flag \
  --skip_post_unlearning_eval \
  --masked_rmisu_max_steps 2 \
  --this_run_id step7_smoke_0521 \
  --masked_rmisu_candidate_paths /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl \
  --masked_rmisu_p_forget /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/P_forget.jsonl \
  --masked_rmisu_p_shared /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/P_shared.jsonl \
  --masked_rmisu_output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_smoke_0521.json
```

### 7.3 smoke test 结果

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_smoke_0521.json
```

结果：

```text
num_loss_records=2
num_modules=34
num_editable_neurons=123
loss step 0 = 74.0
loss step 1 = 70.0
```

说明：

- 真实 Qwen2.5-VL-3B checkpoint 可以加载
- `step6_v1/P_forget` 和 `step6_v1/P_shared` 可以被主训练入口读取
- `PartialLinear` 可以替换真实模型语言侧 `up_proj/gate_proj`
- masked RMisU 可以执行真实 forward/backward/optimizer step
- `--skip_post_unlearning_eval` 可以避免 smoke test 自动进入完整评估

### 7.4.1 视觉侧 mask 支持验证

在代码层面，Step7 已支持视觉 block：

- `build_path_neuron_masks(...)` 会识别 `model.visual.blocks.{j}.mlp.down_proj`
- `apply_masked_rmisu_parameter_mask(...)` 会包装对应 visual MLP 的 `up_proj/gate_proj`
- `DownProjInputTracer` 可追踪视觉侧 2D activation 的 selected intermediate neurons
- `causal_mip/test_step7_masked_rmisu.py` 已覆盖视觉 mask / parameter wrap

真实候选路径检查：

```text
vision_fisher_p000000 -> num_modules=32, module_kind_counts={'vision': 32}, editable_by_kind={'vision': 32}
```

当前生产用 `step6_v1_quantile75` 仍只包含语言侧路径：

```text
num_modules=36, module_kind_counts={'llm': 36}, editable_by_kind={'llm': 45}
```

因此下一步需要用视觉侧 Step5 评分重跑 Step6，生成包含 `vision` / `vision_text` 的稳定路径集合后，再做真实视觉 masked RMisU 训练。

### 7.5 preserve smoke test 结果

在开启 retain/shared preserve 的情况下，已完成真实 2-step smoke test：

```text
--rmu_alpha 0.5
--rmu_beta 0.5
--masked_rmisu_shared_alpha 1.0
--masked_rmisu_max_steps 2
--skip_post_unlearning_eval
```

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_preserve_smoke_0521.json
```

结果：

```text
num_loss_records=2
num_modules=34
num_editable_neurons=123

step 0:
  loss=37.0
  unlearn_loss=74.0
  retain_loss=0.00384521484375
  shared_loss=0.0030059814453125

step 1:
  loss=35.0
  unlearn_loss=70.0
  retain_loss=0.0057373046875
  shared_loss=0.00341796875
```

说明：

- preserve 分支可以在真实 Qwen2.5-VL-3B 训练入口中正常执行
- retain/shared activation MSE loss 非零且可记录
- loss 主要仍由 unlearn loss 主导，符合当前系数设置

### 7.5 1 epoch masked RMisU 小实验

已在真实数据上完成完整 `finetune_epochs=1` masked RMisU 小实验。该次运行使用 preserve 分支，并通过：

```text
--skip_post_unlearning_eval
```

跳过训练后的完整评估，只验证 Step7 训练闭环。

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_full1epoch_0521.json
```

结果：

```text
num_loss_records=1790
num_modules=34
num_editable_neurons=123
loss_mean=31.31417597765363

first step:
  epoch=0
  step=0
  loss=37.0
  unlearn_loss=74.0
  retain_loss=0.00384521484375
  shared_loss=0.0030059814453125

last step:
  epoch=0
  step=1789
  loss=30.5
  unlearn_loss=59.75
  retain_loss=1.28125
  shared_loss=0.0045166015625
```

结论：

- Step7 已能在真实 Qwen2.5-VL-3B + CLEAR forget_ratio=5 上完成 1 epoch masked RMisU 训练
- 当前 `step6_v1` mask 实际开放 34 个 module 中的 123 个 editable neuron 引用
- 训练 loss 从首步 37.0 降至末步 30.5
- retain preserve loss 在长训练中上升到 1.28125，后续 Step8 需要重点检查 retain 任务表现
- 当前输出是训练 summary JSON；如需做完整模型评估，应先确认 Step7 是否已保存可加载 checkpoint，必要时补充 `save_pretrained` 逻辑后重跑

### 7.6 checkpoint 保存版 1 epoch 小实验

已补充 Step7 checkpoint 保存逻辑：

- `--masked_rmisu_checkpoint_dir` 指定保存目录
- 保存前将 `PartialLinear` 合并回普通 `nn.Linear`
- 保存 edited model 的 `safetensors`
- 保存 processor / tokenizer 文件
- summary JSON 中记录 `checkpoint_dir` 和 `merged_partial_linear_modules`

重跑命令使用与 7.5 相同的 preserve 配置，并新增：

```text
--this_run_id step7_full1epoch_ckpt_0521
--masked_rmisu_output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_full1epoch_ckpt_0521.json
--masked_rmisu_checkpoint_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521
```

输出：

```text
summary:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_full1epoch_ckpt_0521.json

checkpoint:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521
```

checkpoint 目录大小：

```text
7.1G
```

关键文件：

```text
config.json
generation_config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
```

训练结果：

```text
num_loss_records=1790
num_modules=34
num_editable_neurons=123
merged_partial_linear_modules=68
loss_mean=31.316550279329608

first step:
  loss=37.0
  unlearn_loss=74.0
  retain_loss=0.00384521484375
  shared_loss=0.0030059814453125

last step:
  loss=30.5
  unlearn_loss=59.75
  retain_loss=1.296875
  shared_loss=0.00482177734375
```

可加载性检查：

```text
AutoProcessor.from_pretrained(checkpoint)
Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)
```

结果：

```text
Qwen2_5_VLProcessor
Qwen2_5_VLForConditionalGeneration
['Qwen2_5_VLForConditionalGeneration']
```

结论：Step7 当前已经产出可被 Step8 直接加载的 edited model checkpoint。本轮按要求未启动 Step8 评估。

### 7.7 Step6/Step7 优化：quantile75 mask 与 forget CE

Step8 初评发现 `step6_v1` 的 Step7 checkpoint 没有降低 `forget_clean` 的目标名字泄露率。

因此进行了两轮优化：

1. 使用 Step6 保守路径集合 `step6_v1_quantile75`
2. 在 Step7 中新增输出层 forget CE 反向项

新增参数：

```text
--masked_rmisu_forget_ce_alpha
```

训练 loss 更新为：

```text
loss =
  beta * unlearn_loss
  + alpha * retain_loss
  + shared_alpha * shared_loss
  - forget_ce_alpha * forget_ce_loss
```

其中 `forget_ce_loss` 是 forget batch 上原始 label 的 CE loss。负号表示对目标答案做梯度上升，降低模型继续生成目标答案的概率。

#### quantile75 mask 统计

```text
P_forget=8
P_shared=3
P_retain=12
P_irrelevant=35

modules=36
forget_refs=84
shared_refs=45
editable_refs=45
modules_with_editable=18
```

#### quantile75 1 epoch

输出：

```text
summary:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_quantile75_full1epoch_0522.json

checkpoint:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_quantile75_full1epoch_0522
```

训练结果：

```text
num_loss_records=1790
num_modules=18
num_editable_neurons=45
merged_partial_linear_modules=36
loss_mean=35.47444134078212
first loss=34.75
last loss=37.25
```

Step8 val 结果：

```text
forget_clean name_hit_rate=0.6667
hard_retain name_hit_rate=0.7222
counterfactual_retain name_hit_rate=0.7222
```

#### quantile75 + forget CE alpha=0.1

输出：

```text
summary:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_quantile75_ce01_full1epoch_0522.json

checkpoint:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_quantile75_ce01_full1epoch_0522
```

训练结果：

```text
num_loss_records=1790
num_modules=18
num_editable_neurons=45
merged_partial_linear_modules=36
loss_mean=35.034265778184604
forget_ce_mean=4.3961696289105126

first step:
  loss=34.316627502441406
  unlearn_loss=69.5
  forget_ce_loss=4.333711624145508
  retain_loss=0.004730224609375
  shared_loss=0.004791259765625

last step:
  loss=36.824607849121094
  unlearn_loss=74.5
  forget_ce_loss=4.253909111022949
  retain_loss=0.02294921875
  shared_loss=0.004791259765625
```

Step8 val 结果：

```text
forget_clean name_hit_rate=0.6111
hard_retain name_hit_rate=0.7222
counterfactual_retain name_hit_rate=0.7222
```

当前结论：

- `step6_v1_quantile75` 明显优于原 `step6_v1`
- `quantile75 + forget CE 0.1` 是当前最佳版本
- 但它只是把 `forget_clean name_hit_rate` 拉回 PEFT baseline 的 0.6111，仍未低于 baseline，因此还不能称为成功遗忘
