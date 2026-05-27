# CHIP-Editor 技术路线图

## 0. 目标

基于当前 `MIP-Editor` 项目，围绕以下三个科学问题，设计一条**从前往后、逻辑严谨、可逐步实现**的技术路线：

1. 在多模态大模型中，哪些路径是真正承载目标知识的因果路径，如何区分“有害意图路径”和“良性主题路径”？
2. 如何定位跨模态因果路径中的有害知识？
3. 如何在删除目标知识时避免破坏 retain knowledge？

---

## 1. 总体逻辑

这三个问题不能并列、同时展开，而要按因果链路依次推进：

```text
baseline MIP-Editor
→ top-k candidate paths
→ clean/corrupt/hard retain pairs
→ activation cache + path patching
→ necessity / sufficiency / retain impact
→ P_forget / P_shared 分类
→ masked RMisU
→ forget / retain / hard retain / counterfactual retain 评估
```

这条链路对应的科学逻辑是：

### 第一层：先找到候选路径

先回答：

> 哪些路径可能承载目标知识？

这一步只能得到：

- `candidate influential paths`

还不能直接得到：

- `causal paths`

### 第二层：再证明哪些是因果路径

通过反事实样本和 activation patching，回答：

> 这些候选路径中，哪些路径对目标知识具有必要性和充分性？

这一步得到：

- `causal paths`

### 第三层：再区分哪些该删，哪些该保

结合 retain impact，回答：

> 哪些路径是真正的 harmful-intent path，哪些只是 benign-topic / shared-utility path？

这一步得到：

- `P_forget`
- `P_shared`

### 第四层：最后做选择性编辑

只编辑 `P_forget`，保护 `P_shared`，从而回答：

> 如何在删除目标知识时避免破坏 retain knowledge？

---

## 2. 对三个科学问题的重新对应

### 科学问题 1

> 在多模态大模型中，哪些路径是真正承载目标知识的因果路径，如何区分“有害意图路径”和“良性主题路径”？

它不是一开始就能回答，而是要通过以下三步得到：

1. 导出 `P_cand`
2. 计算 `Nec/Suf/Ret`
3. 分类 `P_forget / P_shared`

### 科学问题 2

> 如何定位跨模态因果路径中的有害知识？

它依赖：

1. 文本路径、视觉路径、cross-modal bridge path 的候选导出
2. 对这些路径做 causal patching 验证

### 科学问题 3

> 如何在删除目标知识时避免破坏 retain knowledge？

它依赖：

1. 先把 `P_forget` 和 `P_shared` 区分出来
2. 再做 masked RMisU
3. 再用 retain / hard retain / counterfactual retain 验证

---

## 3. 当前项目可直接复用的代码仓库

## 3.1 当前主项目：`MIP-Editor`

作用：

- baseline 主工程
- 本地 `Qwen2.5-VL-3B` 训练
- 远程 `Qwen2.5-VL-7B` 评分
- `IGI / IFI / RMisU / path editing`
- top-k candidate path 导出

可直接复用的位置：

- `main.py`
- `ours.py`
- `train_eval.py`
- `ig.py`
- `fisher.py`
- `partial_linear.py`
- `causal_mip/path_localization/`

在路线中的作用：

- 提供 baseline
- 提供 path candidates
- 提供现成的 retain-aware unlearning 基座

---

## 3.2 可借鉴仓库 1：`NOTICE`

仓库作用：

- clean / corrupt / retain 的反事实样本构造思想

适合借鉴的部分：

- pair 数据组织方式
- same-topic retain 构造方式
- same-reasoning retain 构造方式
- counterfactual retain 设计思路

在路线中的作用：

- 用于 Step 2 的 `causal_pairs` 构造

不建议做的事：

- 不必整仓照搬
- 只借数据构造思想即可

---

## 3.3 可借鉴仓库 2：`ROME`

仓库作用：

- causal tracing / activation restoration 模板

适合借鉴的部分：

- clean run / corrupt run / restored run
- necessity / sufficiency 风格的验证思路
- activation cache 和 patching 的工程组织方式

在路线中的作用：

- 用于 Step 3 的 activation cache + path patching
- 用于 Step 4 的 `Nec/Suf/Ret` 计算

不建议做的事：

- 不必直接复用 ROME 的模型编辑方法
- 只借 intervention / tracing 机制

---

## 3.4 可借鉴仓库 3：`pyvene`

仓库作用：

- intervention 的工程后端

建议：

- MVP 第一版不要强依赖
- 第一版先用手写 `forward hooks`
- 后续如果 patching 稳定，再封装为 `pyvene backend`

在路线中的作用：

- 作为第二阶段工程化参考

---

## 4. 从前往后的技术路线

## Step 1：固定 baseline MIP-Editor

### 目标

先把当前项目作为稳定基座固定下来。

### 直接复用的代码

- `main.py`
- `ours.py`
- `train_eval.py`
- `ig.py`
- `fisher.py`
- `load_model.py`
- `load_data.py`

### 需要得到的产物

- baseline checkpoint
- baseline forget / retain 评估结果

### 目的

这一阶段不回答新的科学问题，只做一件事：

> 提供一个可复现、可对照的 baseline。

---

## Step 2：导出 top-k candidate paths

### 目标

把原始 `MIP-Editor` 的 greedy path 思路升级成：

```text
top-k / candidate path set
```

### 直接复用的代码

- `causal_mip/path_localization/cached_path_export.py`
- `causal_mip/path_localization/path_schema.py`
- `causal_mip/path_localization/mip_topk_wrapper.py`
- `causal_mip/path_localization/beam_search_paths.py`
- `causal_mip/path_localization/cross_modal_path_builder.py`
- `main.py --export_candidate_paths_only`

### 输出文件

```text
outputs/paths/P_cand.jsonl
```

### 当前能回答什么

这一阶段只能回答：

> 哪些路径是潜在的重要候选路径？

还不能回答：

- 哪些路径是因果路径
- 哪些路径是 harmful-intent path

### 对应科学问题

这一步是问题 2 的前置基础。

---

## Step 3：构造反事实 pair 数据

### 目标

为 causal validation 提供统一输入。

### 新增模块建议

```text
causal_mip/data_pairs/build_pairs.py
causal_mip/data_pairs/text_corruption.py
causal_mip/data_pairs/image_corruption.py
causal_mip/data_pairs/hard_retain_builder.py
```

### 可借鉴代码仓库

- `NOTICE`

### 最小数据格式

建议至少包含：

- `forget_clean`
- `forget_corrupt`
- `hard_retain`
- `counterfactual_retain`

### 输出文件

```text
outputs/data_pairs/causal_pairs_train.jsonl
outputs/data_pairs/causal_pairs_val.jsonl
```

### 这一阶段的逻辑意义

这是整条路线的关键转折点。

没有反事实 pair，只能做：

- influential path

不能严谨回答：

- harmful intent 和 benign topic 的区分

### 对应科学问题

这是问题 1 的必要前置条件。

---

## Step 4：实现 activation cache + path patching

### 目标

把候选路径变成可干预、可验证的路径。

### 新增模块建议

```text
causal_mip/interventions/hooks.py
causal_mip/interventions/activation_cache.py
causal_mip/interventions/patching.py
causal_mip/interventions/ablation.py
causal_mip/interventions/restoration.py
```

### 可借鉴代码仓库

- `ROME`

### MVP 第一版建议

第一版只 hook：

```text
language_model.layers[j].mlp
```

不要一开始就同时做：

- `vision_encoder.blocks[i].mlp`
- `mm_projector`
- self-attention patching

### 第一版 metric

优先只用：

```text
target answer log-prob
```

不要一开始就用：

- free-form generation
- Rouge-L
- 复杂 judge 评分

### 这一阶段的逻辑意义

这一步是从：

```text
candidate path
```

升级到：

```text
causal testable path
```

的关键工程层。

### 对应科学问题

这一步同时服务问题 1 和问题 2。

---

## Step 5：计算 necessity / sufficiency / retain impact

### 目标

给每条候选路径建立因果分数。

### 新增模块建议

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/necessity.py
causal_mip/causal_scores/sufficiency.py
causal_mip/causal_scores/retain_impact.py
```

### 统一度量

MVP 第一版建议统一用：

```text
target answer log-prob
```

### 三个核心量

#### `Nec(P)`

含义：

> 删除路径后，forget 目标是否明显下降？

#### `Suf(P)`

含义：

> 在 corrupted 输入中恢复路径后，forget 目标是否明显回来？

#### `Ret(P)`

含义：

> 删除路径后，retain 是否明显受损？

### 输出文件

```text
outputs/scores/path_causal_scores.jsonl
```

### 这一阶段的逻辑意义

当你拿到 `Nec/Suf/Ret` 时，才开始有资格回答：

> 哪些路径是真正承载目标知识的因果路径？

### 对应科学问题

这一步直接回答问题 1 和问题 2 的核心部分。

---

## Step 6：分类 `P_forget / P_shared / P_retain / P_irrelevant`

### 目标

把“有因果作用的路径”进一步区分成“该删的”和“该保的”。

### 新增模块建议

```text
causal_mip/causal_scores/classify_paths.py
```

### 当前实现状态

已实现：

```text
causal_mip/causal_scores/classify_paths.py
causal_mip/test_step6_classify_paths.py
```

Step 6 当前读取 Step 5 的 `path_scores*.jsonl`，先按 `path_id` 聚合多个 pair 上的分数，再对聚合后的 path 做四分类。

为什么先聚合：

- Step 5 的 record 粒度是 `pair_id + path_id`
- Step 6 的输出需要是 path set
- 同一条 path 在多个 pair 上可能有不同 score，因此需要先聚合成 path-level score

当前支持：

- 聚合方式：`mean` / `median`
- 阈值方式：显式阈值或分位数阈值
- 负向 effect 处理：默认把负值 clip 到 0，避免“ablation 后目标更强”被当成 forget 证据
- 输出：
  - `P_forget.jsonl`
  - `P_shared.jsonl`
  - `P_retain.jsonl`
  - `P_irrelevant.jsonl`
  - `P_classified.jsonl`
  - `classification_summary.json`

### MVP 第一版推荐规则

定义：

```text
forget_effect = Nec + α * Suf
```

再结合：

```text
retain_impact = Ret
```

做四类划分：

- `P_forget`
- `P_shared`
- `P_retain`
- `P_irrelevant`

### 最短分类逻辑

#### `P_forget`

- `forget_effect` 高
- `retain_impact` 低

#### `P_shared`

- `forget_effect` 高
- `retain_impact` 高

#### `P_retain`

- `forget_effect` 低
- `retain_impact` 高

#### `P_irrelevant`

- `forget_effect` 低
- `retain_impact` 低

### 当前实现中的实际打分

为了避免负向干预效果污染分类，当前默认使用：

```text
forget_effect = max(0, Nec) + α * max(0, Suf)
retain_impact = max(0, Ret)
```

如果要保留 signed effect，可以在 CLI 中加入：

```text
--use_signed_effects
```

### 运行命令

单独分类 text 5x5：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_text_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

单独分类 vision_text 5x5：

```bash
conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_vision_text_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

合并分类 text + vision_text：

```bash
conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_combined_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

### 已验证结果

单元测试：

```text
Step 6 path-classification tests passed.
```

真实 Step 5 输出分类结果：

```text
text 5x5:
  num_input_score_records=25
  num_classified_paths=5
  P_forget=1
  P_shared=0
  P_retain=0
  P_irrelevant=4

vision_text 5x5:
  num_input_score_records=25
  num_classified_paths=5
  P_forget=1
  P_shared=1
  P_retain=2
  P_irrelevant=1

combined 5x5:
  num_input_score_records=50
  num_classified_paths=10
  P_forget=2
  P_shared=1
  P_retain=2
  P_irrelevant=5
```

扩大规模后的 `step6_v1`：

```text
Step 5 inputs:
  text 20 pairs x 50 paths = 1000 records, status={'ok': 1000}
  vision_text 20 pairs x 8 paths = 160 records, status={'ok': 160}

Step 6 output:
  num_input_score_records=1160
  num_classified_paths=58
  P_forget=27
  P_shared=12
  P_retain=6
  P_irrelevant=13
```

保守对照 `step6_v1_quantile75`：

```text
forget_threshold=0.0021484375
retain_threshold=0.0003824869791666667
P_forget=8
P_shared=3
P_retain=12
P_irrelevant=35
```

当前真实输出目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_text_5x5/
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_vision_text_5x5/
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_combined_5x5/
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1_quantile75/
```

### 借鉴思想说明

Step 6 没有直接照搬外部仓库代码。实现思想与以下方向一致：

- ROME / causal tracing / activation patching：用 clean、corrupt、restore/ablate 的局部干预结果判断组件是否有因果作用。
- 机制化 unlearning / circuit attribution：把 forget-set 影响和 retain-set 影响分开度量，再优先编辑 forget 影响高、retain 影响低的机制。

当前四分类是本项目结合 `Nec/Suf/Ret` 的本地实现，用于连接 Step 5 的因果分数和 Step 7 的 masked RMisU。

### 输出文件

```text
outputs/paths/P_forget.jsonl
outputs/paths/P_shared.jsonl
outputs/paths/P_retain.jsonl
outputs/paths/P_irrelevant.jsonl
```

### 这一阶段的逻辑意义

这一步才能真正回答：

> 如何区分“有害意图路径”和“良性主题路径”？

### 对应科学问题

这一步是问题 1 的核心落点。

---

## Step 7：实现 masked RMisU

### 目标

只编辑 `P_forget`，保护 `P_shared`。

### 当前可复用的代码

- `ours.py`
- `train_eval.py`
- `adaptive_rmu_finetune(...)`

### 新增模块建议

```text
causal_mip/editing/masked_rmisu.py
causal_mip/editing/shared_path_preserve.py
causal_mip/editing/train.py
```

### 当前实现状态

已完成 Step 7 MVP：

```text
causal_mip/editing/__init__.py
causal_mip/editing/masked_rmisu.py
causal_mip/test_step7_masked_rmisu.py
```

当前功能：

- 从 `P_forget.jsonl` / `P_shared.jsonl` 读取 Step 6 路径集合
- 回到 `P_cand.jsonl` 解析语言侧和 Qwen2.5-VL visual block 的 `mlp.down_proj` 节点
- 构建：
  - `forget_neurons`
  - `shared_neurons`
  - `editable_neurons = forget - shared`
  - `preserve_neurons = forget union shared`
- 用 `PartialLinear` 只开放 `up_proj/gate_proj` 中 editable intermediate neurons 的参数更新
- 提供独立的 `masked_rmisu_finetune(...)` MVP 训练函数
- `mm_projector` 和 attention path 仍未纳入 Step 7

真实 `step6_v1` mask 构建检查：

```text
modules=36
forget_neuron_refs=203
shared_neuron_refs=114
editable_neuron_refs=123
```

视觉侧代码能力检查：

```text
vision_fisher_p000000 -> modules=32, module_kind_counts={'vision': 32}, editable_by_kind={'vision': 32}
```

当前生产用 `step6_v1_quantile75` 仍只包含语言侧路径；需要先基于视觉侧 Step5 评分重跑 Step6，生成包含 `vision` / `vision_text` 的路径集合后，再跑真实视觉 masked RMisU。

测试：

```text
Step 7 masked RMisU tests passed.
```

Step 7 已通过显式参数接入 `main.py` / `ours.py` 主训练入口，默认不启用，因此不会影响 baseline 流程。当前已完成真实 2-step smoke test、preserve 2-step smoke test 和 preserve 1 epoch masked RMisU 小实验。

新增参数：

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

真实 smoke test：

```text
Qwen2.5-VL-3B-Instruct
CLEAR forget_ratio=5
step6_v1/P_forget.jsonl
step6_v1/P_shared.jsonl
masked_rmisu_max_steps=2
skip_post_unlearning_eval=True
```

结果：

```text
num_loss_records=2
num_modules=34
num_editable_neurons=123
loss: 74.0 -> 70.0
```

preserve smoke test：

```text
Qwen2.5-VL-3B-Instruct
CLEAR forget_ratio=5
step6_v1/P_forget.jsonl
step6_v1/P_shared.jsonl
masked_rmisu_max_steps=2
rmu_alpha=0.5
rmu_beta=0.5
masked_rmisu_shared_alpha=1.0
skip_post_unlearning_eval=True
```

结果：

```text
output=/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_preserve_smoke_0521.json
num_loss_records=2
num_modules=34
num_editable_neurons=123
loss: 37.0 -> 35.0
unlearn_loss: 74.0 -> 70.0
retain_loss: 0.00384521484375 -> 0.0057373046875
shared_loss: 0.0030059814453125 -> 0.00341796875
```

preserve 1 epoch masked RMisU 小实验：

```text
output=/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/masked_rmisu_step7_full1epoch_0521.json
num_loss_records=1790
num_modules=34
num_editable_neurons=123
loss_mean=31.31417597765363
loss: 37.0 -> 30.5
unlearn_loss: 74.0 -> 59.75
retain_loss: 0.00384521484375 -> 1.28125
shared_loss: 0.0030059814453125 -> 0.0045166015625
```

注意：1 epoch 小实验使用 `--skip_post_unlearning_eval`，当前输出是训练 summary JSON。进入 Step 8 前需要确认 Step7 是否保存了可加载 checkpoint；如果没有，应补充保存逻辑后重跑或在训练后直接接评估。

### MVP 第一版建议

不要第一版就直接做完整 safe-anchor。

先做：

- `P_forget` 上的 random-direction misdirection
- `P_shared` 上的 representation preservation
- retain 样本上的 KL / MSE preserve

### 训练逻辑

```text
只允许 P_forget 对应参数更新
P_shared 对应参数冻结或施加 preserve loss
非 path 参数全部冻结
```

### 这一阶段的逻辑意义

这是从：

```text
causal path identification
```

走向：

```text
causal path selective editing
```

的关键步骤。

### 对应科学问题

这一步直接回答问题 3。

---

## Step 8：做最终评估

### 目标

验证模型是否真正达成：

- 删除目标知识
- 保留 retain knowledge
- 在 hard retain 和 counterfactual retain 上更稳定

### 当前项目可复用的代码

- `train_eval.py`
- `score_by_llm.py`

### 当前实现状态

已完成 Step 8 MVP：

```text
causal_mip/evaluation/__init__.py
causal_mip/evaluation/step8_final_eval.py
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
```

当前评估入口直接消费 Step3 pair 文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl
```

并支持两种模型加载方式：

```text
Step7 full checkpoint:
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521

PEFT baseline:
base model + /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/model_caches/Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth
```

当前输出：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_smoke_val_max1_0522.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_val_full_0522.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_peft_baseline_val_full_0522.json
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/step8_eval_baseline_val_full_0522.json
```

Val 集规模：

```text
causal_pairs_val.jsonl = 18 pairs
total examples = 72
forget_clean = 18
hard_retain = 36
counterfactual_retain = 18
```

关键结果：

```text
PEFT baseline:
forget_clean name_hit_rate=0.6111
hard_retain name_hit_rate=0.6944
counterfactual_retain name_hit_rate=0.7222

Step7 masked RMisU:
forget_clean name_hit_rate=0.7778
hard_retain name_hit_rate=0.7222
counterfactual_retain name_hit_rate=0.7222
```

当前结论：Step7 checkpoint 没有达成预期遗忘，`forget_clean` 目标名字泄露率相对 PEFT baseline 上升。retain 侧没有明显崩坏，但 forget 失败，因此当前 Step7 不能作为成功编辑版本。

### Step6/Step7 优化进展

已继续完成两组优化：

1. 使用 Step6 保守集合 `step6_v1_quantile75`
2. 在 Step7 中加入输出层 forget CE 反向项

新增参数：

```text
--masked_rmisu_forget_ce_alpha
```

优化后对比：

```text
PEFT baseline:
forget_clean name_hit_rate=0.6111

Step7 step6_v1:
forget_clean name_hit_rate=0.7778

Step7 step6_v1_quantile75:
forget_clean name_hit_rate=0.6667

Step7 step6_v1_quantile75 + forget CE alpha=0.1:
forget_clean name_hit_rate=0.6111
```

当前最佳版本：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/checkpoints/masked_rmisu_step7_quantile75_ce01_full1epoch_0522
```

但它只是回到 PEFT baseline，还没有低于 baseline，因此仍不能声明成功遗忘。

### 最低限度评估集合

- forget-set
- retain-set
- hard retain
- counterfactual retain

### 第二阶段再补的评估

- jailbreak
- OCR trigger
- relearning attack

### 对应科学问题

这是问题 3 的最终验证层。

---

## 5. 这条路线如何对应三个科学问题

## 问题 1：哪些路径是真正承载目标知识的因果路径，如何区分“有害意图路径”和“良性主题路径”？

依赖步骤：

```text
Step 2 → Step 3 → Step 4 → Step 5 → Step 6
```

最终回答方式：

- 候选路径来自 `P_cand`
- 因果路径来自 `Nec/Suf`
- harmful-intent vs benign-topic 的区分来自 `Ret` 与 `P_forget/P_shared`

---

## 问题 2：如何定位跨模态因果路径中的有害知识？

依赖步骤：

```text
Step 2 → Step 4 → Step 5
```

最终回答方式：

- 文本路径、视觉路径、cross-modal bridge path 先作为候选
- 再通过 patching 验证其对 harmful target 的必要性和充分性

---

## 问题 3：如何在删除目标知识时避免破坏 retain knowledge？

依赖步骤：

```text
Step 5 → Step 6 → Step 7 → Step 8
```

最终回答方式：

- 通过 `Ret(P)` 找到共享路径
- 通过 `P_shared` 保护避免误伤
- 通过 masked RMisU 实现选择性编辑

---

## 6. 第一版不建议做的内容

为了保证“最短可行”，第一版不要同时做：

- `mm_projector` 精细 patching
- attention head patching
- OCR trigger 和 jailbreak 对抗训练
- safe-anchor 的完整语义空间建模
- `pyvene` 后端封装

第一版要优先保证的是：

- 可跑通
- 可解释
- 可验证

---

## 7. 一句话总结

这条路线的核心不是：

> 直接把 `MIP-Editor` 改成 `CHIP-Editor`

而是：

> 先把当前项目升级成“候选路径 -> 因果验证 -> 路径分类 -> 选择性编辑”的四段式管线，再让这条管线分别回答三个科学问题。

---

## 8. 相关代码仓库清单

### 当前主项目

- `MIP-Editor` 当前仓库：作为 baseline、候选路径、RMisU 基座

### 可借鉴仓库

- `NOTICE`：用于反事实 pair 数据构造思想
- `ROME`：用于 activation patching / causal tracing 模板
- `pyvene`：用于后续 intervention 工程化封装

### 当前项目中最关键的现有代码位置

- `main.py`
- `ours.py`
- `train_eval.py`
- `ig.py`
- `fisher.py`
- `partial_linear.py`
- `causal_mip/path_localization/`
