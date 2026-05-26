# CHIP-Editor Step2-Step8 重点实现汇总

本文档整理 Step2 到 Step8 的核心实现内容，去掉原始记录中的重复日志、长命令和阶段性验证细节，保留后续维护代码和复现实验时真正需要的工程信息。

整体链路如下：

```text
Step2 candidate paths
-> Step3 causal pairs
-> Step4 activation cache / patching
-> Step5 Nec / Suf / Ret causal scores
-> Step6 path classification
-> Step7 masked RMisU editing
-> Step8 final evaluation
```

## Step2: Candidate Path Localization

Step2 的目标是把原始 MIP-Editor 中的 single greedy path 扩展为 top-k candidate paths，为后续 causal validation 提供候选路径集合 `P_cand`。

### 核心输入

- text IG / IGI score cache
- vision Fisher / IFI score cache
- 当前模型与数据配置，例如 `Qwen2.5-VL-3B-Instruct + CLEAR + forget_ratio=5`

### 核心输出

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl
```

每行是一个 `CandidatePath`：

```json
{
  "path_id": "text_igi_p000000",
  "source": "mip_editor_igi",
  "modality": "text",
  "mip_score": 0.00179,
  "nodes": [
    {
      "module": "model.language_model.layers.0.mlp.down_proj",
      "layer": 0,
      "neuron": 5112,
      "token_selector": "image_tokens"
    }
  ]
}
```

### 关键模块

```text
causal_mip/path_localization/path_schema.py
causal_mip/path_localization/mip_topk_wrapper.py
causal_mip/path_localization/beam_search_paths.py
causal_mip/path_localization/cross_modal_path_builder.py
causal_mip/path_localization/cached_path_export.py
```

### 实现重点

- `PathNode` 描述单个路径节点：module、layer、neuron、token_selector。
- `CandidatePath` 描述一条完整候选路径：path_id、source、modality、mip_score、nodes。
- text path 来自 IGI score tensor。
- vision path 来自 Fisher / IFI score tensor。
- cross-modal path 由 vision path 和 text path 组合生成，当前通过 `mm_projector` 作为 bridge 节点。
- 每条路径在每层只保留一个 neuron，避免把每层 top-k neuron 错误塞进同一条 path。
- 支持 greedy path、beam-like variants 和 random combinations，提高候选路径多样性。
- 新增 `main.py --export_candidate_paths_only`，可直接从 cache 导出 `P_cand.jsonl`，不进入完整训练流程。

### 当前边界

- `mm_projector` neuron 仍是占位设计。
- cross-modal path 是 MVP bridge 方案，不是完整全图搜索。
- `P_cand.jsonl` 当前还没有和 Step3 的 `pair_id` 做样本级一一绑定。

## Step3: Causal Pair Construction

Step3 的目标是构造 causal validation 需要的 clean / corrupt / retain 输入，把路径验证问题转化为可执行的反事实数据问题。

### 核心输入

- `CLEAR/forget5+tofu`
- `CLEAR/forget5_perturbed`
- `CLEAR/retain95+tofu`

### 核心输出

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl
```

当前 `CLEAR + forget_ratio=5` 产物规模：

```text
total pairs = 188
train = 170
val = 18
```

### 关键模块

```text
causal_mip/data_pairs/build_pairs.py
causal_mip/data_pairs/text_corruption.py
causal_mip/data_pairs/image_corruption.py
causal_mip/data_pairs/hard_retain_builder.py
```

### Pair Schema

每个 pair 的核心结构：

```json
{
  "pair_id": "...",
  "dataset": "clear",
  "forget_clean": {},
  "forget_corrupt": {},
  "hard_retain": [],
  "counterfactual_retain": {},
  "metadata": {
    "question_template": "...",
    "corruption_type": "...",
    "builder": "causal_mip_step3_notice_inspired"
  }
}
```

### 实现重点

- `forget_clean` 来自 forget set 原始样本，表示希望遗忘的目标知识。
- `forget_corrupt` 优先使用 `forget*_perturbed` 中的显式配对样本。
- 如果没有显式配对，则从 retain pool 中构造 semantic minimal image corruption。
- 如果图像腐蚀也不可用，则退化为 text corruption，例如掩盖名字并保留问题结构。
- `hard_retain` 包含两类：
  - `same_topic`: 同一张 forget 图像，但换成安全视觉问题。
  - `same_reasoning`: retain pool 中同问题模板的样本。
- `counterfactual_retain` 额外选择模板相近但身份不同的 retain 样本。
- 每个样本保留 `image_ref.dataset_path + row_idx`，Step4/5 可回到原始 CLEAR arrow 数据恢复图像。

### 当前边界

- Step2 path 与 Step3 pair 当前是配置级对齐，还没有 `path_id -> pair_id` 的直接映射。
- 正式产物暂时放在 `mip_workspace/outputs/` 下，而不是路线图中原定的 `outputs/data_pairs/`。

## Step4: Activation Cache And Path Patching

Step4 的目标是把 Step2 的 candidate paths 和 Step3 的 causal pairs 接起来，使路径可以被 causal testing。

这一步只负责：

- 缓存路径节点上的 activation。
- 对路径节点执行 ablation / restoration / noise patching。

Step5 才负责把 patching 结果转成 Nec / Suf / Ret 分数。

### 核心输入

```text
P_cand.jsonl
causal_pairs_train.jsonl
causal_pairs_val.jsonl
```

### 关键模块

```text
causal_mip/interventions/hooks.py
causal_mip/interventions/activation_cache.py
causal_mip/interventions/patching.py
causal_mip/interventions/ablation.py
causal_mip/interventions/restoration.py
```

### 支持的 patch 位置

当前支持 Qwen2.5-VL 语言侧和视觉侧 MLP：

```text
model.language_model.layers.{j}.mlp.down_proj
base_model.model.language_model.layers.{j}.mlp.down_proj
language_model.layers.{j}.mlp.down_proj
model.layers.{j}.mlp.down_proj
model.visual.blocks.{j}.mlp.down_proj
base_model.model.visual.blocks.{j}.mlp.down_proj
visual.blocks.{j}.mlp.down_proj
```

当前不支持：

```text
mm_projector
self-attention
attention head
```

### 关键实现修正

Step2 中的 `PathNode.neuron` 来自 FFN intermediate dimension，而不是 transformer hidden size。

因此不能在 `down_proj` 输出侧 patch，否则会出现类似错误：

```text
IndexError: index 5112 is out of bounds for dimension 2 with size 2048
```

当前修正为：

- 在 `mlp.down_proj` 输入侧缓存 activation。
- 通过 forward pre-hook 修改 `down_proj` input。
- `zero / restore / noise` 都作用在 FFN intermediate neuron 维度。
- 语言 activation 支持 `[batch, seq, hidden]`。
- 视觉 activation 支持 `[seq, hidden]`。

### 核心流程

1. 读取 pair 和 candidate path。
2. 用 `SampleReferenceResolver` 根据 `image_ref` 恢复图像。
3. 用 `prepare_sample_batch(...)` 构造模型输入。
4. 解析 `token_selector`，得到 image tokens、answer tokens 或 all tokens。
5. 用 `cache_candidate_path_activations(...)` 缓存 clean activation。
6. 用 `ablate_candidate_path(...)` 对路径节点置零。
7. 用 `restore_path_activations(...)` 在 corrupt forward 中恢复 clean activation。
8. 用 `run_patched_forward(...)` 统一执行 patched forward。

### 当前边界

- `mm_projector` 节点会被跳过。
- Step4 提供 patching 基础设施，不做最终路径打分。
- `P_cand` 和 causal pairs 仍没有样本级绑定，因此更适合做 path x pair 的验证。

## Step5: Causal Score Computation

Step5 的目标是基于 Step4 的 patching 能力，为每条 candidate path 计算因果分数。

统一评分量是：

```text
target answer log-prob
```

核心输出为：

```text
Nec(P)
Suf(P)
Ret(P)
```

### 核心输入

```text
P_cand.jsonl
causal_pairs_train.jsonl
causal_pairs_val.jsonl
```

### 核心输出

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/scores/path_causal_scores.jsonl
```

实际 smoke / 小规模结果也保存过：

```text
path_scores_real_smoke_cuda.jsonl
path_scores_real_text_5x5_cuda.jsonl
path_scores_real_vision_text_5x5_cuda.jsonl
scores/step5_vision_smoke_0522.jsonl
scores/step5_vision_text_smoke_0522.jsonl
```

### 关键模块

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/necessity.py
causal_mip/causal_scores/sufficiency.py
causal_mip/causal_scores/retain_impact.py
causal_mip/causal_scores/build_scores.py
```

### 三个因果量

#### Necessity

含义：删除路径 `P` 后，forget 目标答案 log-prob 是否下降。

```text
Nec(P) = clean_score - ablated_score
```

如果 `Nec(P)` 越大，说明路径越必要。

#### Sufficiency

含义：在 corrupted 输入中恢复路径 `P` 后，forget 目标答案 log-prob 是否回升。

```text
Suf(P) = restored_score - corrupt_score
```

关键实现细节：

- corrupt 输入来自 `forget_corrupt`。
- 计分目标仍然是 `forget_clean` 的原始答案。
- 这由 `forget_corrupt_target_clean_answer` batch 明确表达。

#### Retain Impact

含义：删除路径 `P` 后，retain 样本答案 log-prob 是否受损。

```text
impact = baseline_score - ablated_score
Ret(P) = mean(impact_same_topic, impact_same_reasoning, impact_counterfactual)
```

### Score Record

每条 Step5 输出记录至少包含：

```json
{
  "pair_id": "...",
  "path_id": "...",
  "path_source": "...",
  "path_modality": "...",
  "mip_score": 0.0,
  "num_nodes": 36,
  "num_patchable_nodes": 36,
  "Nec": 0.0,
  "Suf": 0.0,
  "Ret": 0.0,
  "necessity": {},
  "sufficiency": {},
  "retain_details": {}
}
```

### 当前边界

- Step5 输出粒度是 `(pair_id, path_id) -> causal scores`。
- 还不是 path-level 全局最终分数。
- 当前使用 log-prob，不使用自由生成指标或 LLM judge。
- 仍继承 Step4 的 patching 范围，不支持 `mm_projector` 和 attention path。

## Step6: Path Classification

Step6 的目标是把 Step5 的 `Nec / Suf / Ret` 分数转成 Step7 可消费的四类路径集合：

```text
P_forget
P_shared
P_retain
P_irrelevant
```

这是从 causal path scoring 走向 selective path editing 的接口层。

### 核心输入

```text
path_scores*.jsonl
```

输入记录粒度是：

```text
pair_id + path_id
```

Step6 先按 `path_id` 聚合多个 pair 上的分数，再做 path-level 分类。

### 核心输出

输出目录包含：

```text
P_forget.jsonl
P_shared.jsonl
P_retain.jsonl
P_irrelevant.jsonl
P_classified.jsonl
classification_summary.json
```

### 关键模块

```text
causal_mip/causal_scores/classify_paths.py
```

### 聚合规则

默认按 `path_id` 取均值：

```text
Nec = mean(Nec over pairs)
Suf = mean(Suf over pairs)
Ret = mean(Ret over pairs)
```

也支持 median 聚合。

### 分类规则

默认使用非负裁剪：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

也可以使用 signed effect：

```text
--use_signed_effects
```

四分类含义：

| 类别 | forget_effect | retain_impact | 作用 |
| --- | ---: | ---: | --- |
| `P_forget` | 高 | 低 | 优先编辑 |
| `P_shared` | 高 | 高 | 谨慎编辑或保护 |
| `P_retain` | 低 | 高 | 避免编辑 |
| `P_irrelevant` | 低 | 低 | 暂不处理 |

阈值策略：

- 显式阈值：`--forget_threshold`, `--retain_threshold`
- 分位数阈值：`--forget_quantile`, `--retain_quantile`

### 当前边界

- 小规模验证已经跑通 text / vision_text 分类。
- 分类质量依赖 Step5 score 的覆盖范围和稳定性。
- 由于 Step5 是 `(pair_id, path_id)` 粒度，Step6 的 path-level 结果本质上依赖聚合策略。

## Step7: Masked RMisU Editing

Step7 的目标是把 Step6 得到的路径集合接入编辑训练。

当前策略：

```text
P_forget: 执行 random-direction misdirection
P_shared: preserve / freeze
P_retain: 第一版不编辑
P_irrelevant: 第一版不编辑
```

### 核心输入

```text
P_cand.jsonl
P_forget.jsonl
P_shared.jsonl
forget dataloader
retain dataloader
```

### 关键模块

```text
causal_mip/editing/masked_rmisu.py
```

核心对象：

```text
PathNeuronMask
MaskedRMisUConfig
build_path_neuron_masks(...)
apply_masked_rmisu_parameter_mask(...)
masked_rmisu_finetune(...)
```

### Mask 构建

流程：

1. 读取 `P_forget` 和 `P_shared` 中的 path_id。
2. 回到 `P_cand.jsonl` 查完整 CandidatePath。
3. 只保留 patchable `mlp.down_proj` 节点。
4. 按 module 聚合 neuron。
5. 构造：

```text
forget_neurons[module]
shared_neurons[module]
editable_neurons = forget_neurons - shared_neurons
preserve_neurons = forget_neurons union shared_neurons
```

这样可以避免编辑 `P_shared` 中标记为共享机制的 neuron。

### 参数级编辑位置

Step4/5 的 neuron 对应 `down_proj` 输入侧 FFN intermediate neuron。

因此 Step7 真正包装训练的是同一 MLP 中：

```text
up_proj output column
gate_proj output column
```

当前通过项目已有 `PartialLinear` 实现参数级 mask：

- 只训练 `editable_neurons` 对应列。
- 其他参数冻结。

### Loss

`masked_rmisu_finetune(...)` 包含：

```text
unlearn_loss:
  forget batch 上，将 editable neuron activation 推向 random direction

retain_loss:
  retain batch 上，让 updated model 的 preserve neuron activation 接近 frozen model

shared_loss:
  retain batch 上，额外保护 shared neuron activation
```

总损失：

```text
loss = beta * unlearn_loss + alpha * retain_loss + shared_alpha * shared_loss
```

后续又加入过输出层 forget CE 约束：

```text
--masked_rmisu_forget_ce_alpha
```

### 主训练入口

`main.py` / `ours.py` 已接入：

```text
--use_masked_rmisu
--masked_rmisu_candidate_paths
--masked_rmisu_p_forget
--masked_rmisu_p_shared
--masked_rmisu_shared_alpha
--masked_rmisu_output
--masked_rmisu_max_steps
--skip_post_unlearning_eval
--masked_rmisu_forget_ce_alpha
```

默认不启用 masked RMisU，不影响原 baseline。

### 当前边界

- 支持语言侧 `model.language_model.layers.{j}.mlp.down_proj`。
- 支持 Qwen2.5-VL visual block `model.visual.blocks.{j}.mlp.down_proj` 的 mask / wrapper。
- 不支持 `mm_projector` 和 attention path。
- preserve loss 仍是 activation-level MVP，可扩展为 retain KL 或 full hidden-state preserve。

## Step8: Final Evaluation

Step8 的目标是评估 Step7 edited model 是否同时满足：

- 删除目标知识。
- 保留 retain knowledge。
- 在 hard retain / counterfactual retain 上稳定。

### 核心输入

```text
Step7 full checkpoint
causal_pairs_val.jsonl
optional PEFT baseline checkpoint
```

### 关键模块

```text
causal_mip/evaluation/step8_final_eval.py
causal_mip/evaluation/full_clear_remote_eval.py
```

### Step8 Pair-Based Evaluation

`step8_final_eval.py` 基于 Step3 的 pair 文件评估：

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

支持参数：

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

实现重点：

- 可直接加载 Step7 保存的 full checkpoint。
- 可选加载 base model + PEFT adapter，并 merge 后作为 baseline。
- 通过 `image_ref.dataset_path + row_idx` 回读原图像。
- 默认不加载额外 LLM scorer，避免 24GB GPU 上 OOM。
- 输出 raw predictions 和 summary metrics。

### Pair-Based Metrics

当前本地轻量指标：

```text
name_hit_rate
BLEU
ROUGE
```

解释：

- 对 `forget_clean`，`name_hit_rate` 越低越好。
- 对 `hard_retain.same_topic`，name hit 不一定越高越好，因为问题不要求识别人物。
- 对 `hard_retain.same_reasoning` 和 `counterfactual_retain`，name hit 越高通常说明 retain 身份知识保留越好。
- BLEU / ROUGE 只作为文本相似度参考，不替代 LLM judge。

### Full CLEAR Remote Evaluation

`full_clear_remote_eval.py` 用于补充完整 CLEAR 主协议评分：

```text
clf_forget
clf_retain
gen_forget
gen_retain
```

实现重点：

- 从 checkpoint 直接生成 full CLEAR predictions。
- 调用远程 `Qwen2.5-VL-7B-Instruct` 做 judge。
- 支持 request timeout，避免单条远程请求无限卡住。
- 支持断点续跑，已评分样本不会重复评分。
- 支持多 worker 远程评分。
- 单独保存预测文件、远程评分文件和 summary。

核心输出：

```text
full_clear_remote_protocol_summary.json
clf_forgetset_multi_preds_remote_scored.json
clf_retainset_multi_preds_remote_scored.json
gen_forgetset_multi_preds_remote_scored.json
gen_retainset_multi_preds_remote_scored.json
```

### 当前实验结论

当前 Step8 和 Full CLEAR 远程评分都没有证明 masked RMisU 成功遗忘。

关键现象：

- Step8 中 `forget_clean name_hit_rate` 没有低于 PEFT baseline。
- Full CLEAR 中 forget classification / generation remote acc 偏高，说明模型仍能答出 forget-set 目标知识。
- retain 没有明显崩坏，但 forget 失败是主要问题。

### 当前边界

- Pair-based Step8 是轻量本地评估，不等价于完整 CLEAR 主协议。
- Full CLEAR 远程评分依赖 judge 模型，结果受 judge 判断标准影响。
- 当前最关键的技术短板仍是视觉/projector/attention 路径没有完整纳入定位、打分和编辑。

## 跨 Step 的关键数据流

| 阶段 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| Step2 | IGI / IFI score cache | `P_cand.jsonl` | 候选路径 |
| Step3 | CLEAR forget / retain 数据 | `causal_pairs_train/val.jsonl` | clean / corrupt / retain pair |
| Step4 | paths + pairs | cached activations / patched forward | patching 基础设施 |
| Step5 | paths + pairs + patching | `path_scores*.jsonl` | Nec / Suf / Ret |
| Step6 | path score records | `P_forget/P_shared/P_retain/P_irrelevant` | 路径分类 |
| Step7 | classified paths + dataloaders | edited checkpoint | masked RMisU 编辑 |
| Step8 | edited checkpoint + eval data | prediction / metric JSON | 最终评估 |

## 当前实现状态总结

已完成：

- Candidate path schema、导出和多模态候选路径构造。
- CLEAR causal pair 构造。
- Qwen2.5-VL 语言侧和视觉 block 的 activation cache / patching。
- Path-level Nec / Suf / Ret causal score 计算。
- 基于因果分数的四类路径集合分类。
- masked RMisU 主训练入口接入。
- Step8 pair-based 评估和 Full CLEAR 远程评分。

主要限制：

- `P_cand` 与 causal pairs 仍不是样本级一一绑定。
- `mm_projector` 和 attention path 仍未纳入 patching / scoring / editing。
- Step5 的 `Suf` 在多次小规模实验中长期为 0，说明当前 restoration 粒度可能不足。
- Step7 的 activation-level random-direction loss 没有稳定压制输出中的目标身份 token。
- 当前实验结果显示工程链路跑通，但遗忘效果尚未超过 baseline。

## 后续优先改进方向

1. 建立 `path_id -> pair_id / sample_id` 的显式绑定，减少路径和样本错配。
2. 将 `mm_projector` 和 attention path 纳入 Step4/5/7。
3. 强化 Step7 输出层 forget 约束，例如目标 name token NLL 反向项或拒答目标。
4. 改进 Step5 sufficiency 计算，检查 corrupt target clean answer 的 token 对齐和 restore token selector。
5. 在 Step8 中保留 pair-based 轻量评估，同时把 Full CLEAR remote judge 作为主协议结果。
