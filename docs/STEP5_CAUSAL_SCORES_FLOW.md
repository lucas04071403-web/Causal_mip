# Step 5：Necessity / Sufficiency / Retain Impact 计算流程

## 1. 目标

Step 5 的目标是基于 Step 4 已经具备的：

- clean activation cache
- path ablation
- path restoration

把每条 candidate path 变成带有因果分数的可比较对象。

当前统一使用的评分量是：

```text
target answer log-prob
```

因此 Step 5 输出的核心量为：

- `Nec(P)`
- `Suf(P)`
- `Ret(P)`

---

## 2. 与前几步的关系

### 来自 Step 2

Step 5 读取：

- `P_cand.jsonl`

提供 candidate paths。

### 来自 Step 3

Step 5 读取：

- `causal_pairs_train.jsonl`
- `causal_pairs_val.jsonl`

提供：

- `forget_clean`
- `forget_corrupt`
- `same_topic`
- `same_reasoning`
- `counterfactual_retain`

### 来自 Step 4

Step 5 复用：

- `prepare_sample_batch(...)`
- `cache_candidate_path_activations(...)`
- `ablate_candidate_path(...)`
- `restore_path_activations(...)`
- `compute_target_answer_logprob(...)`

因此 Step 5 本质上是：

```text
Step 2 path
+ Step 3 pair
+ Step 4 patching
→ causal score
```

截至 2026-05-21，这条链路已经用真实 `Qwen2.5-VL-3B-Instruct`、真实 CLEAR 图像、真实 `P_cand.jsonl` 和真实 `causal_pairs_train.jsonl` 在 CUDA 上跑通。

---

## 3. 借鉴的思想

Step 5 主要借鉴两类思想：

### 3.1 ROME / causal tracing

借鉴点：

- clean run
- corrupt run
- restore run
- 对指定层状态做局部干预，再看目标答案概率变化

### 3.2 机制解释中的 necessity / sufficiency 评估

借鉴点：

- 必要性：拿掉一个机制，看目标是否掉下去
- 充分性：只恢复一个机制，看目标是否回得来

当前没有照搬外部仓库的打分格式，而是直接结合本项目的：

- candidate path schema
- counterfactual pair schema

做了本地实现。

---

## 4. 当前实现文件

- `causal_mip/causal_scores/metrics.py`
- `causal_mip/causal_scores/necessity.py`
- `causal_mip/causal_scores/sufficiency.py`
- `causal_mip/causal_scores/retain_impact.py`
- `causal_mip/causal_scores/build_scores.py`
- `causal_mip/test_step5_causal_scores.py`

---

## 5. 三个因果量的定义

## 5.1 `Nec(P)`

含义：

> 删除路径 `P` 后，forget 目标答案的 log-prob 是否明显下降？

当前计算方式：

```text
Nec(P) = clean_score - ablated_score
```

其中：

- `clean_score`：`forget_clean` 上原始目标答案 log-prob
- `ablated_score`：对路径节点做 zero ablation 之后的目标答案 log-prob

如果 `Nec(P)` 越大，说明这条路径对目标答案越必要。

---

## 5.2 `Suf(P)`

含义：

> 在 corrupted 输入中恢复路径 `P` 后，forget 目标答案的 log-prob 是否明显回升？

当前计算方式：

```text
Suf(P) = restored_score - corrupt_score
```

其中：

- `corrupt_score`：在 `forget_corrupt` 输入上，对原始 forget 目标答案的 log-prob
- `restored_score`：在 `forget_corrupt` 输入上，把 clean path activation 恢复后的目标答案 log-prob

这里的关键点是：

> `forget_corrupt` 的输入来自 corrupt sample，但计分目标仍然使用 `forget_clean` 的原始答案。

这一步是 Step 5 的关键实现细节，否则 `Suf(P)` 会被算错。

---

## 5.3 `Ret(P)`

含义：

> 删除路径 `P` 后，retain 样本的答案 log-prob 是否明显受损？

当前计算方式：

对多个 retain 变体分别计算：

```text
impact = baseline_score - ablated_score
```

然后取平均：

```text
Ret(P) = mean(impact_same_topic, impact_same_reasoning, impact_counterfactual)
```

当前支持的 retain 变体：

- `same_topic`
- `same_reasoning`
- `counterfactual_retain`

如果 `Ret(P)` 越大，说明删除该路径越容易误伤 retain knowledge。

---

## 6. 核心流程

## 6.1 构造 pair 对应的输入 batch

入口函数：

- `build_pair_prepared_batches(...)`

输出包括：

- `forget_clean`
- `forget_corrupt_target_clean_answer`
- `same_topic`
- `same_reasoning`
- `counterfactual_retain`

其中最关键的是：

- `forget_corrupt_target_clean_answer`

它使用：

- corrupt 样本的图像和问题
- clean 样本的目标答案

这样 `Suf(P)` 才是在问：

> 恢复路径后，原始目标知识会不会回来？

---

## 6.2 计算 `Nec(P)`

入口函数：

- `compute_necessity(...)`

流程：

1. 在 `forget_clean` 上跑原始 forward，得到 `clean_score`
2. 对 path 执行 zero ablation
3. 在 ablated forward 上得到 `ablated_score`
4. 计算：

```text
Nec(P) = clean_score - ablated_score
```

---

## 6.3 计算 `Suf(P)`

入口函数：

- `compute_sufficiency(...)`

流程：

1. 在 `forget_clean` 上缓存 path activation
2. 在 `forget_corrupt_target_clean_answer` 上跑原始 forward，得到 `corrupt_score`
3. 把 clean activation 恢复到 corrupt forward 中
4. 得到 `restored_score`
5. 计算：

```text
Suf(P) = restored_score - corrupt_score
```

当前实现中，restore 时不会硬用 clean 的 token 下标，而是会按 corrupt batch 重新解析 token positions，再注入 clean activation。

这一步是为了兼容 prompt 长度可能变化的情况。

---

## 6.4 计算 `Ret(P)`

入口函数：

- `compute_retain_impact(...)`

流程：

对每个 retain 变体：

1. 跑 baseline forward，得到 `baseline_score`
2. 对 path 执行 zero ablation
3. 跑 ablated forward，得到 `ablated_score`
4. 计算该 retain 变体上的影响：

```text
impact = baseline_score - ablated_score
```

最后求平均，得到：

```text
Ret(P)
```

同时保留每个 retain 变体的明细，便于后面分析：

- `same_topic`
- `same_reasoning`
- `counterfactual_retain`

---

## 6.5 组装成统一 score record

入口函数：

- `compute_path_causal_score_record(...)`

每条记录会至少包含：

- `pair_id`
- `path_id`
- `path_source`
- `path_modality`
- `mip_score`
- `num_nodes`
- `num_patchable_nodes`
- `Nec`
- `Suf`
- `Ret`
- `necessity`
- `sufficiency`
- `retain_details`

这就是后续 Step 6 做路径分类的直接输入。

---

## 7. 运行命令

### 7.1 最小真实 CUDA 冒烟测试

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.build_scores \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/llms/ \
  --output_file_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/ \
  --device cuda \
  --batch_size 2 \
  --epochs 1 \
  --image_resize 224 \
  --pairs_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl \
  --candidate_paths_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_smoke_cuda.jsonl \
  --num_pairs 1 \
  --num_paths 1 \
  --path_modality text
```

### 7.2 text 5x5 测试

```bash
conda run -n mip-editor python -m causal_mip.causal_scores.build_scores \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/llms/ \
  --output_file_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/ \
  --device cuda \
  --batch_size 2 \
  --epochs 1 \
  --image_resize 224 \
  --pairs_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl \
  --candidate_paths_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl \
  --num_pairs 5 \
  --num_paths 5 \
  --path_modality text
```

### 7.3 vision_text 5x5 测试

```bash
conda run -n mip-editor python -m causal_mip.causal_scores.build_scores \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --llm_directory /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/llms/ \
  --output_file_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/ \
  --device cuda \
  --batch_size 2 \
  --epochs 1 \
  --image_resize 224 \
  --pairs_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl \
  --candidate_paths_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl \
  --num_pairs 5 \
  --num_paths 5 \
  --path_modality vision_text
```

---

## 8. 已验证结果

### 8.1 最小真实 CUDA 冒烟测试

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_smoke_cuda.jsonl
```

结果：

```text
num_pairs=1
num_paths=1
num_records=1
status=ok
path_id=text_igi_p000000
num_nodes=36
num_patchable_nodes=36
Nec=-0.0078125
Suf=0.0
Ret=-0.0078125
```

### 8.2 text 5x5

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl
```

结果统计：

```text
records=25
status={'ok': 25}
patchable_nodes=[36]
Nec_range=-0.015625 ~ 0.01171875
Suf_range=0.0 ~ 0.0
Ret_range=-0.013020833333333334 ~ 0.008463541666666666
```

### 8.3 vision_text 5x5

输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl
```

结果统计：

```text
records=25
status={'ok': 25}
num_nodes=[4]
patchable_nodes=[2]
Nec_range=-0.0078125 ~ 0.01171875
Suf_range=0.0 ~ 0.0
Ret_range=-0.013020833333333334 ~ 0.005859375
```

解释：

- `text` path 每条 36 个语言层节点，当前全部可 patch。
- 历史 `vision_text` 5x5 结果生成于视觉侧 patching 扩展前，当时每条 4 节点 cross-modal path 只 patch 2 个语言侧 `down_proj` 节点。
- 当前 `Suf_range` 为 0，说明在这批样本和当前 patch 粒度下，restore 没有提升 corrupt 输入上的目标答案 log-prob；这属于实验结果，不代表链路失败。

### 8.4 vision / vision_text 视觉侧 smoke test

视觉侧 patching 扩展后，使用真实 `Qwen2.5-VL-3B-Instruct + CLEAR + P_cand.jsonl + causal_pairs_train.jsonl` 完成最小 smoke test：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/scores/step5_vision_smoke_0522.jsonl
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/scores/step5_vision_text_smoke_0522.jsonl
```

结果：

```text
vision      1 pair x 1 path = 1 record, status=ok, num_nodes=32, num_patchable_nodes=32
vision_text 1 pair x 1 path = 1 record, status=ok, num_nodes=4,  num_patchable_nodes=3
```

说明：

- pure vision path 不再是 `no_patchable_nodes`
- `vision_text` path 中 Qwen2.5-VL visual block 与 language `down_proj` 均参与评分
- `mm_projector` 仍未纳入 patching，因此 4 节点 cross-modal path 当前为 3 个 patchable nodes

---

## 6.6 批量运行入口

入口脚本：

- `causal_mip/causal_scores/build_scores.py`

支持：

- 读取 pair JSONL
- 读取 candidate path JSONL
- 过滤 modality
- 选择 path / pair 子集
- 写出：

```text
outputs/scores/path_causal_scores.jsonl
```

---

## 7. 当前实现边界

### 7.1 当前仍继承 Step 4 的 patching 范围

因此 Step 5 当前真正能给出有效分数的，主要是：

- `text` path
- `vision` path 中的 Qwen2.5-VL visual block `mlp.down_proj`
- `vision_text` path 中的 visual block + language `mlp.down_proj`

当前仍不支持：

- `mm_projector`
- attention head / attention projection path

### 7.2 `P_cand` 与 pair 仍不是样本级绑定

因此 Step 5 当前的输出更准确地说是：

```text
(pair_id, path_id) -> causal scores
```

而不是：

```text
path_id -> global final causal score
```

如果后续要做全量聚合，需要再定义：

- 对同一路径跨 pair 的聚合规则

### 7.3 当前统一使用 log-prob，而不是生成指标

这是有意为之，符合路线图的 MVP 约束：

- 先保证机制验证闭环
- 再扩展到 Rouge / judge / free-form generation

---

## 8. 当前测试

测试文件：

- `causal_mip/test_step5_causal_scores.py`

当前已验证：

1. `Nec/Suf/Ret` 三个模块都能在 toy model 上运行
2. 统一 score record 与分模块结果一致
3. pure vision path 可以在 toy model 上得到 `status=ok`
4. 真实 Qwen2.5-VL vision / vision_text smoke test 均得到 `status=ok`

另外：

- `build_scores.py --help` 已做过导入级 smoke check

---

## 9. 一句话总结

Step 5 当前已经把：

```text
Step 4 的 activation patching 基础设施
```

升级成：

```text
candidate path × counterfactual pair → Nec / Suf / Ret
```

的统一评分链路，为下一步 `P_forget / P_shared / P_retain / P_irrelevant` 分类提供了直接输入。
