# Step 4：Activation Cache + Path Patching 流程

## 1. 目标

Step 4 的目标是把 Step 2 导出的 `candidate paths` 和 Step 3 构造出的 `clean/corrupt/retain pairs` 连接起来，使路径从：

```text
candidate influential path
```

升级为：

```text
causal-testable path
```

这一步只解决两件事：

- 如何缓存路径节点上的激活
- 如何对这些节点执行 restore / ablate patching

它还不负责最终的 `Nec/Suf/Ret` 打分，那属于 Step 5。

---

## 2. 借鉴 ROME 的部分

Step 4 借鉴了 `ROME` 的 `causal_trace` / `nethook` 思想：

- 多层 hook tracing
- clean run / corrupt run / restored run
- 使用 forward hook / forward pre-hook 对指定层状态做局部干预

当前没有直接依赖 `rome` 包运行，而是把最小必要 tracing 逻辑本地化到了：

- `causal_mip/interventions/hooks.py`

原因是：

- 避免运行时依赖 `rome` 的目录结构
- 保持 `causal_mip` 内部接口独立

---

## 3. 当前实现模块

- `causal_mip/interventions/hooks.py`
- `causal_mip/interventions/activation_cache.py`
- `causal_mip/interventions/patching.py`
- `causal_mip/interventions/ablation.py`
- `causal_mip/interventions/restoration.py`
- `causal_mip/test_step4_interventions.py`

---

## 4. 当前实现边界

当前 Step 4 已从最初的语言侧 MVP 扩展到 Qwen2.5-VL 视觉 block 的 MLP down_proj 路径 patching。

当前支持：

- `model.language_model.layers.{j}.mlp.down_proj`
- `base_model.model.language_model.layers.{j}.mlp.down_proj`
- `language_model.layers.{j}.mlp.down_proj`
- `model.layers.{j}.mlp.down_proj`
- `model.visual.blocks.{j}.mlp.down_proj`
- `base_model.model.visual.blocks.{j}.mlp.down_proj`
- `visual.blocks.{j}.mlp.down_proj`

当前不支持：

- `mm_projector`
- self-attention patching

因此：

- `text` path 可直接 patch
- `vision` path 可 patch Qwen2.5-VL visual block 的 `mlp.down_proj` 节点
- `vision_text` path 可同时提取视觉 block 和语言侧 patchable 节点
- `mm_projector` 节点仍会被跳过

### 4.1 真实 Qwen2.5-VL 中的干预位置

当前 Step 2 的 `PathNode.neuron` 对应的是 FFN 中间维度 neuron，而不是 transformer hidden size 维度。

因此对语言侧和视觉 block：

```text
model.language_model.layers.{j}.mlp.down_proj
model.visual.blocks.{j}.mlp.down_proj
```

Step 4 实际干预的是 `down_proj` 的**输入侧 activation**：

```text
act_fn(gate_proj(x)) * up_proj(x)
```

而不是 `down_proj` 的输出 hidden states。

原因：

- Qwen2.5-VL-3B 的 hidden size 为 `2048`
- FFN intermediate size 大于 hidden size
- `P_cand.jsonl` 中的 neuron index 来自 FFN intermediate dimension，例如 `5112`
- 如果在 `down_proj` 输出侧 patch，会出现真实模型越界：

```text
IndexError: index 5112 is out of bounds for dimension 2 with size 2048
```

当前已修正为：

- `activation_cache.py` 缓存 `down_proj` 输入侧指定 neuron activation
- `patching.py` 通过 forward pre-hook 对 `down_proj` 输入执行 `zero / restore / noise`
- 对语言 activation 支持 `[batch, seq, hidden]`
- 对视觉 activation 支持 `[seq, hidden]`，视觉模块的 `image_tokens` 会解释为视觉序列内全部 patch token
- `hooks.py` 同时支持 forward hook 与 forward pre-hook
- `test_step4_interventions.py` 的断言改为检查输入侧干预

---

## 5. 数据输入来源

Step 4 的输入来自前两步：

### 来自 Step 2

- `outputs/paths/P_cand.jsonl`

提供：

- `CandidatePath`
- `PathNode.module`
- `PathNode.neuron`
- `PathNode.token_selector`

### 来自 Step 3

- `mip_workspace/outputs/causal_pairs_train.jsonl`
- `mip_workspace/outputs/causal_pairs_val.jsonl`

提供：

- `forget_clean`
- `forget_corrupt`
- `hard_retain`
- `counterfactual_retain`

---

## 6. 核心流程

## 6.1 读取 pair 和 candidate path

入口函数：

- `load_pairs_jsonl(...)`
- `load_candidate_paths_jsonl(...)`

作用：

- 加载 Step 3 的 pair 数据
- 加载 Step 2 的路径候选

---

## 6.2 从 pair 中抽取具体样本

入口函数：

- `extract_pair_sample(...)`

支持抽取：

- `forget_clean`
- `forget_corrupt`
- `counterfactual_retain`
- `hard_retain`

对于 `hard_retain`，还可以进一步按类型选择：

- `same_topic`
- `same_reasoning`

---

## 6.3 根据 `image_ref` 恢复图像

入口类：

- `SampleReferenceResolver`

作用：

- 根据 Step 3 中保存的 `dataset_path + row_idx`
- 回到原始 `CLEAR` / `MLLMU` 数据
- 重新取出真实图像

这是 Step 3 和 Step 4 的关键衔接点。

Step 3 只存图像引用，不复制整张图像；Step 4 再按需解析。

---

## 6.4 把样本转成模型输入

入口函数：

- `prepare_sample_batch(...)`

作用：

- 用当前项目已有的 `processor.apply_chat_template(...)` 组织对话
- 构造：
  - `input_ids`
  - `attention_mask`
  - `pixel_values`
  - `image_grid_thw`
  - `labels`

同时额外计算出三类 token 位置：

- `image_token_positions`
- `answer_token_positions`
- `all_token_positions`

这一步是后续 `token_selector` 生效的基础。

---

## 6.5 把 candidate path 解析成可 patch 的节点

入口函数：

- `resolve_candidate_path_targets(...)`

作用：

- 从 `CandidatePath.nodes` 中筛出当前 MVP 可 patch 的节点
- 把：
  - `module`
  - `neuron`
  - `token_selector`

解析成：

- 具体层模块名
- 具体 neuron index
- 具体 token positions

例如：

```text
token_selector = answer_tokens
```

会被解析成该样本中 answer span 对应的 token 下标列表。

---

## 6.6 缓存 clean activation

入口函数：

- `cache_candidate_path_activations(...)`

流程：

1. 对 candidate path 中涉及的模块注册 trace hooks
2. 用 clean sample 跑一次 forward
3. 只截取：
   - 指定 module
   - 指定 token positions
   - 指定 neuron
4. 保存成 `CachedPathActivations`

缓存结构包括：

- `path_id`
- `nodes`
- `target_answer_logprob`

其中每个 node 会记录：

- `module`
- `neuron`
- `token_positions`
- `values`

---

## 6.7 执行 ablation

入口函数：

- `build_zero_ablation_interventions(...)`
- `ablate_candidate_path(...)`

当前 MVP 先实现最简单的：

- 把目标 neuron 在指定 token positions 上置零

注意：对 `mlp.down_proj` 而言，这里的目标 neuron 指的是 `down_proj` 输入侧 FFN intermediate neuron。

用途：

- 后续用于 Step 5 的 `Nec(P)`

即：

> 删除路径后，目标答案 log-prob 是否下降

---

## 6.8 执行 restoration

入口函数：

- `build_restoration_interventions(...)`
- `restore_path_activations(...)`

流程：

1. 先从 clean run 缓存指定路径节点的激活
2. 在 corrupt sample 的 forward 中，对应模块位置注入 clean activation

用途：

- 后续用于 Step 5 的 `Suf(P)`

即：

> 在 corrupted 输入里恢复这条路径后，目标答案 log-prob 是否回来

---

## 6.9 通用 patching 执行器

底层统一入口：

- `run_patched_forward(...)`

它负责：

- 把多个 `NodeIntervention` 分层组织
- 通过 hook / pre-hook 改写指定层状态
- 返回 patched forward 的 outputs 和 traced activations

目前支持的 intervention mode：

- `zero`
- `restore`
- `noise`

---

## 8. 已验证状态

### 8.1 单元测试

在 `mip-editor` 环境中运行：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
python causal_mip/test_step4_interventions.py
```

结果：

```text
Step 4 intervention tests passed.
```

### 8.2 真实模型验证

使用真实 `Qwen2.5-VL-3B-Instruct`、真实 CLEAR 图像、真实 Step 2/3 产物运行 Step 5 时，Step 4 的 cache / ablation / restoration 已被完整调用并通过。

验证规模：

```text
1 pair x 1 text path
5 pairs x 5 text paths
5 pairs x 5 vision_text paths
1 pair x 1 vision path
1 pair x 1 vision_text path
```

关键结果：

```text
text path:        num_nodes=36, num_patchable_nodes=36, status=ok
vision path:      num_nodes=32, num_patchable_nodes=32, status=ok
vision_text path: num_nodes=4,  num_patchable_nodes=3,  status=ok
```

这验证了：

- text path 的所有语言侧节点都能 patch
- vision path 的 Qwen2.5-VL visual block 节点能 patch
- vision_text path 能同时 patch 视觉 block 与语言侧 `down_proj`，并跳过当前不支持的 `mm_projector`
- `down_proj` 输入干预可以在真实 Qwen2.5-VL-3B 的语言侧和视觉侧运行，不再出现 neuron index 越界

---

## 7. 当前 metric

Step 4 当前提供了一个后续可直接复用的基础 metric：

- `compute_target_answer_logprob(...)`

作用：

- 对指定样本的 answer span 计算目标 token 的平均 log-prob

这正是路线图建议的 MVP metric。

---

## 8. 当前测试

测试文件：

- `causal_mip/test_step4_interventions.py`

当前已验证：

1. mixed path 中可解析视觉 block 与语言侧 patchable 节点
2. clean run 可以缓存指定 path 的 neuron activations
3. ablation 后对应 neuron 输出被成功置零
4. restoration 后对应 neuron 输出被成功恢复成 clean activation
5. pure vision path 可以在 toy model 上完成 cache / ablate / restore

另外还做了真实数据 smoke check：

- 能从 `causal_pairs_train.jsonl` 读取第一条 pair
- 能从 `image_ref` 回溯出原始图像
- 能读取 `P_cand.jsonl` 的 candidate path

---

## 9. 当前限制

### 9.1 只 patch LLM-side MLP

这符合路线图的 MVP 约束，但意味着当前还不能直接完成：

- vision path causal patching
- full cross-modal bridge patching

### 9.2 `P_cand` 和 pair 仍不是样本级一一绑定

Step 4 现在已经能分别读：

- path
- pair

但还没有：

- `path_id -> pair_id` 的直接映射层

所以当前更适合：

- 对指定 path 做 causal patching 原型验证

而不是直接做全量自动打分。

### 9.3 当前只提供 patching 基础设施，不含完整评分流水线

真正的：

- `Nec(P)`
- `Suf(P)`
- `Ret(P)`

打分模块仍属于 Step 5。

---

## 10. 一句话总结

Step 4 当前已经把：

```text
Step 2 的 candidate paths
+ Step 3 的 clean/corrupt pairs
```

接成了一条可运行的：

```text
activation cache → ablation / restoration patching
```

闭环，为下一步计算 `Nec/Suf/Ret` 提供了直接可用的工程基础。
