# Step 3：反事实 Pair 数据构造流程

## 1. 目标

Step 3 的目标是为后续的 causal validation 提供统一输入，使后续能够基于：

- `forget_clean`
- `forget_corrupt`
- `hard_retain`
- `counterfactual_retain`

来计算路径的必要性、充分性和 retain impact。

这一步的核心作用不是直接找因果路径，而是把 Step 2 的候选路径验证问题，转换成一个可操作的反事实数据问题。

---

## 2. 与 Step 1 / Step 2 的关系

### Step 1：Baseline

Step 3 复用 Step 1 的数据设定和任务配置。

当前 `CLEAR + forget_ratio=5` 的构造来源是：

- `CLEAR/forget5+tofu`
- `CLEAR/forget5_perturbed`
- `CLEAR/retain95+tofu`

因此，Step 3 和当前 baseline 是一致的，不是另一套独立数据流程。

### Step 2：Candidate Paths

Step 2 导出的 `P_cand.jsonl` 提供的是：

- 候选文本路径
- 候选视觉路径
- 候选跨模态路径

Step 3 不直接修改这些路径，而是提供后续验证这些路径所需的 `clean/corrupt/retain` 输入。

因此两者关系是：

```text
Step 2 给出 candidate paths
Step 3 给出 causal validation 的输入样本
Step 4 再把二者结合起来做 patching / tracing
```

### 2.1 当前真实产物

截至 2026-05-21，当前 workspace 已经生成真实 CLEAR pair 文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl
```

规模为：

```text
causal_pairs_train.jsonl = 170 pairs
causal_pairs_val.jsonl   = 18 pairs
```

这些 pair 已被 Step 5 真实 CUDA 评分链路消费，验证了 `image_ref.dataset_path + row_idx` 可以回到原始 CLEAR arrow 数据并恢复图像。

---

## 3. 借鉴 NOTICE 的思想

Step 3 参考了 `NOTICE` 的两个核心思想：

### 3.1 Semantic Minimal Pairs (SMP)

用于图像腐蚀。

对应实现：

- `causal_mip/data_pairs/image_corruption.py`

当前优先级：

1. 优先使用 `CLEAR` 里显式存在的 `forget*_perturbed` 配对样本
2. 如果没有显式配对，再从 retain pool 里选择同模板、长度接近的图像替代样本

### 3.2 Symmetric Token Replacement (STR)

用于文本腐蚀。

对应实现：

- `causal_mip/data_pairs/text_corruption.py`

当前做法：

- 优先掩盖问题中的名字
- 如果原问题没有显式名字，则退化成模板化问题替换

---

## 4. 当前实现文件

Step 3 当前由以下模块组成：

- `causal_mip/data_pairs/build_pairs.py`
- `causal_mip/data_pairs/text_corruption.py`
- `causal_mip/data_pairs/image_corruption.py`
- `causal_mip/data_pairs/hard_retain_builder.py`
- `causal_mip/test_step3_data_pairs.py`

---

## 5. 数据流程

## 5.1 读取并标准化样本

入口在：

- `causal_mip/data_pairs/build_pairs.py`

首先加载原始数据：

- `forget_clean`
- `forget_corrupt`
- `retain_clean`

然后统一整理出标准字段：

- `id`
- `name`
- `question`
- `answer`
- `caption`
- `image_ref`
- `question_template`

其中：

- `question_template` 用于后续 same-reasoning 匹配
- `image_ref` 保留对原图像样本的引用信息，供后续 patching 阶段继续使用

---

## 5.2 构造 `forget_clean`

`forget_clean` 直接来自 forget 集原始样本。

它表示：

> 模型原本应该记住、现在希望遗忘的目标知识样本

---

## 5.3 构造 `forget_corrupt`

这一部分是 Step 3 的关键。

构造顺序如下：

### 路径 A：显式配对腐蚀样本

如果 `forget_corrupt` 数据中存在同 `id` 的样本，则直接使用。

当前 `CLEAR` 就主要走这一路径。

对应逻辑：

- `select_explicit_corruption(...)`

### 路径 B：语义最小图像对

如果没有显式腐蚀样本，则从 retain pool 中找：

- 问题模板尽量一致
- 文本长度尽量接近
- 但样本身份不同

对应逻辑：

- `select_semantic_minimal_image_corruption(...)`

### 路径 C：文本替换退化方案

如果图像腐蚀也无法构造，则退化为文本腐蚀：

- 掩盖名字
- 保留问题结构

对应逻辑：

- `build_text_corrupt_sample(...)`

因此当前 `forget_corrupt` 的优先顺序是：

```text
显式 perturbed pair
→ semantic minimal image pair
→ symmetric token replacement
```

---

## 5.4 构造 `hard_retain`

`hard_retain` 不是普通 retain，而是更接近“容易误伤”的 retain 样本。

当前由两部分组成：

### same_topic

基于同一张 forget 图像，但改成安全问题：

- 不问身份
- 只问场景、物体、视觉内容

作用：

> 检查删除身份知识时，是否还保留对同一图像的良性理解能力

### same_reasoning

从 retain pool 中找与 clean 样本问题模板相同的样本。

例如：

- 都是 “Who is the person in this image?”
- 或都是 “Can you identify this person?”

作用：

> 检查模型是否保留同类推理模板上的一般能力

对应实现：

- `build_same_topic_retain(...)`
- `build_same_reasoning_retain(...)`

---

## 5.5 构造 `counterfactual_retain`

`counterfactual_retain` 从 retain pool 中再选一个不同样本，要求尽量保持模板相近，但避免和 `same_reasoning` 重复。

作用：

> 提供额外的 retain 对照，避免模型只是在某一个模板上偶然保留

对应实现：

- `build_counterfactual_retain(...)`

---

## 5.6 组装成 Pair

每个 pair 最终被组织成：

```json
{
  "pair_id": "...",
  "dataset": "...",
  "forget_clean": {...},
  "forget_corrupt": {...},
  "hard_retain": [...],
  "counterfactual_retain": {...},
  "metadata": {
    "question_template": "...",
    "corruption_type": "...",
    "builder": "causal_mip_step3_notice_inspired"
  }
}
```

其中：

- `pair_id` 是后续 causal validation 的最小单元
- `metadata.corruption_type` 记录当前样本是走了哪类腐蚀路径

---

## 6. 已验证状态

### 6.1 单元测试

在 `mip-editor` 环境中运行：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
python causal_mip/test_step3_data_pairs.py
```

结果：

```text
Step 3 data-pair tests passed.
```

### 6.2 真实链路验证

Step 3 的真实输出已参与以下 Step 5 真实 CUDA 评分：

```text
Qwen2.5-VL-3B-Instruct
CLEAR forget_ratio=5
causal_pairs_train.jsonl
P_cand.jsonl
```

已跑通：

```text
1 pair x 1 text path
5 pairs x 5 text paths
5 pairs x 5 vision_text paths
```

这说明 Step 3 输出的 `forget_clean / forget_corrupt / hard_retain / counterfactual_retain` 字段，以及其中的 `image_ref`，能够被 Step 4/5 正常解析和使用。

---

## 5.7 划分 train / val

构造完成后，当前实现会再做一次 train/val 切分。

对应逻辑：

- `split_pairs_train_val(...)`

目前支持参数：

- `--val_ratio`
- `--val_output`

例如：

- `val_ratio=0.1`

时，会把构造后的 pair 随机打乱后切分成训练集和验证集。

---

## 6. 当前正式产物

当前已生成的正式产物为：

- `mip_workspace/outputs/causal_pairs_train.jsonl`
- `mip_workspace/outputs/causal_pairs_val.jsonl`

当前 `CLEAR + forget_ratio=5` 的结果是：

- 总 pair 数：`188`
- train：`170`
- val：`18`

说明：

- 路线图原本建议输出到 `outputs/data_pairs/`
- 但当前 `mip_workspace/outputs/data_pairs` 已被旧文件占用，不是目录
- 因此正式产物暂时直接输出到 `mip_workspace/outputs/` 下

---

## 7. 运行命令

当前正式构造命令为：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m causal_mip.data_pairs.build_pairs \
  --dataset clear \
  --base_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/datasets \
  --forget_ratio 5 \
  --output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_train.jsonl \
  --val_output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/causal_pairs_val.jsonl \
  --val_ratio 0.1
```

---

## 8. 当前能做到什么

Step 3 当前已经可以：

- 基于 `CLEAR` 正式生成反事实 pair 数据
- 统一组织 `forget_clean / forget_corrupt / hard_retain / counterfactual_retain`
- 和 Step 1 的 baseline 配置保持一致
- 为 Step 4 的 activation patching 提供标准输入

---

## 9. 当前限制

Step 3 当前仍有两个明确限制：

### 9.1 与 Step 2 还不是样本级一一绑定

当前 `P_cand.jsonl` 中只有：

- `path_id`
- `source`
- `modality`
- `mip_score`
- `nodes`

还没有：

- `pair_id`
- `sample_id`
- `forget_id`

因此目前 Step 2 和 Step 3 只能做到：

- 配置级对齐
- 数据集级对齐

还不能做到：

- 路径和 pair 的直接一一映射

### 9.2 `outputs/data_pairs/` 路径被旧文件占用

这不是 Step 3 逻辑错误，而是已有输出目录中存在命名冲突。

如果后续要完全符合路线图的输出路径规范，需要先处理这个旧文件。

---

## 10. 一句话总结

Step 3 当前已经从 `NOTICE` 借鉴了图像腐蚀和文本腐蚀的核心思想，并在当前 `MIP-Editor` 项目中落地成了一条可运行的反事实 pair 构造流程，其作用是：

> 为后续基于 candidate paths 的 causal validation 提供标准化的 clean/corrupt/retain 输入。
