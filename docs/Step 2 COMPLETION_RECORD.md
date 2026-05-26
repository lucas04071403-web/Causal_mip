# Step 2 完成记录：Single Greedy Path → Top-k Candidate Paths

**更新时间**: 2026-05-19
**最近验证**: 2026-05-21
**状态**: ✅ 已修正并通过真实 CUDA 链路验证

---

## 1. 目标概述

将 MIP-Editor 的单一 greedy path 扩展为 top-k candidate paths，为后续的 causal validation 提供稳定的候选路径集合 `P_cand`。

Step 2 的目标不是立刻执行编辑，而是先把：

```text
IGI / IFI score tensor
→ sample-level top-k candidate paths
→ text / vision / cross-modal path export
→ outputs/paths/P_cand.jsonl
```

这条链路做成真实可运行、可被 Step 3/4 直接消费的输入模块。

---

## 2. 最终实现状态

这一步目前已经包含两部分：

1. `causal_mip/path_localization/` 下的 candidate path 抽取与合并逻辑。
2. `main.py` 中基于现有 IG/Fisher cache 直接导出 `P_cand.jsonl` 的主流程入口。

与之前版本相比，已经修正了以下关键问题：

- 候选 path 现在是**每层一个 neuron** 的真实链式 path，不再把每层 top-k neuron 全部塞进一条 path。
- text / vision 的 beam 和随机候选现在会生成**不同组合**，不再重复 greedy path。
- 候选路径按 **sample-level** 从 score tensor 中构造，支持真实 batch 缓存导出。
- `module` 字段改成更接近后续 hook / patching 的真实模块名。
- 新增 `main.py --export_candidate_paths_only`，可不加载整套训练/评估主流程，直接从 cache 导出 Step 2 产物。

---

## 3. 目录结构

```text
MIP-Editor/
├── main.py
├── causal_mip/
│   ├── __init__.py
│   ├── test_step2.py
│   └── path_localization/
│       ├── __init__.py
│       ├── path_schema.py
│       ├── mip_topk_wrapper.py
│       ├── beam_search_paths.py
│       ├── cross_modal_path_builder.py
│       └── cached_path_export.py
```

其中新增的关键文件：

- `cached_path_export.py`
  作用：直接从 `text_ja_*.pt` / `multi_fisher_*.pt` 导出 Step 2 candidate paths。

---

## 4. 核心模块说明

### 4.1 `path_schema.py`

定义 `PathNode` 和 `CandidatePath`。

当前 schema 重点字段：

```python
@dataclass
class PathNode:
    module: str
    layer: Optional[int]
    neuron: int
    token_selector: str

@dataclass
class CandidatePath:
    path_id: str
    source: str
    modality: str
    mip_score: float
    nodes: list[PathNode]
```

当前 `module` 会写成可用于后续干预定位的具体模块名，例如：

- `model.language_model.layers.8.mlp.down_proj`
- `model.visual.blocks.12.mlp.down_proj`
- `mm_projector`

### 4.2 `mip_topk_wrapper.py`

文本候选路径抽取模块。

提供：

- `compute_ig_topk_paths()`
- `calculate_ig_topk_batch()`
- `extract_text_paths_from_mip_scores()`

当前实现方式：

- 输入 score tensor 形状为 `(samples, layers, neurons)`。
- 每个 sample 单独构造 candidate paths。
- 每条 path 在每一层只保留一个 neuron。
- 先生成 greedy path，再生成 beam-like 变体，再补随机组合。

### 4.3 `beam_search_paths.py`

视觉候选路径抽取模块。

提供：

- `compute_fisher_topk_paths()`
- `calculate_fisher_topk_batch()`
- `extract_vision_paths_from_mip_scores()`

与文本分支一致：

- 按 sample 构造。
- 每层一个 neuron。
- 输出 `vision_fisher_p*` 路径。

### 4.4 `cross_modal_path_builder.py`

跨模态路径构造模块。

当前有两类接口：

- `build_simple_cross_modal_paths()`
  作用：轻量 mock / demo。
- `build_cross_modal_paths_from_unimodal_paths()`
  作用：从真实 text / vision candidate paths 组合出 bridge path。

当前 cross-modal path 结构为：

```text
vision path first node
→ mm_projector
→ text path first node (image-side)
→ text path last node (answer-side)
```

说明：

- `mm_projector` neuron 目前仍是占位设计。
- 这已经足够支撑 Step 3/4 的 schema 对接和路径级别分析。

### 4.5 `cached_path_export.py`

这是当前 Step 2 最关键的主流程承接模块。

它会：

1. 根据 `args.dataset / args.model / args.forget_ratio` 定位 score cache。
2. 读取：
   - `text_ja_*.pt`
   - `multi_fisher_*.pt`
3. 分别导出：
   - text candidate paths
   - vision candidate paths
   - cross-modal candidate paths
4. 合并保存到：

```text
outputs/paths/P_cand.jsonl
```

---

## 5. 路径生成策略

### 5.1 文本路径（IGI）

每个 sample 的文本 score tensor：

```text
(num_layers, num_neurons)
```

生成策略：

| 策略 | 说明 |
| --- | --- |
| Greedy | 每层取 top-1 neuron |
| Beam-like variants | 用每层 top-2 / top-3 替换 greedy 位置，形成不同组合 |
| Random combinations | 从每层 top-k 中随机抽样，补充多样性 |

### 5.2 视觉路径（Fisher）

与文本路径相同：

| 策略 | 说明 |
| --- | --- |
| Greedy | 每层取 top-1 neuron |
| Beam-like variants | 每层 top-2 / top-3 组合 |
| Random combinations | 从每层 top-k 中随机抽样 |

### 5.3 跨模态路径

当前实现不做复杂全图搜索，而是先复用 Step 2 的 unimodal candidate paths，再构造 bridge path：

```text
vision candidate path
+ text candidate path
→ cross-modal bridge path
```

这符合 MVP 的简化路线，足够给后续 causal validation 提供候选输入。

---

## 6. 输出格式

### 6.1 输出文件

默认输出：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl
```

### 6.2 JSONL schema

每行一个 `CandidatePath`：

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
    },
    {
      "module": "model.language_model.layers.1.mlp.down_proj",
      "layer": 1,
      "neuron": 6036,
      "token_selector": "image_tokens"
    }
  ]
}
```

### 6.3 路径类型

| 路径类型 | path_id 前缀 | modality | source |
| --- | --- | --- | --- |
| 文本路径 | `text_igi_p` | `text` | `mip_editor_igi` |
| 视觉路径 | `vision_fisher_p` | `vision` | `mip_editor_ifi` |
| 跨模态路径 | `cross_modal_p` | `vision_text` | `cross_modal` |

---

## 7. 使用方法

### 7.1 运行 Step 2 测试

环境：

```bash
source /home/lucas/miniconda3/etc/profile.d/conda.sh
conda activate mip-editor
```

测试命令：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
python -m causal_mip.test_step2
```

### 7.2 直接从 cache 导出 `P_cand.jsonl`

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
python main.py \
  --export_candidate_paths_only \
  --dataset clear \
  --model Qwen2.5-VL-3B-Instruct \
  --forget_ratio 5 \
  --path_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/influential_paths/ \
  --candidate_paths_output /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl \
  --candidate_num_paths 4 \
  --candidate_per_layer_topk 3 \
  --candidate_cross_modal_paths 8
```

### 7.3 `main.py` 新增参数

```python
--export_candidate_paths
--export_candidate_paths_only
--candidate_num_paths
--candidate_per_layer_topk
--candidate_cross_modal_paths
--candidate_paths_output
```

说明：

- `--export_candidate_paths_only` 只导出 candidate paths，不进入训练/评估主流程。
- `--export_candidate_paths` 可以在主流程前先导出候选路径。

---

## 8. 已验证结果

### 8.1 环境修复

`mip-editor` 环境原先缺少：

```text
typing_extensions
```

已补齐最小依赖，确保 `torch` 可以正常导入并运行 Step 2 测试。

### 8.2 Step 2 测试结果

在 `mip-editor` 环境中实跑结果：

```text
============================================================
Causal-MIP-Editor MVP: Step 2 Test
============================================================
  ✓ PASS: Path Schema
  ✓ PASS: IGI Top-k Paths
  ✓ PASS: Fisher Top-k Paths
  ✓ PASS: Cross-Modal Builder
  ✓ PASS: Cross-Modal From Unimodal
  ✓ PASS: Merge and Save
  ✓ PASS: Full Pipeline

Total: 7/7 tests passed
```

### 8.3 `P_cand.jsonl` 实际导出结果

基于当前 cache 真实导出：

```text
Saved 2312 candidate paths to .../mip_workspace/outputs/paths/P_cand.jsonl
text=1552, vision=752, vision_text=8, total=2312
```

输出文件已存在：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/P_cand.jsonl
```

---

## 9. 对后续 Step 3 / Step 4 的意义

当前 Step 2 已经能为后续提供稳定输入：

- Step 3 可以围绕 `P_cand.jsonl` 中的 path_id 构造 clean/corrupt/hard retain 对齐样本。
- Step 4 可以基于 `nodes[].module / layer / neuron / token_selector` 做 activation cache 和 path patching。

当前最重要的承诺不是“已经做了因果验证”，而是：

```text
Step 2 的候选路径输入层已经真实可运行、可导出、可被后续步骤消费。
```

截至 2026-05-21，这一承诺已经被真实 Step 3/4/5 链路验证：

```text
P_cand.jsonl
+ causal_pairs_train.jsonl
+ Qwen2.5-VL-3B-Instruct
+ CUDA
→ path_scores_real_*.jsonl
```

已跑通：

```text
1 pair x 1 text path
5 pairs x 5 text paths
5 pairs x 5 vision_text paths
```

验证结果：

```text
text        25 records, status={'ok': 25}, num_patchable_nodes=[36]
vision_text 25 records, status={'ok': 25}, num_nodes=[4], num_patchable_nodes=[2]
```

其中 `vision_text` 每条 path 的视觉节点和 `mm_projector` 节点当前由 Step 4 MVP 跳过，只验证语言侧 `down_proj` 节点。

---

## 10. 仍然保留的简化与后续补充

- `mm_projector` neuron 仍是占位符，后续可基于真实 projector 激活再细化。
- 当前 cross-modal path 仍属于 MVP 简化桥接方案，不是完整全图搜索。
- 候选路径数量目前由：
  - `candidate_num_paths`
  - `candidate_per_layer_topk`
  - `candidate_cross_modal_paths`
  控制，后续可根据 causal validation 成本再调节。
- 如果后续需要完整训练/评估主流程，`mip-editor` 环境里可能还需要继续补齐 `transformers/requests` 相关依赖；这不影响 Step 2 的 candidate path 导出。

---

## 11. 真实 CUDA 验证中的关键修复

真实 `Qwen2.5-VL-3B-Instruct` 测试暴露了一个 toy test 没覆盖的问题：

```text
IndexError: index 5112 is out of bounds for dimension 2 with size 2048
```

原因是：

- Step 2 导出的 `PathNode.neuron` 来自 FFN intermediate dimension。
- `mlp.down_proj` 的输出维度是 hidden size，例如 Qwen2.5-VL-3B 为 `2048`。
- 因此如果 Step 4 在 `down_proj` 输出侧 patch，会出现 neuron index 越界。

已修复为：

- Step 4 在 `down_proj` 输入侧 cache / ablate / restore。
- `hooks.py` 增加 forward pre-hook 支持。
- `patching.py` 对 module input 执行 `zero / restore / noise`。
- `activation_cache.py` 缓存 module input 中的目标 neuron activation。
- `test_step4_interventions.py` 更新为检查 input-side patching。

修复后验证：

```text
Step 4 intervention tests passed.
Step 5 causal-score tests passed.
真实 CUDA Step5 text 5x5 passed.
真实 CUDA Step5 vision_text 5x5 passed.
```
