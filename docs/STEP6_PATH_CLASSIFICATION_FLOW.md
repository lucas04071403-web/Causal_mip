# Step 6：路径分类 P_forget / P_shared / P_retain / P_irrelevant 完成记录

**更新时间**: 2026-05-21
**状态**: 已实现并通过真实 Step 5 输出验证

---

## 1. 目标

Step 6 的目标是把 Step 5 生成的路径因果分数：

- `Nec(P)`
- `Suf(P)`
- `Ret(P)`

进一步转成可供 Step 7 masked RMisU 使用的四类路径集合：

- `P_forget`
- `P_shared`
- `P_retain`
- `P_irrelevant`

这一步是从：

```text
causal path scoring
```

走向：

```text
selective path editing
```

的关键接口层。

---

## 2. 与前几步的关系

Step 6 读取 Step 5 的输出：

```text
path_scores*.jsonl
```

其中每条记录的粒度是：

```text
pair_id + path_id
```

但 Step 6 的输出需要是 path-level set，因此当前实现会先按 `path_id` 聚合多个 pair 上的分数，再对聚合后的路径做分类。

整体关系为：

```text
Step 2 P_cand
+ Step 3 causal pairs
+ Step 4 patching
+ Step 5 Nec/Suf/Ret
→ Step 6 P_forget / P_shared / P_retain / P_irrelevant
```

---

## 3. 当前实现文件

新增文件：

```text
causal_mip/causal_scores/classify_paths.py
causal_mip/test_step6_classify_paths.py
```

已更新文档：

```text
PROJECT_STRUCTURE.md
CHIP_EDITOR_TECHNICAL_ROADMAP.md
```

---

## 4. 分类规则

### 4.1 聚合

输入是多个 Step 5 score records：

```json
{
  "pair_id": "...",
  "path_id": "...",
  "Nec": ...,
  "Suf": ...,
  "Ret": ...
}
```

Step 6 默认按 `path_id` 聚合：

```text
Nec = mean(Nec over pairs)
Suf = mean(Suf over pairs)
Ret = mean(Ret over pairs)
```

当前支持：

- `mean`
- `median`

### 4.2 forget effect

路线图中的基础定义是：

```text
forget_effect = Nec + alpha * Suf
```

当前实现默认使用非负裁剪：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

原因是：

- 如果 `Nec < 0`，表示 ablation 后目标答案 log-prob 反而升高
- 如果 `Ret < 0`，表示 ablation 后 retain log-prob 反而升高
- 这些负向值不应被当成“需要遗忘”或“容易误伤 retain”的正证据

如果需要保留 signed effect，可以使用：

```bash
--use_signed_effects
```

### 4.3 四分类

根据 `forget_effect` 和 `retain_impact` 是否超过阈值划分：

| 类别 | forget_effect | retain_impact | 含义 |
| --- | --- | --- | --- |
| `P_forget` | 高 | 低 | 应优先编辑的目标遗忘路径 |
| `P_shared` | 高 | 高 | 既影响 forget 又影响 retain，需要保护或谨慎编辑 |
| `P_retain` | 低 | 高 | 主要影响 retain，应避免编辑 |
| `P_irrelevant` | 低 | 低 | 当前证据下不重要 |

当前支持两种阈值策略：

1. 显式阈值：

```bash
--forget_threshold 0.0
--retain_threshold 0.0
```

2. 分位数阈值：

```bash
--forget_quantile 0.75
--retain_quantile 0.75
```

当前真实 5x5 小样本验证使用的是显式阈值 `0.0`，便于把所有正向 effect 直接分出来。

---

## 5. 输出格式

Step 6 输出目录包含：

```text
P_forget.jsonl
P_shared.jsonl
P_retain.jsonl
P_irrelevant.jsonl
P_classified.jsonl
classification_summary.json
```

每条 path-level record 包含：

```json
{
  "path_id": "...",
  "path_source": "...",
  "path_modality": "...",
  "mip_score": ...,
  "num_nodes": ...,
  "num_patchable_nodes": ...,
  "num_score_records": ...,
  "pair_ids": [...],
  "Nec": ...,
  "Suf": ...,
  "Ret": ...,
  "forget_effect": ...,
  "retain_impact": ...,
  "aggregation": "mean",
  "alpha": 1.0,
  "clip_negative_effects": true,
  "category": "P_forget",
  "thresholds": {...}
}
```

---

## 6. 使用方法

### 6.1 单元测试

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor
conda run -n mip-editor python causal_mip/test_step6_classify_paths.py
```

已验证结果：

```text
Step 6 path-classification tests passed.
```

### 6.2 分类 text 5x5

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_text_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

### 6.3 分类 vision_text 5x5

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_vision_text_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

### 6.4 合并分类 text + vision_text

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_combined_5x5 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

---

## 7. 已验证真实结果

### 7.1 text 5x5

输入：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_5x5_cuda.jsonl
```

输出：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_text_5x5/
```

结果：

```text
num_input_score_records=25
num_skipped_score_records=0
num_classified_paths=5
P_forget=1
P_shared=0
P_retain=0
P_irrelevant=4
```

### 7.2 vision_text 5x5

输入：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_5x5_cuda.jsonl
```

输出：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_vision_text_5x5/
```

结果：

```text
num_input_score_records=25
num_skipped_score_records=0
num_classified_paths=5
P_forget=1
P_shared=1
P_retain=2
P_irrelevant=1
```

### 7.3 combined 5x5

输入：

```text
path_scores_real_text_5x5_cuda.jsonl
path_scores_real_vision_text_5x5_cuda.jsonl
```

输出：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_combined_5x5/
```

结果：

```text
num_input_score_records=50
num_skipped_score_records=0
num_classified_paths=10
P_forget=2
P_shared=1
P_retain=2
P_irrelevant=5
```

输出文件：

```text
P_forget.jsonl
P_shared.jsonl
P_retain.jsonl
P_irrelevant.jsonl
P_classified.jsonl
classification_summary.json
```

---

## 8. 借鉴思想

Step 6 没有直接照搬外部仓库代码。

当前实现借鉴的是以下思想：

### 8.1 ROME / causal tracing

ROME 的 causal tracing 思想强调：

- clean run
- corrupt run
- ablation / restoration
- 通过目标 log-prob 变化判断局部机制是否具有因果作用

本项目 Step 5 已经把这种思想转成 `Nec/Suf/Ret`，Step 6 在此基础上做 path-level 分类。

### 8.2 机制化 unlearning / selective editing

机制化 unlearning 的核心思想是：

- 不只看 forget-set 影响
- 还要看 retain-set 影响
- 优先编辑 forget effect 高、retain impact 低的机制
- 对 forget 和 retain 都重要的共享机制要保护

这正对应：

```text
P_forget  -> Step 7 优先编辑
P_shared  -> Step 7 保护或谨慎编辑
P_retain  -> Step 7 避免编辑
```

---

## 9. 当前限制

### 9.1 当前真实验证样本规模较小

当前真实结果来自：

```text
text 5 paths x 5 pairs
vision_text 5 paths x 5 pairs
```

这足以验证 Step 6 链路和输出格式，但还不足以作为最终路径集合。

后续建议扩大到：

```text
text 50x20
vision_text all paths x 20 pairs
```

或根据 GPU 时间进一步扩大。

### 9.2 当前 Suf 在 5x5 中为 0

当前 Step 5 的真实 5x5 结果中 `Suf_range=0.0~0.0`。

因此 Step 6 当前分类主要由：

```text
Nec
Ret
```

驱动。

这不影响 Step 6 的工程正确性，但说明后续需要继续观察更大样本或更强 restoration 粒度下 `Suf` 是否有区分度。

### 9.3 vision_text 当前只分类语言侧可 patch 节点

当前 Step 4 MVP 只 patch 语言侧 `down_proj`。

因此：

```text
vision_text path: num_nodes=4, num_patchable_nodes=2
```

Step 6 可以分类 `vision_text` path，但其分数当前只反映其中语言侧节点的 causal effect，还不是完整视觉侧 + projector + 语言侧的全路径因果分数。

---

## 10. 下一步

建议后续按以下顺序推进：

1. 扩大 Step 5 分数规模，生成更稳定的 Step 6 输入。
2. 用 Step 6 输出检查 `P_forget / P_shared` 的路径分布和分数分布。
3. 将 `P_forget` 和 `P_shared` 接入 Step 7 masked RMisU。
4. 如果 Step 7 结果显示跨模态路径解释不足，再回头扩展 Step 4 的 vision / `mm_projector` patching。

---

## 11. step6_v1 稳定路径集合

在 5x5 冒烟验证之后，已进一步扩大 Step 5/6 规模，生成当前建议用于 Step 7 的 `step6_v1` 路径集合。

### 11.1 Step 5 输入规模

```text
text:
  20 pairs x 50 paths = 1000 score records
  status={'ok': 1000}
  patchable_nodes=[36]
  Nec_range=-0.0234375 ~ 0.0390625
  Suf_range=0.0 ~ 0.0
  Ret_range=-0.020833333333333332 ~ 0.0234375

vision_text:
  20 pairs x 8 paths = 160 score records
  status={'ok': 160}
  num_nodes=[4]
  patchable_nodes=[2]
  Nec_range=-0.01171875 ~ 0.015625
  Suf_range=0.0 ~ 0.0
  Ret_range=-0.014322916666666666 ~ 0.018229166666666668
```

对应 Step 5 输出文件：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_20x50_cuda.jsonl
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_20x8_cuda.jsonl
```

### 11.2 step6_v1 主输出

命令：

```bash
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7/MIP-Editor

conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_20x50_cuda.jsonl \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_20x8_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1 \
  --alpha 1.0 \
  --forget_threshold 0.0 \
  --retain_threshold 0.0
```

结果：

```text
num_input_score_records=1160
num_skipped_score_records=0
num_classified_paths=58
modalities:
  text=50
  vision_text=8

P_forget=27
P_shared=12
P_retain=6
P_irrelevant=13
```

输出目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/
```

包含：

```text
P_forget.jsonl
P_shared.jsonl
P_retain.jsonl
P_irrelevant.jsonl
P_classified.jsonl
classification_summary.json
```

### 11.3 quantile75 对照输出

同时生成了更保守的 75 分位阈值版本：

```bash
conda run -n mip-editor python -m causal_mip.causal_scores.classify_paths \
  --scores_path \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_text_20x50_cuda.jsonl \
    /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/path_scores_real_vision_text_20x8_cuda.jsonl \
  --output_dir /home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1_quantile75 \
  --alpha 1.0 \
  --forget_quantile 0.75 \
  --retain_quantile 0.75
```

结果：

```text
num_input_score_records=1160
num_classified_paths=58
forget_threshold=0.0021484375
retain_threshold=0.0003824869791666667

P_forget=8
P_shared=3
P_retain=12
P_irrelevant=35
```

输出目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1_quantile75/
```

### 11.4 对 Step 7 的建议

建议 Step 7 第一版使用：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_v1/
```

作为主输入。

其中：

- `P_forget.jsonl` 用作优先 misdirection 的路径集合
- `P_shared.jsonl` 用作 preserve / freeze 的保护集合
- `P_retain.jsonl` 和 `P_irrelevant.jsonl` 第一版可先不编辑

`step6_v1_quantile75` 可作为保守对照。如果 Step 7 主版本过度损伤 retain，可以改用 quantile75 的 `P_forget/P_shared`。
