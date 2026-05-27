# MIP-Editor 项目文档

> **项目名称**: MIP-Editor (Multimodal Influential Path Editor)
> **论文**: Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models
> **会议**: AAAI 2026
> **论文链接**: [https://arxiv.org/pdf/2511.06793](https://arxiv.org/pdf/2511.06793)

---

## 1. 项目概述

MIP-Editor 是一个用于**多模态大语言模型 (MLLM) 选择性遗忘**的工具包。它通过识别和编辑跨模态（视觉-语言）影响力神经元路径，实现对模型特定知识的精准移除。

### 核心方法

1. **Fisher Information** (视觉) + **Integrated Gradients** (语言) 识别影响力神经元路径
2. **神经元剪枝** 将影响力路径置零
3. **Partial Linear** 层实现选择性微调
4. **Adaptive RMU** 微调平衡遗忘与保留效果

---

## 2. 目录结构

```
MIP_fusion6/
├── MIP-Editor/                    # 核心源代码目录
│   ├── main.py                    # 主入口：编排整个流水线
│   ├── ours.py                    # MIP-Editor 核心实现
│   ├── unlearning.py              # 基线方法 (GA, KL, NPO)
│   ├── load_model.py              # 模型加载工具
│   ├── load_data.py                # 数据集加载工具
│   ├── train_eval.py               # 训练与评估函数
│   ├── mllmu_bench.py              # MLLMU-Bench 数据集类
│   ├── clear.py                    # CLEAR 数据集类
│   ├── fisher.py                   # 视觉模态 Fisher 信息计算
│   ├── ig.py                       # 文本模态 Integrated Gradients
│   ├── manu.py                     # MANU 剪枝基线
│   ├── utils.py                    # 工具函数
│   ├── partial_linear.py           # PartialLinear 层
│   ├── myqwen2vl.py               # Qwen2-VL 模型封装
│   ├── myqwen25vl.py              # Qwen2.5-VL 模型封装
│   ├── score_by_llm.py            # LLM 评分评估
│   ├── test_data.py                # 测试数据工具
│   ├── metrics/                    # 评估指标
│   └── requirements.txt            # Python 依赖
│
├── mip_workspace/                  # 数据工作目录
│   ├── datasets/                  # 训练/评估数据集
│   │   ├── CLEAR/                 # CLEAR 遗忘基准
│   │   └── MLLMU-Bench/           # MLLMU 遗忘基准
│   ├── llms/                      # 本地模型存储
│   │   ├── Qwen2.5-VL-3B-Instruct/
│   │   └── Qwen2-7B-Instruct/
│   └── influential_paths/          # 预计算神经元重要性
│
├── MIP-Editor论文实验复现流程.md     # 实验复现指南
├── MIP_Editor_Summary.md           # 项目摘要
└── 01_Causal_MIP_Editor_MVP_主线实现路线.md  # 实现路线图
```

---

## 3. 核心文件详解

### 3.1 主流程文件


| 文件              | 作用                  | 关键函数/类                                                                                 |
| --------------- | ------------------- | -------------------------------------------------------------------------------------- |
| `main.py`       | 主入口，编排整个流水线         | 支持 `--unlearning {ga,kl,npo,our,manu}`                                                 |
| `ours.py`       | **MIP-Editor 核心实现** | `ours()`                                                                               |
| `unlearning.py` | 基线遗忘方法              | `ga_difference_training()`, `kl_min()`, `npo()`                                        |
| `train_eval.py` | 训练与评估               | `train()`, `finetune()`, `adaptive_rmu_finetune()`, `evaluate_clf()`, `evaluate_gen()` |


### 3.2 模型与数据加载


| 文件               | 作用         | 关键函数/类                                                                                  |
| ---------------- | ---------- | --------------------------------------------------------------------------------------- |
| `load_model.py`  | 模型加载       | `load_base_model()`, `load_model()`, `load_peft_model()`                                |
| `load_data.py`   | 数据集加载      | `load_data_train()`, `load_data_forget()`, `load_data_retain()`, `load_data_finetune()` |
| `mllmu_bench.py` | MLLMU 数据集类 | `MMLMU_Dataset`, `MMLMU_Clf_Dataset`, `MMLMU_Gen_Dataset`                               |
| `clear.py`       | CLEAR 数据集类 | `CLEAR_Dataset`, `CLEAR_Clf_Dataset`, `CLEAR_Gen_Dataset`                               |
| `test_data.py`   | 测试数据封装     | 批量评估数据封装                                                                                |


### 3.3 神经元分析文件


| 文件          | 作用           | 方法                                                                                               |
| ----------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `fisher.py` | 视觉模态神经元重要性计算 | Integrated Gradients along Fisher paths                                                          |
| `ig.py`     | 文本模态神经元重要性计算 | Integrated Gradients                                                                             |
| `manu.py`   | MANU 基线剪枝器   | Modality-Aware importance scoring                                                                |
| `utils.py`  | 工具函数         | `ffn_at_layer()`, `vb_at_layer()`, `model_layer_prune_to_zero()`, `visual_block_prune_to_zero()` |


### 3.4 模型架构封装


| 文件                  | 模型         | 用途             |
| ------------------- | ---------- | -------------- |
| `myqwen2vl.py`      | Qwen2-VL   | 自定义封装          |
| `myqwen25vl.py`     | Qwen2.5-VL | 自定义封装          |
| `partial_linear.py` | -          | 选择性训练层（冻结部分参数） |


### 3.5 评估工具


| 文件                | 作用            |
| ----------------- | ------------- |
| `score_by_llm.py` | 使用 LLM 评分生成结果 |
| `metrics/`        | 20+ 评估指标实现    |


**metrics 目录包含**:

- 文本生成: `bleu/`, `rouge/`, `google_bleu/`
- 分类: `accuracy/`, `f1/`, `precision/`, `recall/`
- 回归: `mse/`, `mae/`, `mape/`, `mase/`
- 高级: `perplexity/`, `bertscore/`, `bleurt/`
- 问答: `squad/`, `super_glue/`, `glue/`

---

## 4. 流水线流程

### 4.1 主流程 (`main.py`)

```
main.py
    │
    ├── [1] 加载 Processor
    │
    ├── [2] 加载数据集
    │       ├── load_data_forget()     # 遗忘集
    │       ├── load_data_retain()     # 保留集
    │       └── load_data_finetune()   # 微调集
    │
    ├── [3] (可选) 训练基础模型
    │       └── load_model() → train() → 保存 PEFT 权重
    │
    ├── [4] 加载 PEFT 模型
    │       └── load_peft_model()
    │
    ├── [5] (可选) 遗忘前评估
    │       └── evaluate_clf() / evaluate_gen()
    │
    ├── [6] 应用遗忘方法
    │       ├── "ga"    → ga_difference_training()
    │       ├── "kl"   → kl_min()
    │       ├── "npo"   → npo()
    │       ├── "manu"  → MANUPruner
    │       └── "our"   → ours()  ← MIP-EDITOR
    │
    └── [7] 遗忘后评估
            └── evaluate_clf() / evaluate_gen()
```

### 4.2 MIP-Editor 核心流程 (`ours.py`)

```
ours()
    │
    ├── [1] 加载预计算影响力路径
    │       ├── multi_fisher_*.pt  (视觉)
    │       └── text_ja_*.pt      (语言)
    │
    ├── [2] 获取每层 top-k 影响力神经元
    │
    ├── [3] 剪枝神经元至零
    │       ├── visual_block_prune_to_zero()  (视觉)
    │       └── model_layer_prune_to_zero()   (语言)
    │
    ├── [4] 冻结参数 (PartialLinear)
    │       └── freeze_parameters()
    │
    └── [5] Adaptive RMU 微调
            ├── 遗忘方向: MSE loss to random direction
            └── 保留方向: MSE loss to frozen model
```

---

## 5. 支持的模型与数据集

### 5.1 模型


| 模型                     | 路径/名称                         |
| ---------------------- | ----------------------------- |
| Qwen2.5-VL-3B-Instruct | `Qwen2.5-VL-3B-Instruct` (主要) |
| Qwen2-VL-2B-Instruct   | `Qwen2-VL-2B-Instruct`        |
| Qwen2.5-VL-7B-Instruct | `Qwen2.5-VL-7B-Instruct`      |
| LLaVA-1.5-7B           | `llava-1.5-7b-hf`             |


### 5.2 数据集


| 数据集         | 类型                        | 用途      |
| ----------- | ------------------------- | ------- |
| MLLMU-Bench | Multimodal LLM Unlearning | 多模态遗忘基准 |
| CLEAR       | Cross-Modal unlearning    | 跨模态遗忘基准 |


### 5.3 遗忘方法


| 方法     | 描述                                  | 参数        |
| ------ | ----------------------------------- | --------- |
| `ga`   | Gradient Ascent Difference Training | 梯度上升      |
| `kl`   | KL Divergence Minimization          | KL 散度最小化  |
| `npo`  | Negative Preference Optimization    | 负偏好优化     |
| `manu` | Modality-Aware Neuron Unlearning    | 模态感知神经元遗忘 |
| `our`  | **MIP-Editor**                      | 本文方法      |


---

## 6. 预计算影响力路径

位置: `mip_workspace/influential_paths/`


| 文件                                                 | 大小        | 描述                   |
| -------------------------------------------------- | --------- | -------------------- |
| `multi_fisher_all_mllmu_Qwen2.5-VL-3B-Instruct.pt` | 1.6 GB    | MLLMU 全量 Fisher (视觉) |
| `multi_fisher_forget*_clear_*.pt`                  | 8-83 MB   | CLEAR Fisher (视觉)    |
| `text_ja_all_mllmu_*.pt`                           | 58 MB     | MLLMU IG (语言)        |
| `text_ja_forget*_clear_*.pt`                       | 59-617 MB | CLEAR IG (语言)        |


---

## 7. 主要配置参数

```python
model = "Qwen2.5-VL-3B-Instruct"      # 模型名称
dataset = "clear"                      # 数据集: clear / mllmu
unlearning = "our"                     # 遗忘方法
batch_size = 6                         # 批次大小
epochs = 1                             # 训练轮数
finetune_epochs = 1                    # 微调轮数
learning_rate = 5e-4                   # 学习率
forget_ratio = 5                       # 遗忘比例: 1%, 5%, 10%
topk = 5                               # top-k 影响力神经元
use_neuron_cache_flag = True          # 使用预计算路径
device = "cuda"                        # 设备
```

---

## 8. 模型层架构模式

### 视觉层 (Vision Layers)

```python
# Qwen 模型
model.visual.blocks[layer].mlp

# LLaVA 模型
model.vision_tower.vision_model.encoder.layers[layer].mlp
```

### 语言层 (Language Layers)

```python
# Qwen 模型
model.model.layers[layer].mlp
# 或
model.model.language_model.layers[layer].mlp

# LLaVA 模型
model.language_model.model.layers[layer].mlp
```

---

## 9. 依赖环境

关键依赖 (`requirements.txt`):

```
transformers==4.57.1
torch==2.5.1
peft==0.17.1              # LoRA 微调
flash_attn==2.7.2.post1   # 快速注意力
datasets==3.1.0
scikit_learn==1.4.0
rouge_score==0.1.2
bert_score==0.3.13
```

**环境激活**: `conda activate mip-editor`

---

## 10. 使用示例

### 基本运行

```bash
conda activate mip-editor
cd MIP-Editor
python main.py --unlearning our --model Qwen2.5-VL-3B-Instruct --dataset clear
```

### 指定参数

```bash
python main.py \
    --unlearning our \
    --model Qwen2.5-VL-3B-Instruct \
    --dataset clear \
    --batch_size 6 \
    --epochs 1 \
    --forget_ratio 5 \
    --topk 5
```

---

## 11. 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (编排层 - Orchestration)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     load_model.py   load_data.py   train_eval.py
          │               │               │
          ▼               ▼               ▼
     PEFT Model       Datasets      Training Loop
     (Qwen2.5-VL)    (MLLMU/CLEAR)    + Evaluation
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
                      ours.py
                   (MIP-Editor 核心)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       fisher.py        ig.py      partial_linear.py
      (视觉 IG)      (文本 IG)      (选择性训练)
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
                  Adaptive RMU 微调
```

---

## 12. 快速参考


| 需求      | 查找位置                                         |
| ------- | -------------------------------------------- |
| 修改主流程   | `main.py`                                    |
| 实现新遗忘方法 | `ours.py` 或 `unlearning.py`                  |
| 添加新数据集  | `load_data.py`, `mllmu_bench.py`, `clear.py` |
| 调整神经元剪枝 | `utils.py` (`model_layer_prune_to_zero`)     |
| 视觉模态重要性 | `fisher.py`                                  |
| 文本模态重要性 | `ig.py`                                      |
| 选择性微调   | `partial_linear.py`                          |
| 评估指标    | `metrics/` 目录                                |


---

