# MIP-Editor 项目框架文档

## 项目概述

**MIP-Editor** 是 AAAI 2026 (Oral Presentation) 论文 *"Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models"* 的官方实现。

### 核心功能

在多模态大语言模型 (MLLMs) 中进行**跨模态遗忘 (Cross-Modal Unlearning)**，即选择性删除模型中不需要的知识（如隐私数据、有害关联、知识产权内容），同时保留模型的通用能力。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        本地机器 (训练)                            │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ Qwen2.5-VL-3B     │         │  评估模块           │        │
│  │ (本地 GPU)         │────────▶│  train_eval.py     │        │
│  └─────────────────────┘         └──────────┬──────────┘        │
└──────────────────────────────────────────────┼───────────────────┘
                                               │ HTTP API 调用
                                               │ (OpenAI 兼容)
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        超算服务器 (评分)                         │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ API 服务           │◀────────│  Qwen2.5-VL-7B    │        │
│  │ (端口 21936)       │         │  (评分模型)          │        │
│  └─────────────────────┘         └─────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**说明：**

- **训练模型**：本地加载 Qwen2.5-VL-3B-Instruct
- **评分模型**：调用超算服务器上的 Qwen2.5-VL-7B-Instruct（远程 API）

---

## 目录结构

```
MIP-Editor/                       # 核心代码库
    ├── main.py                       # 主入口文件 ⭐
    ├── ours.py                       # MIP-Editor 核心算法 ⭐
    ├── fisher.py                     # 视觉模态影响力计算（Fisher Information）
    ├── ig.py                         # 文本模态影响力计算（Integrated Gradients）
    ├── unlearning.py                 # 基线遗忘方法（GA、KL、NPO）
    ├── manu.py                       # MANU 基线剪枝方法
    ├── utils.py                      # 工具函数
    ├── train_eval.py                 # 训练和评估函数
    ├── load_model.py                 # 模型加载
    ├── load_data.py                  # 数据加载
    ├── clear.py                      # CLEAR 数据集定义
    ├── mllmu_bench.py                # MLLMU-Bench 数据集定义
    ├── test_data.py                  # 测试数据处理
    ├── score_by_llm.py               # LLM 评分模块（支持本地/远程 API）⭐
    ├── test_remote_scoring.py        # 远程评分测试脚本
    ├── partial_linear.py             # 部分线性层（用于选择性训练）
    ├── myqwen2vl.py                  # Qwen2-VL 自定义实现
    ├── myqwen25vl.py                 # Qwen2.5-VL 自定义实现
    ├── write_log.py                  # 日志记录
    ├── rmu_layer_utils.py            # RMU 层工具函数
    ├── test_rmu_layer_resolution.py  # RMU 层分辨率测试
    ├── requirements.txt              # Python 依赖
    ├── README.md                     # 项目说明
    │
    ├── causal_mip/                  # 因果 MIP 扩展模块 ⭐⭐
    │   ├── __init__.py
    │   ├── test_step2.py
    │   ├── test_step3_data_pairs.py
    │   ├── test_step4_interventions.py
    │   ├── test_step5_causal_scores.py
    │   ├── test_step6_classify_paths.py
    │   ├── test_step7_masked_rmisu.py
    │   ├── data_pairs/              # Step 3 反事实 pair 构造子模块
    │   │   ├── __init__.py
    │   │   ├── build_pairs.py       # NOTICE-inspired pair 构造入口
    │   │   ├── text_corruption.py   # STR 风格文本腐蚀
    │   │   ├── image_corruption.py  # SMP 风格图像腐蚀
    │   │   └── hard_retain_builder.py  # hard retain / counterfactual retain
    │   ├── interventions/           # Step 4 激活缓存与路径 patching 子模块
    │   │   ├── __init__.py
    │   │   ├── hooks.py             # ROME-inspired tracing hooks（支持 forward / pre-forward hook）
    │   │   ├── activation_cache.py  # path activation cache + pair input builder（语言侧 + Qwen2.5-VL 视觉 block）
    │   │   ├── patching.py          # 通用路径级 patch 执行器（down_proj 输入侧干预，支持 2D/3D activation）
    │   │   ├── ablation.py          # 零化/噪声 ablation
    │   │   └── restoration.py       # clean→corrupt activation restoration
    │   ├── causal_scores/           # Step 5 因果评分子模块
    │   │   ├── __init__.py
    │   │   ├── metrics.py           # Step 5 统一评分编排
    │   │   ├── necessity.py         # Nec(P)
    │   │   ├── sufficiency.py       # Suf(P)
    │   │   ├── retain_impact.py     # Ret(P)
    │   │   ├── classify_paths.py    # Step 6 路径四分类入口
    │   │   └── build_scores.py      # 批量 path score 生成入口
    │   ├── editing/                 # Step 7 masked RMisU 子模块
    │   │   ├── __init__.py
    │   │   └── masked_rmisu.py      # P_forget/P_shared masked RMisU（语言侧训练已跑通，视觉 block mask 已支持）
    │   ├── evaluation/              # Step 8 最终评估子模块
    │   │   ├── __init__.py
    │   │   └── step8_final_eval.py  # Step3 pair + Step7 checkpoint final eval
    │   └── path_localization/        # 路径定位子模块
    │       ├── __init__.py
    │       ├── path_schema.py        # 路径数据结构定义
    │       ├── mip_topk_wrapper.py  # IG Top-K 路径计算
    │       ├── beam_search_paths.py  # Fisher Top-K 路径计算
    │       ├── cross_modal_path_builder.py  # 跨模态路径构建
    │       └── cached_path_export.py # 缓存路径导出
    │
    ├── metrics/                     # 评估指标模块
    │   ├── accuracy/
    │   ├── bleu/
    │   ├── rouge/
    │   ├── bertscore/
    │   ├── f1/
    │   ├── glue/
    │   ├── sacrebleu/
    │   ├── squad/
    │   ├── squad_v2/
    │   ├── super_glue/
    │   ├── perplexity/
    │   ├── precision/
    │   ├── recall/
    │   ├── mae/
    │   ├── mse/
    │   ├── mape/
    │   ├── mean_iou/
    │   ├── mase/
    │   ├── mauve/
    │   ├── roc_auc/
    │   ├── bleurt/
    │   └── google_bleu/
    │
    ├── docs/                        # 开发文档
    │   └── Step 2 COMPLETION_RECORD.md
    │   └── STEP7_MASKED_RMISU_FLOW.md
    │   └── STEP8_FINAL_EVALUATION_FLOW.md
    │
    ├── pictures/                     # 图片资源
    │
    ├── STEP3_CAUSAL_PAIRS_FLOW.md          # Step 3 pair 构造流程
    ├── STEP4_ACTIVATION_PATCHING_FLOW.md   # Step 4 activation patching 流程
    ├── STEP5_CAUSAL_SCORES_FLOW.md         # Step 5 因果评分流程
    ├── STEP6_PATH_CLASSIFICATION_FLOW.md   # Step 6 路径四分类完成记录
    ├── CHIP_EDITOR_TECHNICAL_ROADMAP.md   # 技术路线图
    ├── CAUSAL_MVP_INTERFACE_CHECKLIST.md   # 因果 MVP 接口检查清单
    ├── OOM_SOLUTION_RMU_CACHE.md           # OOM 解决方案文档
    └── EXPERIMENT_RESULT_0430_141446.md    # 实验结果记录
```

### 当前 workspace 关键产物

当前真实链路使用的模型、数据和中间产物位于 `mip_workspace/`：

```text
mip_workspace/
    ├── llms/
    │   └── Qwen2.5-VL-3B-Instruct/              # 本地真实评分/训练模型权重
    ├── datasets/
    │   └── CLEAR/
    │       ├── forget5+tofu/                    # Step 3 forget_clean
    │       ├── forget5_perturbed/               # Step 3 forget_corrupt
    │       └── retain95+tofu/                   # Step 3 retain_clean
    └── outputs/
        ├── paths/
        │   └── P_cand.jsonl                     # Step 2 候选路径，2312 条
        ├── causal_pairs_train.jsonl             # Step 3 train pairs，170 条
        ├── causal_pairs_val.jsonl               # Step 3 val pairs，18 条
        ├── path_scores_real_smoke_cuda.jsonl    # Step 5 真实 CUDA 1x1 冒烟结果
        ├── path_scores_real_text_5x5_cuda.jsonl # Step 5 真实 CUDA text 5x5 结果
        └── path_scores_real_vision_text_5x5_cuda.jsonl
                                                   # Step 5 真实 CUDA vision_text 5x5 结果
        ├── path_scores_real_text_20x50_cuda.jsonl
        │                                          # Step 5 text 20 pairs x 50 paths
        ├── path_scores_real_vision_text_20x8_cuda.jsonl
        │                                          # Step 5 vision_text 20 pairs x all 8 paths
        ├── scores/
        │   ├── step5_vision_smoke_0522.jsonl      # Step 5 vision 1x1 视觉侧 patching smoke
        │   └── step5_vision_text_smoke_0522.jsonl # Step 5 vision_text 1x1 视觉+语言 patching smoke
        ├── masked_rmisu_step7_smoke_0521.json
        │                                          # Step 7 无 preserve 2-step smoke test
        ├── masked_rmisu_step7_preserve_smoke_0521.json
        │                                          # Step 7 preserve 2-step smoke test
        ├── masked_rmisu_step7_full1epoch_0521.json
        │                                          # Step 7 preserve 1 epoch masked RMisU 小实验 summary
        ├── masked_rmisu_step7_full1epoch_ckpt_0521.json
        │                                          # Step 7 checkpoint 保存版 1 epoch summary
        ├── step8_eval_smoke_val_max1_0522.json
        │                                          # Step 8 Step7 checkpoint val smoke eval
        ├── step8_eval_val_full_0522.json
        │                                          # Step 8 Step7 checkpoint val full eval
        ├── step8_eval_peft_baseline_val_full_0522.json
        │                                          # Step 8 PEFT baseline val full eval
        ├── step8_eval_baseline_val_full_0522.json
        │                                          # Step 8 raw base model val full eval
        ├── masked_rmisu_step7_quantile75_full1epoch_0522.json
        │                                          # Step 7 Step6 quantile75 mask summary
        ├── masked_rmisu_step7_quantile75_ce01_full1epoch_0522.json
        │                                          # Step 7 quantile75 + forget CE summary
        ├── step8_eval_quantile75_val_full_0522.json
        │                                          # Step 8 quantile75 checkpoint eval
        ├── step8_eval_quantile75_ce01_val_full_0522.json
        │                                          # Step 8 quantile75 + forget CE checkpoint eval
        ├── checkpoints/
        │   └── masked_rmisu_step7_full1epoch_ckpt_0521/
        │       ├── model-00001-of-00002.safetensors
        │       ├── model-00002-of-00002.safetensors
        │       ├── model.safetensors.index.json
        │       ├── config.json
        │       ├── generation_config.json
        │       ├── preprocessor_config.json
        │       └── tokenizer.json
        │   └── masked_rmisu_step7_quantile75_full1epoch_0522/
        │   └── masked_rmisu_step7_quantile75_ce01_full1epoch_0522/
        └── paths/
            ├── step6_text_5x5/                   # Step 6 text 5x5 四分类结果
            ├── step6_vision_text_5x5/            # Step 6 vision_text 5x5 四分类结果
            └── step6_combined_5x5/               # Step 6 text+vision_text 合并四分类结果
            ├── step6_v1/                         # Step 6 扩大规模主路径集合
            └── step6_v1_quantile75/              # Step 6 75 分位阈值保守对照
```

截至 2026-05-21，`P_cand.jsonl` 的路径类型分布为：

```text
text=1552
vision=752
vision_text=8
total=2312
```

真实 CUDA 验证已使用 `Qwen2.5-VL-3B-Instruct + CLEAR + P_cand.jsonl + causal_pairs_train.jsonl` 跑通：

```text
text        5 pairs x 5 paths = 25 records, status={'ok': 25}, patchable_nodes=[36]
vision      1 pair  x 1 path  = 1 record,  status=ok,       patchable_nodes=32
vision_text 1 pair  x 1 path  = 1 record,  status=ok,       patchable_nodes=3
```

说明：Step 4/5 已支持语言侧 `down_proj` 与 Qwen2.5-VL visual block `down_proj` 输入侧 patching。`mm_projector` 仍未纳入 patching，因此 4 节点 cross-modal path 当前通常有 3 个 patchable nodes。

Step 6 已基于上述 Step 5 输出生成四类路径集合：

```text
combined 5x5:
P_forget=2
P_shared=1
P_retain=2
P_irrelevant=5
```

扩大规模后的 `step6_v1` 已生成：

```text
text 20x50 + vision_text 20x8:
score_records=1160
classified_paths=58
P_forget=27
P_shared=12
P_retain=6
P_irrelevant=13
```

保守对照 `step6_v1_quantile75`：

```text
P_forget=8
P_shared=3
P_retain=12
P_irrelevant=35
```

输出目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7/mip_workspace/outputs/paths/step6_combined_5x5/
```

Step 7 已完成真实 Qwen2.5-VL-3B masked RMisU 训练入口验证：

```text
无 preserve smoke:
masked_rmisu_step7_smoke_0521.json
num_loss_records=2
num_modules=34
num_editable_neurons=123
loss: 74.0 -> 70.0

preserve smoke:
masked_rmisu_step7_preserve_smoke_0521.json
num_loss_records=2
num_modules=34
num_editable_neurons=123
loss: 37.0 -> 35.0

preserve 1 epoch 小实验:
masked_rmisu_step7_full1epoch_0521.json
num_loss_records=1790
num_modules=34
num_editable_neurons=123
loss_mean=31.31417597765363
loss: 37.0 -> 30.5

checkpoint 保存版 preserve 1 epoch:
masked_rmisu_step7_full1epoch_ckpt_0521.json
checkpoints/masked_rmisu_step7_full1epoch_ckpt_0521/
checkpoint_size=7.1G
num_loss_records=1790
num_modules=34
num_editable_neurons=123
merged_partial_linear_modules=68
loss_mean=31.316550279329608
loss: 37.0 -> 30.5
```

说明：Step 7 小实验使用 `--skip_post_unlearning_eval`。checkpoint 保存版已通过 `AutoProcessor.from_pretrained(...)` 和 `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` 加载检查，可供后续 Step 8 使用。

Step 8 已基于 Step3 val pairs 完成 Step7 checkpoint 与 PEFT baseline 对比：

```text
eval_set=causal_pairs_val.jsonl
pairs=18
examples=72

PEFT baseline:
forget_clean name_hit_rate=0.6111
hard_retain name_hit_rate=0.6944
counterfactual_retain name_hit_rate=0.7222

Step7 masked RMisU:
forget_clean name_hit_rate=0.7778
hard_retain name_hit_rate=0.7222
counterfactual_retain name_hit_rate=0.7222
```

结论：当前 Step7 checkpoint 未达成预期遗忘，`forget_clean` 的目标名字泄露率相对 PEFT baseline 上升。

Step6/Step7 优化后结果：

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

结论：`quantile75 + forget CE 0.1` 是当前最佳版本，但只是回到 PEFT baseline，尚未形成强于 baseline 的成功遗忘。

---

## 核心文件详解

### 1. main.py - 主入口文件 ⭐

程序的主入口，负责配置参数和执行完整流程。

**主要功能：**

- 定义所有命令行参数
- 加载模型和数据
- 调用遗忘算法
- 执行评估

**关键参数：**

```python
--model              # 模型名称 (Qwen2.5-VL-3B-Instruct)
--dataset            # 数据集 (clear / mllmu)
--unlearning         # 遗忘方法 (our / ga / kl / npo / manu)
--forget_ratio       # 遗忘比例 (1 / 5 / 10)
--topk               # 选取的神经元数量
--batch_size         # 批大小
--learning_rate      # 学习率
--eval_flag          # 是否评估
--train_flag         # 是否训练
--use_neuron_cache_flag  # 是否使用缓存的影响力路径
```

**执行流程：**

1. 加载 Processor 和数据集
2. 加载/训练模型
3. 执行遗忘算法
4. 评估结果

---

### 2. ours.py - MIP-Editor 核心算法 ⭐

实现论文提出的跨模态遗忘方法。

**算法流程：**

1. 加载预计算的影响力路径（Fisher / IG 分数）
2. 合并 LoRA 并卸载为全量模型
3. 在视觉层识别关键神经元路径（`multi_idxs_forget_visual`）
4. 在语言层识别关键神经元路径（`text_idxs_forget_model`）
5. 使用 Hook 技术将关键路径置零
6. 执行 `adaptive_rmu_finetune` 进行表示误导微调

**关键函数：**

```python
def our(model, forget_loader, forget_text_loader, forget_indices,
        retain_loader, retain_text_loader, retain_indices,
        sampled_forget_loader, args)
```

---

### 3. fisher.py - Fisher Information 计算

使用 **Fisher Information** 识别视觉模态中与遗忘相关的神经元。

**核心函数：**

```python
def calculate_fisher(model, data_loader, args)
def compute_fisher_path(inputs, model, ig_total_step, top_k, batch_size, model_name)
def fisher_main(model, forget_loader, retain_loader, forget_indices, retain_indices, args)
```

**原理：**

- 对 FFN 激活路径计算梯度
- 使用路径积分近似 Fisher Information
- 识别对遗忘集影响最大的神经元

---

### 4. ig.py - Integrated Gradients 计算

使用 **Integrated Gradients** 识别文本模态中与遗忘相关的神经元。

**核心函数：**

```python
def calculate_ig(model, data_loader, topk, ig_total_step)
def compute_ig_path(inputs, model, ig_total_step, top_k, batch_size, model_name)
def ig_main(model, forget_text_loader, forget_indices, retain_text_loader, retain_indices, args)
```

---

### 5. unlearning.py - 基线遗忘方法

实现多个基线遗忘方法用于对比。


| 方法  | 函数                         | 描述          |
| --- | -------------------------- | ----------- |
| GA  | `ga_difference_training()` | 梯度上升/下降差分训练 |
| KL  | `kl_min()`                 | KL 散度最小化    |
| NPO | `npo()`                    | 负偏好优化       |


---

### 6. manu.py - MANU 剪枝方法

实现 MANU (Modality-Aware Neuron Unlearning) 基线方法。

**核心类：**

```python
class MANUPruner:
    def compute_importance(self, model, data_loader, tau=0.1, epsilon=1e-6)
    def prune(self, model, forget_scores, retain_scores, alpha=0.1, decay=0.1)
```

---

### 7. utils.py - 工具函数

提供各种辅助功能。

**主要函数：**

```python
# 答案检查
def check_answer(question, options, pred, truth)

# 数据处理
def data_process_clf_mllmu(dataset, processor, model, args, data_type)
def data_process_gen_mllmu(dataset, processor, model, args, data_type)

# 模型工具
def ffn_at_layer(llm, layer)      # 获取指定层的 FFN
def vb_at_layer(llm, layer)       # 获取视觉块的 MLP
def freeze_parameters(model, multi_idxs, text_idxs, apply_to_visual)

# 剪枝工具
def model_layer_prune_to_zero(model, prune_indices)
def visual_block_prune_to_zero(model, prune_indices)

# 路径工具
def get_unique_idxs(paths, scores, length)
def get_language_model_num_layers(model)
def get_vision_model_num_layers(model)
```

---

### 8. train_eval.py - 训练和评估

**训练函数：**

```python
def train(model, data_loader, optimizer, args, save=True)
def finetune(model, retain_loader, sampled_forget_loader, args)
def adaptive_rmu_finetune(updated_model, frozen_model, retain_loader, forget_loader, args)
```

**评估函数：**

```python
def evaluate_clf(dataset, processor, dataset_split, data_modality, args, model)
def evaluate_gen(forgetset, processor, dataset_split, data_modality, args, model)
def score_batch(preds, task, args)
```

---

### 9. load_model.py - 模型加载

**主要函数：**

```python
def load_base_model(args)          # 加载基础模型
def load_model(args, visual_trainable=True)  # 加载 LoRA 模型
def load_peft_model(args, trainable=False, identifier="")  # 加载 PEFT 检查点
```

**支持的模型：**

- `Qwen2.5-VL-3B-Instruct`
- `Qwen2.5-VL-7B-Instruct`
- `Qwen2-VL-2B-Instruct`
- `llava-1.5-7b-hf`

---

### 10. load_data.py - 数据加载

**主要函数：**

```python
def load_data_train(processor, args)
def load_data_forget(processor, args)
def load_data_retain(processor, args)
def load_data_finetune(processor, forget_indices, retain_indices, args)
```

**返回内容：**

- `forgetset`: 遗忘数据集
- `retainset`: 保留数据集
- `forget_loader` / `retain_loader`: 多模态 DataLoader
- `forget_text_loader` / `retain_text_loader`: 纯文本 DataLoader

---

### 11. 数据集文件

#### clear.py - CLEAR 数据集

```python
class CLEAR_Dataset       # 主数据集（训练用）
class CLEAR_Clf_Dataset   # 分类任务数据集
class CLEAR_Gen_Dataset   # 生成任务数据集
```

#### mllmu_bench.py - MLLMU-Bench 数据集

```python
class MMLMU_Dataset       # 主数据集
class MMLMU_Clf_Dataset   # 分类任务数据集
class MMLMU_Gen_Dataset   # 生成任务数据集
```

---

### 12. test_data.py - 测试数据处理

用于批量评估的数据集类：

```python
class MllmuGenDataset    # MLLMU 生成任务
class MllmuClfDataset    # MLLMU 分类任务
class ClearGenDataset     # CLEAR 生成任务
class ClearClfDataset     # CLEAR 分类任务
```

---

### 13. score_by_llm.py - LLM 评分

使用另一个 LLM 对预测结果进行评分。支持**本地模型**和**远程 API**两种模式。

#### 本地评分模式

```python
def load_llm(path, device)
def score_by_llm_batch(model, tokenizer, pred_list, batch_size, task, device)
```

#### 远程评分模式（超算服务器）

```python
class RemoteLLMClient          # 远程 API 客户端（OpenAI 兼容格式）
def load_remote_llm(base_url, api_key, model_name)  # 加载远程客户端
def score_by_remote_llm_batch(remote_client, pred_list, batch_size, task)  # 远程批量评分
```

**远程评分配置：**

```bash
--use_remote_scoring True \
--remote_scoring_url "http://210.40.56.85:21936/v1" \
--score_llm Qwen2.5-VL-7B-Instruct
```

---

### 14. partial_linear.py - 部分线性层

实现选择性训练的线性层分解：

```python
class PartialLinear(nn.Module):
    def __init__(self, original_linear, trainable_cols)
    def forward(self, x)
    def merge_to_linear(self)
```

---

### 15. metrics/ - 评估指标模块

包含 22 种评估指标：


| 目录           | 指标          | 说明               |
| ------------ | ----------- | ---------------- |
| bleu/        | BLEU        | n-gram 精确率       |
| rouge/       | ROUGE       | 召回率指标            |
| bertscore/   | BERTScore   | 基于 BERT 的相似度     |
| f1/          | F1          | 精确率和召回率的调和平均     |
| accuracy/    | Accuracy    | 准确率              |
| glue/        | GLUE        | 自然语言理解基准         |
| sacrebleu/   | SacreBLEU   | 标准化的 BLEU        |
| squad/       | SQuAD       | 阅读理解评估           |
| squad_v2/    | SQuAD v2    | 含无答案的阅读理解        |
| super_glue/  | SuperGLUE   | 高级 NLU 基准        |
| perplexity/  | Perplexity  | 语言模型困惑度          |
| precision/   | Precision   | 精确率              |
| recall/      | Recall      | 召回率              |
| mae/         | MAE         | 平均绝对误差           |
| mse/         | MSE         | 均方误差             |
| mape/        | MAPE        | 平均绝对百分比误差        |
| mean_iou/    | Mean IoU    | 平均交并比            |
| mase/        | MASE        | 缩放误差             |
| mauve/       | MAUVE       | 生成文本分布距离         |
| roc_auc/     | ROC-AUC     | 接收者操作特征曲线下面积     |
| bleurt/      | BLEURT      | 基于预训练的 BLEU      |
| google_bleu/ | Google BLEU | Google 的 BLEU 实现 |


---

### 16. causal_mip/ - 因果 MIP 扩展模块 ⭐⭐

扩展 MIP-Editor，使用因果验证区分遗忘特定路径和共享路径。

**path_localization/ 子模块：**
- `path_schema.py` - 路径数据结构定义（CandidatePath, PathNode）
- `mip_topk_wrapper.py` - IG Top-K 路径计算
- `beam_search_paths.py` - Fisher Top-K 路径计算
- `cross_modal_path_builder.py` - 跨模态路径构建
- `cached_path_export.py` - 缓存路径导出

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ load_model.py │    │ load_data.py  │    │ train_eval.py │
│   加载模型    │    │   加载数据    │    │   训练评估    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └──────────┬──────────┴─────────────────────┘
                   ▼
        ┌─────────────────────┐
        │  unlearning.py /    │
        │     ours.py         │
        │   遗忘算法执行       │
        └─────────────────────┘
                   │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│fisher.py│ │  ig.py  │ │ manu.py │
│ 视觉FI  │ │ 文本IG  │ │ MANU剪枝 │
└─────────┘ └─────────┘ └─────────┘
     │            │            │
     └────────────┴────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   utils.py          │
        │ 工具函数与剪枝钩子   │
        └─────────────────────┘

causal_mip/ (独立模块，可单独导出候选路径)
    │
    └─ path_localization/
           ├─ mip_topk_wrapper.py ──── IG Top-K 路径提取
           ├─ beam_search_paths.py ──── Fisher Top-K 路径提取
           ├─ cross_modal_path_builder.py ──── 跨模态路径构建
           └─ cached_path_export.py ──── 从缓存导出候选路径
```

---

## 使用方法

### 基本用法

```bash
# 使用 MIP-Editor 方法在 CLEAR 数据集上训练
python main.py --unlearning our --dataset clear --forget_ratio 5

# 使用 GA 基线方法
python main.py --unlearning ga --dataset clear --forget_ratio 5

# 使用 NPO 基线方法
python main.py --unlearning npo --dataset clear --forget_ratio 5

# 在 MLLMU 数据集上训练
python main.py --unlearning our --dataset mllmu --forget_ratio 10
```

### 完整参数示例

```bash
python main.py \
  --model Qwen2.5-VL-3B-Instruct \
  --dataset clear \
  --unlearning our \
  --forget_ratio 5 \
  --batch_size 6 \
  --learning_rate 5e-4 \
  --epochs 1 \
  --finetune_epochs 1 \
  --topk 5 \
  --eval_flag \
  --use_neuron_cache_flag True \
  --base_path /path/to/datasets \
  --llm_directory /path/to/llms \
  --output_file_path /path/to/output/
```

### 常用运行命令

**训练 + 评估（使用远程评分）：**

```bash
unset http_proxy https_proxy all_proxy socks_proxy
python main.py \
  --model Qwen2.5-VL-3B-Instruct \
  --dataset clear \
  --unlearning our \
  --forget_ratio 5 \
  --batch_size 4 \
  --train_flag \
  --eval_flag \
  --use_remote_scoring \
  --remote_scoring_url "http://210.40.56.85:21936/v1" \
  --score_llm Qwen2.5-VL-7B-Instruct

unset http_proxy https_proxy all_proxy socks_proxy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py \
  --model Qwen2.5-VL-3B-Instruct \
  --dataset clear \
  --unlearning our \
  --forget_ratio 5 \
  --batch_size 2 \
  --train_flag \
  --eval_flag \
  --use_remote_scoring \
  --remote_scoring_url "http://210.40.56.85:21936/v1" \
  --score_llm Qwen2.5-VL-7B-Instruct
```



> **注意**：运行前需禁用代理环境变量（`unset http_proxy ...`），否则远程 API 调用会失败。

### 远程评分配置（超算服务器）

当本地 GPU 显存不足以加载评分模型时，可使用部署在超算服务器上的 Qwen2.5-VL-7B-Instruct 进行远程评分：

```bash
python main.py \
  --model Qwen2.5-VL-3B-Instruct \
  --dataset clear \
  --unlearning our \
  --forget_ratio 5 \
  --eval_flag \
  --use_remote_scoring True \
  --remote_scoring_url "http://210.40.56.85:21936/v1" \
  --score_llm Qwen2.5-VL-7B-Instruct
```

**远程评分参数说明：**


| 参数                     | 类型   | 默认值                                                  | 说明            |
| ---------------------- | ---- | ---------------------------------------------------- | ------------- |
| `--use_remote_scoring` | bool | False                                                | 是否使用远程 API 评分 |
| `--remote_scoring_url` | str  | [http://localhost:9192/v1](http://localhost:9192/v1) | 超算 API 地址     |
| `--score_llm`          | str  | Qwen2.5-VL-7B-Instruct                               | 评分模型名称        |


---
