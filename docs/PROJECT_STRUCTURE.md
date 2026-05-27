# MIP-Editor 项目结构

更新时间：2026-05-27

本文档按当前工作区重新整理 `MIP-Editor/` 的实际文件结构。已删除的旧 Step2-Step8 单步文档不再列入；当前 Step2-Step8 的实现说明统一查看：

```text
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
```

当前代码库路径：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
```

真实实验 workspace 主要位于仓库父目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

仓库内也存在一个轻量 `mip_workspace/` 目录，目前只包含少量本地输出占位，不作为主实验 workspace。

## 1. 顶层结构

```text
MIP-Editor/
├── README.md
├── requirements.txt
├── main.py
├── ours.py
├── unlearning.py
├── manu.py
├── fisher.py
├── ig.py
├── load_model.py
├── load_data.py
├── train_eval.py
├── score_by_llm.py
├── clear.py
├── mllmu_bench.py
├── test_data.py
├── partial_linear.py
├── rmu_layer_utils.py
├── myqwen2vl.py
├── myqwen25vl.py
├── utils.py
├── write_log.py
├── test_remote_scoring.py
├── test_rmu_layer_resolution.py
├── causal_mip/
├── metrics/
├── docs/
├── pictures/
└── mip_workspace/
```

顶层主要文件说明：

| 文件 | 作用 |
| --- | --- |
| `main.py` | 主入口，负责训练、遗忘、评估和 causal MIP 扩展入口参数 |
| `ours.py` | MIP-Editor 核心遗忘算法 |
| `unlearning.py` | GA / KL / NPO 等基线遗忘方法 |
| `manu.py` | MANU 基线剪枝方法 |
| `fisher.py` | 视觉模态 Fisher / IFI 影响力计算 |
| `ig.py` | 文本模态 IG / IGI 影响力计算 |
| `load_model.py` | 模型加载 |
| `load_data.py` | 数据加载 |
| `train_eval.py` | 训练和评估主逻辑 |
| `score_by_llm.py` | 本地/远程 LLM judge 评分 |
| `clear.py` | CLEAR 数据集定义 |
| `mllmu_bench.py` | MLLMU-Bench 数据集定义 |
| `test_data.py` | 测试数据处理 |
| `partial_linear.py` | masked RMisU 使用的部分线性层封装 |
| `rmu_layer_utils.py` | RMU 层解析与工具函数 |
| `myqwen2vl.py` | Qwen2-VL 自定义实现 |
| `myqwen25vl.py` | Qwen2.5-VL 自定义实现 |
| `utils.py` | 通用工具函数 |
| `write_log.py` | 日志写入工具 |
| `test_remote_scoring.py` | 远程评分连通性测试 |
| `test_rmu_layer_resolution.py` | RMU 层解析测试 |

## 2. causal_mip 扩展模块

`causal_mip/` 是当前 Step2-Step8 causal selective editing 路线的主要代码目录。

```text
causal_mip/
├── __init__.py
├── project_paths.py
├── test_step2.py
├── test_step3_data_pairs.py
├── test_step3_path_pair_bindings.py
├── test_step4_interventions.py
├── test_step5_causal_scores.py
├── test_step6_classify_paths.py
├── test_step7_masked_rmisu.py
├── test_step8_protocol.py
├── path_localization/
├── data_pairs/
├── interventions/
├── causal_scores/
├── editing/
└── evaluation/
```

测试文件：

| 文件 | 覆盖范围 |
| --- | --- |
| `test_step2.py` | candidate path schema / export 基础行为 |
| `test_step3_data_pairs.py` | causal pair 构造 |
| `test_step3_path_pair_bindings.py` | sample-level path/pair 绑定 |
| `test_step4_interventions.py` | activation cache / patching / ablation / restoration |
| `test_step5_causal_scores.py` | Nec / Suf / Ret score 计算 |
| `test_step6_classify_paths.py` | 四类路径分类 |
| `test_step7_masked_rmisu.py` | masked RMisU mask 和训练封装 |
| `test_step8_protocol.py` | Step8 主协议 gating 逻辑 |

### 2.1 path_localization

Step2：候选路径定位与导出。

```text
causal_mip/path_localization/
├── __init__.py
├── path_schema.py
├── mip_topk_wrapper.py
├── beam_search_paths.py
├── cross_modal_path_builder.py
└── cached_path_export.py
```

| 文件 | 作用 |
| --- | --- |
| `path_schema.py` | 统一 path / node 数据结构 |
| `mip_topk_wrapper.py` | 文本 IG / IGI top-k path 包装 |
| `beam_search_paths.py` | 视觉 Fisher / IFI top-k path 搜索 |
| `cross_modal_path_builder.py` | vision-text 跨模态路径构建 |
| `cached_path_export.py` | 从缓存导出 `P_cand.jsonl` |

### 2.2 data_pairs

Step3：反事实 pair 构造与 path/pair 绑定。

```text
causal_mip/data_pairs/
├── __init__.py
├── build_pairs.py
├── bind_paths_to_pairs.py
├── text_corruption.py
├── image_corruption.py
└── hard_retain_builder.py
```

| 文件 | 作用 |
| --- | --- |
| `build_pairs.py` | 构造 `forget_clean` / `forget_corrupt` / retain pair |
| `bind_paths_to_pairs.py` | 将 candidate path 绑定到具体 causal pair |
| `text_corruption.py` | STR 风格文本腐蚀 |
| `image_corruption.py` | 图像腐蚀 |
| `hard_retain_builder.py` | hard retain / counterfactual retain 样本构造 |

### 2.3 interventions

Step4：activation cache、ablation 与 restoration patching。

```text
causal_mip/interventions/
├── __init__.py
├── hooks.py
├── activation_cache.py
├── patching.py
├── ablation.py
└── restoration.py
```

| 文件 | 作用 |
| --- | --- |
| `hooks.py` | forward / pre-forward hook 工具 |
| `activation_cache.py` | clean/corrupt activation 缓存与输入构建 |
| `patching.py` | 通用 path-level patch 执行 |
| `ablation.py` | zero/noise ablation |
| `restoration.py` | clean-to-corrupt activation restoration |

### 2.4 causal_scores

Step5-Step6：因果分数计算与路径分类。

```text
causal_mip/causal_scores/
├── __init__.py
├── metrics.py
├── necessity.py
├── sufficiency.py
├── retain_impact.py
├── classify_paths.py
└── build_scores.py
```

| 文件 | 作用 |
| --- | --- |
| `metrics.py` | Step5 统一评分编排 |
| `necessity.py` | `Nec(P)` 必要性计算 |
| `sufficiency.py` | `Suf(P)` 充分性计算 |
| `retain_impact.py` | `Ret(P)` retain impact 计算 |
| `classify_paths.py` | `P_forget/P_shared/P_retain/P_irrelevant` 四分类 |
| `build_scores.py` | 批量 path score 生成入口 |

### 2.5 editing

Step7：选择性编辑。

```text
causal_mip/editing/
├── __init__.py
└── masked_rmisu.py
```

| 文件 | 作用 |
| --- | --- |
| `masked_rmisu.py` | 基于 `P_forget/P_shared` 的 masked RMisU 编辑训练 |

### 2.6 evaluation

Step8：最终评估与固定协议。

```text
causal_mip/evaluation/
├── __init__.py
├── step8_final_eval.py
├── step8_protocol.py
└── full_clear_remote_eval.py
```

| 文件 | 作用 |
| --- | --- |
| `step8_final_eval.py` | 基于 Step3 pair 的 val eval |
| `step8_protocol.py` | pair screen + Full CLEAR 的固定 Step8 主协议 |
| `full_clear_remote_eval.py` | Full CLEAR 四任务远程评分与可恢复评估 |

## 3. metrics 指标模块

`metrics/` 保留一组可复用评估指标实现。各子目录通常包含指标实现文件与 `app.py` 包装入口。

```text
metrics/
├── accuracy/
├── bertscore/
├── bleu/
├── bleurt/
├── f1/
├── glue/
├── google_bleu/
├── mae/
├── mape/
├── mase/
├── mauve/
├── mean_iou/
├── mse/
├── perplexity/
├── precision/
├── recall/
├── roc_auc/
├── rouge/
├── sacrebleu/
├── squad/
├── squad_v2/
└── super_glue/
```

常用指标：

| 目录 | 指标 |
| --- | --- |
| `accuracy/` | accuracy |
| `bleu/` | BLEU / NMT BLEU |
| `rouge/` | ROUGE |
| `bertscore/` | BERTScore |
| `sacrebleu/` | SacreBLEU |
| `perplexity/` | perplexity |
| `precision/`, `recall/`, `f1/` | 分类指标 |
| `squad/`, `squad_v2/` | QA 指标 |
| `glue/`, `super_glue/` | GLUE / SuperGLUE 指标 |

## 4. docs 文档目录

当前 `docs/` 目录：

```text
docs/
├── PROJECT_STRUCTURE.md
├── CHIP_EDITOR_TECHNICAL_ROADMAP.md
├── STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
├── STEP2_TO_STEP8_CODE_REVIEW_REPORT.md
├── CHIP_EDITOR_ROUTE_IMPLEMENTATION_AUDIT.md
├── STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md
├── EXPERIMENT_RESULT_STEP9_BOUND_PROJECTOR_0526_214028.md
├── EXPERIMENT_RESULT_CHIP_FULL_0526_1348.md
├── EXPERIMENT_RESULT_0430_141446.md
├── MACHINE_UNLEARNING_SURVEY_OPTIMIZATION_ROUTE.md
├── MIP_Editor_Summary.md
└── MIP_Editor_项目文档.md
```

文档说明：

| 文件 | 作用 |
| --- | --- |
| `PROJECT_STRUCTURE.md` | 当前项目结构说明 |
| `CHIP_EDITOR_TECHNICAL_ROADMAP.md` | CHIP-Editor 四层技术路线图 |
| `STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md` | Step2-Step8 汇总实现说明，替代旧单步文档 |
| `STEP2_TO_STEP8_CODE_REVIEW_REPORT.md` | Step2-Step8 代码审查报告 |
| `CHIP_EDITOR_ROUTE_IMPLEMENTATION_AUDIT.md` | 技术路线实现审计 |
| `STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md` | Step9 闭环实验 runbook |
| `EXPERIMENT_RESULT_STEP9_BOUND_PROJECTOR_0526_214028.md` | 最新 Step9 bound projector 实验结果 |
| `EXPERIMENT_RESULT_CHIP_FULL_0526_1348.md` | 0526 CHIP full 实验记录 |
| `EXPERIMENT_RESULT_0430_141446.md` | 原始 0430 实验结果 |
| `MACHINE_UNLEARNING_SURVEY_OPTIMIZATION_ROUTE.md` | 基于 unlearning survey 的优化路线 |
| `MIP_Editor_Summary.md` | MIP-Editor 概览总结 |
| `MIP_Editor_项目文档.md` | 项目中文说明文档 |

历史清理记录：以下旧单步文档已从 `docs/` 删除，不再作为可访问文档维护。

```text
docs/Step 2 COMPLETION_RECORD.md
docs/STEP3_CAUSAL_PAIRS_FLOW.md
docs/STEP4_ACTIVATION_PATCHING_FLOW.md
docs/STEP5_CAUSAL_SCORES_FLOW.md
docs/STEP6_PATH_CLASSIFICATION_FLOW.md
docs/STEP7_MASKED_RMISU_FLOW.md
docs/STEP8_FINAL_EVALUATION_FLOW.md
```

这些旧文档内容已合并到：

```text
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
```

## 5. pictures

```text
pictures/
├── mainfig.png
└── mainfig.pdf
```

用于论文/README/说明文档中的主图资源。

## 6. workspace 与实验产物

### 6.1 主实验 workspace

主实验产物位于仓库父目录：

```text
../mip_workspace/
├── llms/
├── datasets/
├── influential_paths/
└── outputs/
```

当前关键目录：

```text
../mip_workspace/llms/Qwen2.5-VL-3B-Instruct/
../mip_workspace/datasets/CLEAR/
../mip_workspace/datasets/MLLMU-Bench/
../mip_workspace/influential_paths/
../mip_workspace/outputs/
```

关键输出示例：

```text
../mip_workspace/outputs/paths/
../mip_workspace/outputs/scores/
../mip_workspace/outputs/checkpoints/
../mip_workspace/outputs/full_clear_remote_eval/
../mip_workspace/outputs/logs/
../mip_workspace/outputs/full_clear_step9_bound_projector_0526_214028.json
../mip_workspace/outputs/full_clear_peft_baseline_step9_bound_projector_0526_214028.json
../mip_workspace/outputs/step8_protocol_step9_bound_projector_0526_214028.json
../mip_workspace/outputs/masked_rmisu_step9_bound_projector_0526_214028.json
```

### 6.2 仓库内 workspace

仓库内当前也存在：

```text
MIP-Editor/mip_workspace/
└── outputs/
    └── data_pairs/
```

该目录不是当前完整实验使用的主 workspace。后续复现实验时优先使用父目录的：

```text
../mip_workspace/
```

## 7. 当前推荐阅读顺序

如果目标是理解当前 causal selective editing 路线，建议按以下顺序阅读：

1. `docs/CHIP_EDITOR_TECHNICAL_ROADMAP.md`
2. `docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md`
3. `docs/STEP2_TO_STEP8_CODE_REVIEW_REPORT.md`
4. `docs/CHIP_EDITOR_ROUTE_IMPLEMENTATION_AUDIT.md`
5. `docs/STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md`
6. `docs/EXPERIMENT_RESULT_STEP9_BOUND_PROJECTOR_0526_214028.md`

## 8. 清理说明

以下目录/文件属于缓存、备份或临时文件，不作为项目主结构维护：

```text
.git/
.git_old_20260526_183907/
__pycache__/
*/__pycache__/
.git_old_backup_name
Untitled
```

这些内容在遍历项目时存在，但不应作为源码结构或文档结构的长期依赖。
