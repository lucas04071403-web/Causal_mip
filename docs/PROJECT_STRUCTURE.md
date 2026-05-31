# MIP-Editor 项目结构

更新时间：2026-05-31

本文档按当前工作区实际代码和 `causal_mip 阶段性成功实验与经验总结.md` 更新。当前主线不再是 Baseline MIP-Editor 或 Step9 bound projector，而是：

```text
causal_mip
+ target name token 对齐
+ name-aware path selection
+ projector-dim selective editing
+ Step8 cheap diagnostic gate
```

需要先明确：

```text
当前阶段不是最终 unlearning 成功。
当前阶段成功的是 causal path 定位、target-name 对齐、projector/projector-dim 编辑、
Step7 训练链路和 Step8 cheap diagnostic 闭环。
```

当前代码库路径：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
```

真实实验 workspace 主要位于仓库父目录：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

仓库内也存在轻量 `mip_workspace/`，目前只包含少量本地输出占位，不作为主实验 workspace。

---

## 1. 当前顶层结构

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
| `README.md` | 当前项目运行文档 |
| `requirements.txt` | Python 依赖 |
| `main.py` | 主入口，负责训练、遗忘、评估和 causal MIP 扩展入口参数 |
| `ours.py` | MIP-Editor 核心遗忘算法，已接入 masked RMisU / causal_mip 相关参数 |
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

---

## 2. causal_mip 扩展模块

`causal_mip/` 是当前 causal selective editing 主线代码目录。当前已经形成 Step2-Step8 的工程闭环，并在 Step10 阶段补上 target name token 对齐、projector-dim 选择和 name preference objective。

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
├── test_step8_probability_diagnostic.py
├── test_name_token_metrics.py
├── test_retain_aware_localization.py
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
| `test_step5_causal_scores.py` | Nec / Suf / Ret 与 name-token score 计算 |
| `test_step6_classify_paths.py` | 四类路径分类、name-aware 分类、projector top-k |
| `test_step7_masked_rmisu.py` | masked RMisU mask、projector/projector-dim、name objective |
| `test_step8_protocol.py` | Step8 主协议 gating 逻辑 |
| `test_step8_probability_diagnostic.py` | target name token probability diagnostic |
| `test_name_token_metrics.py` | target name token 定位、scoped CE / logprob |
| `test_retain_aware_localization.py` | retain-aware localization 与 projector-dim export |

---

## 3. Step2-Step8 当前实现结构

### 3.1 path_localization

Step2：候选路径定位、筛选与导出。

```text
causal_mip/path_localization/
├── __init__.py
├── path_schema.py
├── mip_topk_wrapper.py
├── beam_search_paths.py
├── cross_modal_path_builder.py
├── cached_path_export.py
├── saliency_specific_export.py
├── filter_candidates.py
├── node_specific_export.py
└── projector_dim_export.py
```

| 文件 | 作用 |
| --- | --- |
| `path_schema.py` | 统一 path / node 数据结构 |
| `mip_topk_wrapper.py` | 文本 IG / IGI top-k path 包装 |
| `beam_search_paths.py` | 视觉 Fisher / IFI top-k path 搜索 |
| `cross_modal_path_builder.py` | vision-text 跨模态路径构建 |
| `cached_path_export.py` | 从缓存导出 `P_cand.jsonl` |
| `saliency_specific_export.py` | 从 Step5 saliency scores 导出 forget-specific `P_cand` 和绑定文件 |
| `filter_candidates.py` | candidate path 过滤工具 |
| `node_specific_export.py` | node-specific candidate 导出 |
| `projector_dim_export.py` | projector 维度级 candidate 导出，是当前推荐路线的关键入口 |

### 3.2 data_pairs

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

### 3.3 interventions

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
| `activation_cache.py` | clean/corrupt activation 缓存与输入构建，支持 projector whole-vector 语义 |
| `patching.py` | 通用 path-level patch 执行，支持 `WHOLE_VECTOR_NEURON = -1` |
| `ablation.py` | zero/noise ablation |
| `restoration.py` | clean-to-corrupt activation restoration |

### 3.4 causal_scores

Step5-Step6：因果分数计算、name-token score、saliency specificity、稳定性分析与路径分类。

```text
causal_mip/causal_scores/
├── __init__.py
├── metrics.py
├── necessity.py
├── sufficiency.py
├── retain_impact.py
├── name_token_metrics.py
├── saliency_specificity.py
├── specificity_probe.py
├── stability_report.py
├── classify_paths.py
└── build_scores.py
```

| 文件 | 作用 |
| --- | --- |
| `metrics.py` | Step5 统一评分编排，支持 answer-level 与 name-token score |
| `necessity.py` | `Nec(P)` 必要性计算 |
| `sufficiency.py` | `Suf(P)` 充分性计算 |
| `retain_impact.py` | `Ret(P)` retain impact 计算 |
| `name_token_metrics.py` | target name token 定位、NameNec / NameSuf / NameRet、scoped CE/logprob |
| `saliency_specificity.py` | SalUn/SSD 风格 forget-vs-retain gradient/Fisher specificity 计算 |
| `specificity_probe.py` | 基于 Step5 聚合结果导出 specificity ranking / projector probe |
| `stability_report.py` | 路径分数稳定性与聚合报告 |
| `classify_paths.py` | `P_forget/P_shared/P_retain/P_irrelevant` 分类，支持 name-aware projector top-k |
| `build_scores.py` | 批量 path score 生成入口，支持 `--compute_name_token_scores` |

当前 Step5 新增字段：

```text
NameNec
NameSuf
NameRet
name_forget_effect
name_retain_impact
target_name
target_name_token_positions
name_match_status
```

当前 Step6 关键接口：

```bash
--name_aware_forget
--name_forget_quantile 0.75
--name_retain_quantile 0.75
--min_name_sufficiency 0.0
--max_forget_projector_paths 4
--projector_name_effect_ratio_threshold 0.0
--projector_topk_metric name_editable_score
```

`name_editable_score` 的用途：

```text
name_editable_score = name_forget_effect / (1e-6 + name_retain_impact)
```

它用于优先选择 target-name sensitive 且 retain 风险低的 projector / projector-dim path。

### 3.5 editing

Step7：选择性编辑。

```text
causal_mip/editing/
├── __init__.py
└── masked_rmisu.py
```

| 文件 | 作用 |
| --- | --- |
| `masked_rmisu.py` | 基于 `P_forget/P_shared` 的 masked RMisU 编辑训练，支持 name-token CE、projector whole-vector 和 projector-dim 编辑 |

当前有效 objective：

```text
activation_random
activation_random_ce
activation_random_answer_ce
activation_random_name_ce
ce_ascent
answer_ce_ascent
name_ce_ascent
name_preference_unlearning
redacted_name_preference
```

当前推荐重点：

```text
forget_objective = name_ce_ascent 或 redacted_name_preference
forget_ce_scope = name
projector_edit_mode 指向 Qwen projector edit module: visual.merger.mlp.0
projector-dim candidate 优先于 whole-vector projector
```

`name_preference_unlearning` 当前语义：

```text
负向：压低 target name token。
正向：保留 answer_without_name token。
```

`redacted_name_preference` 当前语义：

```text
负向：压低 target name token。
正向：在 name span 上监督 generic identity token，例如 "The person"。
```

当前实验结论：

```text
redacted_name_preference 已实现并通过 Step7 单测和 GPU smoke。
全答案 redacted positive 会抵消姓名压制，不再作为主配置。
name-span redacted positive 可恢复正向压制，但仍未通过 cheap gate。
当前最佳行为配置仍是 top8/31-dim + name_ce_ascent ce=0.30。
```

### 3.6 evaluation

Step8：最终评估、固定协议、概率诊断与 Full CLEAR remote eval。

```text
causal_mip/evaluation/
├── __init__.py
├── step8_final_eval.py
├── step8_protocol.py
├── step8_probability_diagnostic.py
├── full_clear_remote_eval.py
└── diagnose_projector_path_flow.py
```

| 文件 | 作用 |
| --- | --- |
| `step8_final_eval.py` | 基于 Step3 pair 的 val eval，包含 pair generation name_hit |
| `step8_protocol.py` | pair screen + Full CLEAR 的固定 Step8 主协议 |
| `step8_probability_diagnostic.py` | target name token CE / logprob / margin cheap diagnostic |
| `full_clear_remote_eval.py` | Full CLEAR 四任务远程评分与可恢复评估 |
| `diagnose_projector_path_flow.py` | projector path 在 Step5/Step6/Step7/Step8 中的流向诊断 |

当前 gate 原则：

```text
先跑 Step8 probability diagnostic 和 pair generation eval；
cheap gate 不通过，不进入 Full CLEAR。
```

推荐最小成功标准：

```text
forget_clean target_name_ce delta > 0.01
forget_clean name_hit <= baseline 0.6111
hard_retain name_hit >= 0.6444
counterfactual_retain name_hit >= 0.6722
```

---

## 4. 当前最新方法：causal_mip name-token 对齐路线

当前项目最新方法不是“每一步都完整跑一遍”的旧手册，而是下面这条闭环：

```text
Step2 projector-dim candidate
-> Step3 pair/path binding
-> Step5 NameNec / NameSuf / NameRet
-> Step6 name-aware classification + name_editable_score
-> Step7 projector-dim masked RMisU + name objective
-> Step8 cheap probability / generation gate
-> gate 通过后才进入 Full CLEAR
```

### 4.1 为什么从 Step9 转向 Step10

Step9 corrected 的核心失败点：

```text
P_forget 中没有 projector / vision_text path；
projector path 被 answer-level retain guard 吞入 P_shared；
forget_clean target_name_ce 没有明显上升；
forget_clean name_hit 反而高于 baseline。
```

因此当前不再把 answer-level Nec/Suf/Ret 当作身份遗忘的唯一选择标准。身份遗忘必须直接对准 target name token。

### 4.2 当前已经验证的成功点

| 成功点 | 结论 |
| --- | --- |
| target-name sensitive path 存在 | projector / vision_text path 的 NameSuf 明显高于 non-projector |
| Step6 name-aware 分类有效 | 能把 target-name sensitive projector path 从 P_shared 拉回 P_forget |
| strict name-token CE scope 生效 | `forget_ce_scope=name` 时只使用 `name_token_positions` |
| projector whole-vector 语义已修正 | coarse projector placeholder 可展开为 whole-vector，但不再作为主路线 |
| projector-dim candidate 可执行 | 可导出具体 projector dims，并进入 dim-level P_forget |
| name_preference_unlearning 已接入 | target name CE 与 answer_without_name positive CE 均可记录 token 覆盖 |
| cheap diagnostic 有效 | 可在 Full CLEAR 前否决无效 run |

### 4.3 当前推荐实验配置方向

当前最值得继续的小网格：

| run | candidate | objective | 参数方向 | 目标 |
| --- | --- | --- | --- | --- |
| I | projector-dim top8 / 31 dims | `name_ce_ascent` | `ce=0.30`, checkpoint sweep | 当前最佳行为配置，找最佳早停点 |
| J | projector-dim top8 / 31 dims | `redacted_name_preference` | name-span positive, `beta=0.005/0.01/0.02` | 测试低权重替代身份目标 |
| K | projector-dim top12 / 48 dims | `name_ce_ascent` | `ce=0.30` | 测试 31 dims 是否仍覆盖不足 |
| L | projector-dim top8 重筛 | `name_ce_ascent` | 排除 NameRet 偏高 path 对照 | 修 counterfactual_retain 边界问题 |

当前阶段不建议直接扩大 Full CLEAR，因为 Step11 projector-dim top8 cheap gate 仍未完全通过：

```text
当前最佳 forget 行为：
step11_top8_name_ce030
forget_clean target_name_ce delta = +0.002199
forget_clean name_hit = 0.5556
counterfactual_retain name_hit = 0.6667
hard_retain name_hit = 0.7222

当前最佳 CE delta：
step11_top8_name_ce050
forget_clean target_name_ce delta = +0.003591
forget_clean name_hit = 0.6667
```

这说明 top8/31-dim 扩容有实际改善，但 target_name_ce delta 仍未达到 +0.01。

---

## 5. 不再作为当前主线的方法

以下内容属于旧方法、错误中间路线或只保留为对照，不应再作为当前主线推进。

| 方法 | 当前处理 |
| --- | --- |
| 只用 answer-level Nec/Suf/Ret 选择身份遗忘路径 | 不再作为主线，必须加入 NameNec / NameSuf / NameRet |
| Step9 bound projector corrected 作为主实验路线 | 已被 Step10 name-token 对齐路线替代 |
| coarse `mm_projector` placeholder 直接代表 projector 编辑粒度 | 不再作为推荐主线，优先 projector-dim candidate |
| whole-vector projector 编辑 | 只作为诊断/对照，不作为当前首选 |
| `P_forget` 与 `P_shared` 粗粒度同时包含 projector placeholder | 视为错误风险，会导致 editable = forget - shared 后 projector 被抵消 |
| `activation_random_name_ce` 小网格直接扩大 Full CLEAR | 不推荐；小网格行为未通过 |
| 训练 loss 下降作为 unlearning 成功证据 | 不成立，必须看 target_name_ce delta 和 name_hit |
| cheap gate 未通过就跑 Full CLEAR | 不推荐，浪费实验资源且结论噪声大 |

---

## 6. metrics 指标模块

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

---

## 7. docs 文档目录

当前 `docs/` 目录：

```text
docs/
├── PROJECT_STRUCTURE.md
├── CHIP_EDITOR_EXPERIMENTS_SUMMARY.md
├── CHIP_EDITOR_IMPLEMENTATION_AUDIT.md
├── CHIP_EDITOR_ROUTE_STATUS.md
├── FORGET_PATH_CAUSAL_SELECTION_FIX_PLAN.md
├── MACHINE_UNLEARNING_SURVEY_OPTIMIZATION_ROUTE.md
├── MIP_Editor_Summary.md
├── NEXT_OPTIMIZATION_DISCUSSION_0527.md
├── NEXT_ROUTE_TARGET_NAME_PROJECTOR_DIM_OPTIMIZATION.md
├── SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md
├── SCIENCE_Q1_MAIN_REPORT.md
├── SCIENCE_Q1_NEXT_STEPS.md
├── STEP11_PROJECTOR_DIM_TARGET_NAME_OPTIMIZATION_RESULT_0531.md
├── STEP10_NAME_TOKEN_ALIGNED_ROUTE_0530.md
├── STEP10_NEXT_OPTIMIZATION_FROM_UNLEARNING_SURVEY_0530.md
├── STEP2_TO_STEP8_CODE_REVIEW_REPORT.md
├── STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
├── STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md
└── causal_mip 阶段性成功实验与经验总结.md
```

文档说明：

| 文件 | 作用 |
| --- | --- |
| `PROJECT_STRUCTURE.md` | 当前项目结构、当前有效方法和不再推荐方法 |
| `README.md` | 项目运行文档，位于仓库根目录 |
| `causal_mip 阶段性成功实验与经验总结.md` | 当前阶段最重要的实验结论与经验总结 |
| `STEP10_NAME_TOKEN_ALIGNED_ROUTE_0530.md` | Step10 target name token 对齐路线 |
| `STEP10_NEXT_OPTIMIZATION_FROM_UNLEARNING_SURVEY_0530.md` | 从 machine unlearning survey 借鉴后的下一步优化路线 |
| `CHIP_EDITOR_EXPERIMENTS_SUMMARY.md` | CHIP/MIP 相关关键实验汇总 |
| `CHIP_EDITOR_IMPLEMENTATION_AUDIT.md` | 当前路线实现审计 |
| `CHIP_EDITOR_ROUTE_STATUS.md` | 当前技术路线状态 |
| `FORGET_PATH_CAUSAL_SELECTION_FIX_PLAN.md` | forget path causal selection 修复计划 |
| `MACHINE_UNLEARNING_SURVEY_OPTIMIZATION_ROUTE.md` | 基于 unlearning survey 的优化路线 |
| `MIP_Editor_Summary.md` | MIP-Editor 概览总结 |
| `NEXT_OPTIMIZATION_DISCUSSION_0527.md` | 0527 下一步优化讨论 |
| `NEXT_ROUTE_TARGET_NAME_PROJECTOR_DIM_OPTIMIZATION.md` | target-name projector-dim 下一轮优化路线 |
| `SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md` | 科学问题 1 实现与实验 |
| `SCIENCE_Q1_MAIN_REPORT.md` | 科学问题 1 主报告 |
| `SCIENCE_Q1_NEXT_STEPS.md` | 科学问题 1 后续计划 |
| `STEP11_PROJECTOR_DIM_TARGET_NAME_OPTIMIZATION_RESULT_0531.md` | Step11 top8/31-dim 与 redacted objective 实验验证结果 |
| `STEP2_TO_STEP8_CODE_REVIEW_REPORT.md` | Step2-Step8 代码审查报告 |
| `STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md` | Step2-Step8 汇总实现说明 |
| `STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md` | Step9 闭环实验 runbook，当前主要作为历史对照 |

说明：

```text
旧的单次实验结果文档已不再作为当前 docs 主结构维护。
当前实验判断优先看中文阶段总结和 Step10/unlearning survey 优化路线。
```

---

## 8. pictures

```text
pictures/
├── mainfig.png
└── mainfig.pdf
```

用于论文/README/说明文档中的主图资源。

---

## 9. workspace 与实验产物

### 9.1 主实验 workspace

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

关键输出目录：

```text
../mip_workspace/outputs/paths/
../mip_workspace/outputs/scores/
../mip_workspace/outputs/checkpoints/
../mip_workspace/outputs/diagnostics/
../mip_workspace/outputs/full_clear_remote_eval/
../mip_workspace/outputs/logs/
```

当前 causal_mip 关键产物类型：

```text
../mip_workspace/outputs/scores/path_name_scores_<RUN_ID>.jsonl
../mip_workspace/outputs/paths/step6_name_aware_<RUN_ID>/
../mip_workspace/outputs/diagnostics/step8_probability_diag_<RUN_ID>_val.json
../mip_workspace/outputs/masked_rmisu_<RUN_ID>.json
../mip_workspace/outputs/step8_protocol_<RUN_ID>.json
```

### 9.2 仓库内 workspace

仓库内当前也存在：

```text
MIP-Editor/mip_workspace/
└── outputs/
    └── data_pairs/
```

该目录不是当前完整实验使用的主 workspace。后续复现实验时优先使用父目录：

```text
../mip_workspace/
```

---

## 10. 当前推荐阅读顺序

如果目标是理解当前项目运行方式：

1. `README.md`
2. `docs/PROJECT_STRUCTURE.md`
3. `docs/causal_mip 阶段性成功实验与经验总结.md`

如果目标是理解当前 causal_mip 科学路线：

1. `docs/SCIENCE_Q1_MAIN_REPORT.md`
2. `docs/SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md`
3. `docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md`
4. `docs/STEP10_NAME_TOKEN_ALIGNED_ROUTE_0530.md`
5. `docs/STEP10_NEXT_OPTIMIZATION_FROM_UNLEARNING_SURVEY_0530.md`
6. `docs/causal_mip 阶段性成功实验与经验总结.md`

如果目标是继续实现或调试当前最新方法：

1. `causal_mip/path_localization/projector_dim_export.py`
2. `causal_mip/causal_scores/name_token_metrics.py`
3. `causal_mip/causal_scores/metrics.py`
4. `causal_mip/causal_scores/build_scores.py`
5. `causal_mip/causal_scores/classify_paths.py`
6. `causal_mip/editing/masked_rmisu.py`
7. `causal_mip/evaluation/step8_probability_diagnostic.py`
8. `causal_mip/evaluation/step8_final_eval.py`
9. `causal_mip/evaluation/step8_protocol.py`

---

## 11. 当前推荐实验闭环

推荐最小闭环：

1. 用 `projector_dim_export.py` 导出 projector-dim candidates。
2. 用 Step3 binding 保留 pair/path 对齐关系。
3. 用 `build_scores.py --compute_name_token_scores` 计算 NameNec / NameSuf / NameRet。
4. 用 `classify_paths.py --name_aware_forget --projector_topk_metric name_editable_score` 生成 P_forget / P_shared。
5. 用 `masked_rmisu.py` 运行 projector-dim masked RMisU。
6. 优先比较 `name_ce_ascent` 与 `name_preference_unlearning`。
7. 用 `step8_probability_diagnostic.py` 检查 target_name_ce delta。
8. 用 `step8_final_eval.py` 检查 forget/retain name_hit。
9. cheap gate 通过后再跑 `step8_protocol.py` 和 `full_clear_remote_eval.py`。

每次实验必须记录：

```text
NameSuf positive count
P_forget projector/projector-dim count
projector_modules
projector_editable_neurons
whole-vector projector 是否为 false
forget_ce_scope
forget_ce_token_count
preference_positive_token_count
forget_clean target_name_ce delta
forget_clean name_hit
hard_retain name_hit
counterfactual_retain name_hit
```

---

## 12. 清理说明

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
