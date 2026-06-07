# MIP-Editor 项目结构

更新时间：2026-06-07

本文档记录当前工作区的有效项目结构和保留产物。详细实验总结见：

```text
docs/CAUSAL_MIP_PROJECT_SUMMARY_0607.md
```

当前主线是：

```text
causal_mip
+ target-name token alignment
+ name-aware path selection
+ projector-dim selective editing
+ masked RMisU
+ Step8 cheap gate
+ Full CLEAR remote evaluation
```

当前成功候选是 H5：

```text
H5 = H1 retain-safe base + p000006 top15 small add-back
objective = pii_name_token_noise
forget_ce_alpha = 0.40
pii_noise_alpha = 0.08
max_steps = 200
```

H5 已通过 Step8 cheap gate，在更大 split 上保持 target-name CE 增益稳定，并完成 Full CLEAR 远程评估。结果是：retain 基本持平，forget 小幅改善。

---

## 1. 工作区

代码库：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor
```

实验 workspace：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

当前 `mip_workspace/outputs` 已清理，仅保留 H5 成功链路和 baseline 对比产物。

---

## 2. 顶层结构

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
├── myqwen25vl.py
├── causal_mip/
├── metrics/
├── docs/
└── pictures/
```

主要入口：

| 路径 | 作用 |
| --- | --- |
| `README.md` | 当前 H5 运行指南和关键指标 |
| `main.py` | 训练、遗忘和 masked RMisU 入口 |
| `ours.py` | MIP-Editor / masked RMisU 接入逻辑 |
| `clear.py` | CLEAR 数据集定义 |
| `score_by_llm.py` | 本地/远程 judge 评分工具 |
| `causal_mip/` | 当前 causal selective editing 主线 |
| `docs/` | 当前文档，只保留项目结构和 0607 总结 |

---

## 3. causal_mip 结构

```text
causal_mip/
├── project_paths.py
├── data_pairs/
├── path_localization/
├── causal_scores/
├── interventions/
├── editing/
├── evaluation/
└── test_*.py
```

核心目录：

| 目录 | 作用 |
| --- | --- |
| `data_pairs/` | 构造 forget_clean、hard_retain、counterfactual_retain 等 pair |
| `path_localization/` | candidate path / projector-dim 定位与导出 |
| `causal_scores/` | Nec / Suf / Ret / name-token score / saliency specificity |
| `interventions/` | activation cache、patching、ablation、restoration |
| `editing/` | masked RMisU 编辑实现 |
| `evaluation/` | Step8 probability diagnostic、generation eval、Full CLEAR remote eval |

当前最重要的评估脚本：

| 文件 | 作用 |
| --- | --- |
| `causal_mip/evaluation/step8_probability_diagnostic.py` | target-name CE / logprob 诊断 |
| `causal_mip/evaluation/step8_final_eval.py` | pair-level generation eval |
| `causal_mip/evaluation/full_clear_remote_eval.py` | Full CLEAR 四任务远程评分 |

---

## 4. 当前保留的 outputs

当前保留的成功 checkpoint：

```text
../mip_workspace/outputs/checkpoints/
└── masked_rmisu_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603/
```

当前保留的 baseline adapter：

```text
../mip_workspace/outputs/model_caches/
└── Qwen2.5-VL-3B-Instruct_clear_batch2_epochs1_img_resize224.pth/
```

当前保留的 H5 paths：

```text
../mip_workspace/outputs/paths/
├── P_cand_projector_dim_hybrid_h5_top16_p000006_to_top15_0603.jsonl
├── P_cand_projector_dim_hybrid_h5_top16_p000006_to_top15_bound_0603.jsonl
├── hybrid_h5_top16_p000006_to_top15_summary_0603.json
└── step6_hybrid_h5_top16_p000006_to_top15_0603/
    ├── P_forget.jsonl
    └── P_shared.jsonl
```

当前保留的 Step8 结果：

```text
../mip_workspace/outputs/diagnostics/
├── step8_probability_diag_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_train.json
└── step8_probability_diag_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_val.json

../mip_workspace/outputs/
├── step8_eval_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_train.json
├── step8_eval_hybrid_h5_top16_p000006_to_top15_pii_noise_ce040_noise008_200steps_0603_val.json
├── step8_eval_peft_baseline_train.json
└── step8_eval_peft_baseline_val_full_0522.json
```

当前保留的 Full CLEAR 结果：

```text
../mip_workspace/outputs/full_clear_remote_eval/
├── hybrid_h5_top16_p000006_to_top15_full_remote_full_0607/
└── peft_baseline_full_remote_step9_bound_projector_0526_214028/
```

当前保留的数据 pair：

```text
../mip_workspace/outputs/
├── causal_pairs_train.jsonl
└── causal_pairs_val.jsonl
```

---

## 5. 当前文档

`docs/` 目前只保留两个文档：

```text
docs/
├── CAUSAL_MIP_PROJECT_SUMMARY_0607.md
└── PROJECT_STRUCTURE.md
```

| 文档 | 作用 |
| --- | --- |
| `CAUSAL_MIP_PROJECT_SUMMARY_0607.md` | 当前项目和 H5 成功实验的完整总结 |
| `PROJECT_STRUCTURE.md` | 当前代码结构、保留产物和运行上下文 |

旧 Step9-Step15、Science Q1、CHIP Editor、阶段性失败/成功分散文档已删除，因为它们记录的是历史优化过程，且多处结论已经被 H5 Full CLEAR 结果取代。

---

## 6. 当前运行顺序

推荐按照 README 中的命令运行：

```text
1. activate mip-editor environment
2. run H5 training if checkpoint needs regeneration
3. run Step8 val probability diagnostic
4. run Step8 val generation eval
5. run larger split probability / generation eval
6. run Full CLEAR remote eval
7. compare H5 against PEFT baseline
```

下一步实验建议：

```text
以 H5 为 base，优先做小剂量 counterfactual retain anchor，
目标是在保持 forget CE >= +0.010 的同时修复 counterfactual generation 的小幅损失。
```
