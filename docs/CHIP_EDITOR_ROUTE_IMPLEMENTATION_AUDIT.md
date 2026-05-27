# CHIP-Editor Route Implementation Audit

本文档对 `CHIP_EDITOR_TECHNICAL_ROADMAP.md` 中的四层技术路线做当前实现审计，重点判断：

1. 第一层 candidate path 到第四层 selective editing 是否已经完整闭环。
2. 当前实现是否足以回答路线图中的三个科学问题。
3. 下一步应优先补哪些工程和实验缺口。

审计基于当前工作区：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7
```

注意：部分旧文档和脚本默认路径仍指向：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7
```

因此后续复现实验时应先统一 `main.py` 默认路径、文档路径和实际 workspace 路径。

---

## 1. 总体结论

当前状态不是“路线完全完成”，而是：

```text
Step2-Step8 工程 MVP 已基本跑通，
但科学闭环尚未完全成立。
```

具体判断：

| 层级 | 路线图目标 | 当前状态 | 审计判断 |
| --- | --- | --- | --- |
| 第一层 | 找到 candidate influential paths | 已有 `P_cand.jsonl`，覆盖 text / vision / vision_text | 基本完成 |
| 第二层 | 用 causal pair + activation patching 证明因果路径 | Step3/4/5 已实现，但 `Suf` 全为 0，`mm_projector` 未进入 patching | 部分完成 |
| 第三层 | 分类 `P_forget / P_shared / P_retain / P_irrelevant` | Step6 已实现并有真实产物 | 工程完成，但依赖的 Step5 证据不够稳 |
| 第四层 | 只编辑 `P_forget`，保护 `P_shared` | Step7 masked RMisU 已实现并可保存 checkpoint | 工程完成，实验效果未证明成功 |
| Final Eval | 验证 forget 下降且 retain 保持 | Step8 已实现，但当前最佳结果未超过 PEFT baseline | 不能声明完成 |

当前最核心的问题不是代码不能跑，而是：

```text
候选路径 -> 因果验证 -> 路径分类 -> 选择性编辑
```

这条链路中的因果验证和最终遗忘效果还不够支撑论文式结论。

---

## 2. 四层路线实现审计

## 第一层：先找到候选路径

路线图目标：

```text
哪些路径可能承载目标知识？
```

对应 Step：

```text
Step 2: export top-k candidate paths
```

当前实现：

- `causal_mip/path_localization/path_schema.py`
- `causal_mip/path_localization/cached_path_export.py`
- `causal_mip/path_localization/mip_topk_wrapper.py`
- `causal_mip/path_localization/beam_search_paths.py`
- `causal_mip/path_localization/cross_modal_path_builder.py`
- `main.py --export_candidate_paths_only`

当前产物：

```text
mip_workspace/outputs/paths/P_cand.jsonl
total = 2312
text = 1552
vision = 752
vision_text = 8
```

审计结论：

```text
第一层基本完成。
```

但有两个边界：

1. `vision_text` 路径数量较少，当前只有 8 条。
2. cross-modal path 中包含 `mm_projector` 节点，但后续 Step4/5/7 没有完整支持该节点。

---

## 第二层：再证明哪些是因果路径

路线图目标：

```text
这些候选路径中，哪些路径对目标知识具有必要性和充分性？
```

对应 Step：

```text
Step 3: causal pairs
Step 4: activation cache + path patching
Step 5: Nec / Suf / Ret causal scores
```

### Step 3 Pair Construction

当前实现：

- `causal_mip/data_pairs/build_pairs.py`
- `causal_mip/data_pairs/text_corruption.py`
- `causal_mip/data_pairs/image_corruption.py`
- `causal_mip/data_pairs/hard_retain_builder.py`

当前产物：

```text
mip_workspace/outputs/causal_pairs_train.jsonl = 170
mip_workspace/outputs/causal_pairs_val.jsonl = 18
```

pair schema 覆盖：

- `forget_clean`
- `forget_corrupt`
- `hard_retain`
- `counterfactual_retain`

审计结论：

```text
Step 3 工程完成。
```

主要边界：

```text
P_cand 与 causal pair 仍不是样本级绑定。
```

也就是说，当前 Step5 是把一批 path 与一批 pair 做笛卡尔组合式评估，而不是严格使用“该样本导出的路径”去验证“该样本对应的 pair”。这会削弱因果路径结论。

### Step 4 Activation Patching

当前实现：

- `causal_mip/interventions/hooks.py`
- `causal_mip/interventions/activation_cache.py`
- `causal_mip/interventions/patching.py`
- `causal_mip/interventions/ablation.py`
- `causal_mip/interventions/restoration.py`

支持的 patchable module：

```text
LLM:    *.mlp.down_proj
Vision: *.mlp.down_proj
```

不支持：

```text
mm_projector
attention path
```

当前 cross-modal path 示例结构：

```text
visual down_proj
-> mm_projector
-> language image-token down_proj
-> language answer-token down_proj
```

但实际 Step5 中 `vision_text` score 的 `num_patchable_nodes=2`，说明 4 节点 cross-modal path 没有完整被 patch。

审计结论：

```text
Step 4 MVP 完成，但跨模态因果验证不完整。
```

### Step 5 Causal Scores

当前实现：

- `causal_mip/causal_scores/metrics.py`
- `causal_mip/causal_scores/necessity.py`
- `causal_mip/causal_scores/sufficiency.py`
- `causal_mip/causal_scores/retain_impact.py`
- `causal_mip/causal_scores/build_scores.py`

当前产物：

```text
path_scores_real_text_20x50_cuda.jsonl = 1000 records
path_scores_real_vision_text_20x8_cuda.jsonl = 160 records
scores/path_scores_real_vision_20x20_cuda_0522.jsonl = 400 records
```

这些记录的状态均为：

```text
status = ok
```

但是当前 score 分布有一个关键问题：

```text
text 20x50:
  positive_Nec = 400
  positive_Suf = 0
  positive_Ret = 394

vision_text 20x8:
  positive_Nec = 75
  positive_Suf = 0
  positive_Ret = 79

vision 20x20:
  positive_Nec = 179
  positive_Suf = 0
  positive_Ret = 197
```

审计结论：

```text
Step 5 工程完成，但 causal sufficiency 尚未成立。
```

这意味着当前只能较弱地说明：

```text
ablation 后目标答案 log-prob 可能下降
```

但还不能可靠说明：

```text
在 corrupt 输入中恢复该路径能恢复目标知识
```

因此，第二层目前不能严格宣称已经找到完整 causal paths。

---

## 第三层：再区分哪些该删，哪些该保

路线图目标：

```text
哪些路径是真正的 harmful-intent path，
哪些只是 benign-topic / shared-utility path？
```

对应 Step：

```text
Step 6: classify P_forget / P_shared / P_retain / P_irrelevant
```

当前实现：

- `causal_mip/causal_scores/classify_paths.py`
- `causal_mip/test_step6_classify_paths.py`

当前主要产物：

```text
paths/step6_v1/
  P_forget = 27
  P_shared = 12
  P_retain = 6
  P_irrelevant = 13

paths/step6_v1_quantile75/
  P_forget = 8
  P_shared = 3
  P_retain = 12
  P_irrelevant = 35

paths/step6_vision_stable_0522/
  P_forget = 5
  P_shared = 2
  P_retain = 5
  P_irrelevant = 16
```

分类逻辑：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

审计结论：

```text
Step 6 工程完成。
```

但因为 `Suf=0`，当前分类实际更接近：

```text
forget_effect ~= max(0, Nec)
```

因此 `P_forget / P_shared` 的科学含义仍依赖一个不完整的因果分数基础。

---

## 第四层：最后做选择性编辑

路线图目标：

```text
只编辑 P_forget，保护 P_shared。
```

对应 Step：

```text
Step 7: masked RMisU
Step 8: final evaluation
```

### Step 7 Masked RMisU

当前实现：

- `causal_mip/editing/masked_rmisu.py`
- `causal_mip/test_step7_masked_rmisu.py`
- `ours.py` 已接入 `--use_masked_rmisu`
- `main.py` 已提供 masked RMisU CLI 参数

当前机制：

1. 从 `P_forget.jsonl` / `P_shared.jsonl` 读取 path id。
2. 回到 `P_cand.jsonl` 找到对应 path nodes。
3. 构建：

```text
editable_neurons = forget_neurons - shared_neurons
preserve_neurons = forget_neurons union shared_neurons
```

4. 只包装对应 MLP 的：

```text
up_proj
gate_proj
```

5. 使用 `PartialLinear` 只开放 selected intermediate neurons 的参数更新。

当前产物示例：

```text
masked_rmisu_step7_full1epoch_ckpt_0521.json:
  num_modules = 34
  num_editable_neurons = 123
  num_loss_records = 1790
  checkpoint saved

masked_rmisu_step7_quantile75_ce01_full1epoch_0522.json:
  num_modules = 18
  num_editable_neurons = 45
  num_loss_records = 1790
  checkpoint saved
```

审计结论：

```text
Step 7 工程完成，但选择性编辑效果未被证明。
```

主要边界：

1. `mm_projector` 未纳入 editing。
2. attention path 未纳入 editing。
3. activation-level random-direction loss 没有稳定压制目标身份 token。
4. `forget_ce_alpha` 已加入，但当前最佳版本仅回到 baseline。

### Step 8 Final Evaluation

当前实现：

- `causal_mip/evaluation/step8_final_eval.py`
- `causal_mip/evaluation/full_clear_remote_eval.py`

pair-based Step8 当前结果：

```text
PEFT baseline:
  forget_clean name_hit_rate = 0.6111
  hard_retain name_hit_rate = 0.6944
  counterfactual_retain name_hit_rate = 0.7222

Step7 masked RMisU:
  forget_clean name_hit_rate = 0.7778
  hard_retain name_hit_rate = 0.7222
  counterfactual_retain name_hit_rate = 0.7222

Step7 quantile75 + forget CE alpha=0.1:
  forget_clean name_hit_rate = 0.6111
  hard_retain name_hit_rate = 0.7222
  counterfactual_retain name_hit_rate = 0.7222
```

审计结论：

```text
Step 8 工程完成，但不能证明 Step7 成功遗忘。
```

当前最佳版本只回到 PEFT baseline，没有低于 baseline。

---

## 3. 三个科学问题回答状态

## 科学问题 1

问题：

```text
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分“有害意图路径”和“良性主题路径”？
```

路线图依赖：

```text
Step 2 -> Step 3 -> Step 4 -> Step 5 -> Step 6
```

当前回答状态：

```text
只能部分回答，不能严格完整回答。
```

原因：

1. `P_cand` 已存在。
2. `Nec/Ret` 有正向信号。
3. 但 `Suf` 当前全为 0，充分性验证失败。
4. `P_cand` 与 causal pair 没有样本级绑定。
5. 因此 `P_forget/P_shared` 可以作为工程分类结果，但不能作为完全可靠的因果结论。

最低补齐条件：

- 建立 `path_id -> sample_id/pair_id` 显式绑定。
- 修复 Step5 sufficiency，使部分路径出现稳定正向 restoration。
- 重新生成 Step6 分类并做稳定性分析。

## 科学问题 2

问题：

```text
如何定位跨模态因果路径中的有害知识？
```

路线图依赖：

```text
Step 2 -> Step 4 -> Step 5
```

当前回答状态：

```text
只能部分回答。
```

原因：

1. Step2 已生成 `vision_text` candidate paths。
2. 但 `vision_text` 只有 8 条，覆盖有限。
3. cross-modal path 中的 `mm_projector` 没有进入 patching/scoring/editing。
4. 当前 `vision_text` score 实际只 patch 部分节点，不是完整跨模态路径。

最低补齐条件：

- 支持 `mm_projector` activation cache / ablation / restoration。
- 重新跑 `vision_text` Step5。
- 让 `vision_text` 的 `num_patchable_nodes` 覆盖完整路径。
- 重新生成 cross-modal `P_forget/P_shared`。

## 科学问题 3

问题：

```text
如何在删除目标知识时避免破坏 retain knowledge？
```

路线图依赖：

```text
Step 5 -> Step 6 -> Step 7 -> Step 8
```

当前回答状态：

```text
工程方法已有，实验结论未成立。
```

原因：

1. Step7 确实只开放 masked path neurons。
2. Step7 确实有 retain/shared preserve loss。
3. Step8 显示 retain 没有明显崩坏。
4. 但 forget 没有低于 PEFT baseline。

最低补齐条件：

- Step7 目标从 random-direction activation loss 转向更直接的输出层目标。
- 保留 masked path 参数约束。
- 加入 name-token suppression / refusal target / DPO 或 NPO 风格 forget objective。
- 用 Step8 pair eval 过筛，再用 Full CLEAR remote judge 作主协议。

---

## 4. 当前已通过的代码验证

轻量单元测试状态：

```text
python causal_mip/test_step2.py              passed
python causal_mip/test_step3_data_pairs.py  passed
python causal_mip/test_step4_interventions.py passed
python causal_mip/test_step5_causal_scores.py passed
python causal_mip/test_step6_classify_paths.py passed
python causal_mip/test_step7_masked_rmisu.py passed
python -m py_compile causal_mip/evaluation/step8_final_eval.py causal_mip/evaluation/full_clear_remote_eval.py passed
```

这说明当前不是基础代码断裂问题，而是科学闭环质量问题。

---

## 5. 必须优先修正的路线缺口

## P0: 统一 workspace 路径

当前代码默认路径、旧文档和实际工作区之间存在不一致。

应统一为当前工作区：

```text
/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7
```

否则 Step7/Step8 很容易读到旧产物或写到旧目录。

## P1: 建立 path 与 sample/pair 的绑定

当前缺口：

```text
P_cand 只描述路径，不知道来自哪个 forget sample。
causal_pairs 只描述 pair，不知道该用哪些 candidate paths。
```

建议新增产物：

```text
outputs/paths/P_cand_bound.jsonl
```

每条记录至少包含：

```json
{
  "path_id": "...",
  "sample_id": "...",
  "pair_id": "...",
  "source_row_idx": 0,
  "modality": "text|vision|vision_text",
  "nodes": []
}
```

Step5 应优先支持：

```text
只对 pair 对应的 paths 计算 Nec/Suf/Ret
```

而不是全量 path 与全量 pair 交叉组合。

## P2: 修复 Step5 Sufficiency

当前现象：

```text
positive_Suf = 0 across text / vision / vision_text
```

需要检查：

1. `forget_corrupt_target_clean_answer` 的 prompt 是否正确。
2. clean answer token 是否和 corrupt prompt 下的 target token 对齐。
3. restoration 的 token selector 是否应该从 answer tokens 改为 last prompt token 或 image tokens。
4. hidden activation restore 是否发生在正确的 module input。
5. `mm_projector` 缺失是否导致 cross-modal restoration 失效。

建议给 Step5 输出增加诊断字段：

```text
clean_score
corrupt_score
restored_score
target_answer_text
answer_token_ids
answer_token_positions
resolved_nodes
restore_token_positions
```

验收标准：

```text
至少在一部分 path 上 positive_Suf > 0，
并且 restoration_score 明显高于 corrupt_score。
```

## P3: 支持 mm_projector

当前 `mm_projector` 只出现在 Step2 candidate path 里，后续 Step4/5/7 都没有完整消化。

建议先做 MVP：

```text
mm_projector whole-vector patching
```

而不是一开始做精细 neuron patching。

后续再升级为：

```text
mm_projector hidden-dim patching
```

验收标准：

```text
vision_text path 的 num_patchable_nodes 覆盖完整 path，
至少不再系统性跳过 mm_projector。
```

## P4: 重做 Step6 分类

在 P1-P3 修完后重跑：

```text
Step5 -> Step6
```

并产出新的主分类目录，例如：

```text
outputs/paths/step6_bound_suf_projector_v1/
```

该目录应作为 Step7 主输入，而不是继续沿用旧的 `step6_v1` 或 `step6_v1_quantile75`。

## P5: 改造 Step7 forget objective

当前 random-direction activation loss 对目标身份 token 压制不稳定。

建议保留：

```text
masked parameter update
P_shared preserve
retain preserve
```

但将 forget objective 升级为可直接解释的输出目标：

1. target name token probability suppression
2. refusal / unknown target
3. DPO / NPO style forget objective
4. retain KL / CE preserve

验收标准：

```text
Step8 forget_clean name_hit_rate < PEFT baseline 0.6111
hard_retain / counterfactual_retain 不明显低于 baseline
```

## P6: 固定 Step8 主协议

建议分两层评估：

1. pair-based Step8：快速筛选实验。
2. Full CLEAR remote judge：作为主协议。

不能只用 pair-based `name_hit_rate` 宣称最终成功。

---

## 6. 推荐下一阶段路线

建议把下一阶段命名为：

```text
Step9: Causal-Selective Editing Closure
```

执行顺序：

```text
Step9.1 统一路径和运行配置
Step9.2 建立 P_cand_bound.jsonl
Step9.3 修复 Step5 Suf 并增加诊断字段
Step9.4 支持 mm_projector patching/scoring
Step9.5 重跑 Step5/Step6 得到新 P_forget/P_shared
Step9.6 用新 Step6 跑 masked output-objective editing
Step9.7 Step8 pair eval + Full CLEAR remote eval
```

不建议下一步优先做：

```text
attention head patching
pyvene backend
jailbreak / OCR trigger
safe-anchor 完整语义空间
```

这些可以放在主闭环稳定之后。

---

## 7. 当前完成度判定

按“代码是否存在并能跑”判断：

```text
完成度约 75%-80%
```

按“路线图科学闭环是否成立”判断：

```text
完成度约 55%-60%
```

当前最准确的表述是：

```text
CHIP-Editor 的四段式管线已经有可运行 MVP；
但还不能声明已经完整回答三个科学问题。
必须先补 causal sufficiency、sample-bound path validation、
mm_projector cross-modal validation 和更强的 selective editing objective。
```

---

## 8. 继续完成记录：P0-P6 工程补齐状态

本节记录在本审计之后继续完成的工程改动。它更新的是“代码能力”和“可复现入口”，不等同于已经完成重型 CUDA 实验验收。

### P0 workspace 路径

已完成。

新增统一路径入口：

- `causal_mip/project_paths.py`

默认 workspace：

```text
<repo_parent>/mip_workspace
```

也可以通过环境变量覆盖：

```bash
export MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

已更新 `main.py`、`full_clear_remote_eval.py`、`test_remote_scoring.py` 等入口，避免继续硬编码旧目录。

### P1 path 与 pair 绑定

已完成。

新增：

- `causal_mip/data_pairs/bind_paths_to_pairs.py`
- `causal_mip/test_step3_path_pair_bindings.py`

当前已生成：

```text
mip_workspace/outputs/paths/P_cand_bound_train.jsonl = 1360
mip_workspace/outputs/paths/P_cand_bound_val.jsonl = 144
```

`Step5 build_scores.py` 已支持：

```bash
--path_pair_bindings
```

用于只计算 pair 对应的 candidate paths，而不是 pair/path 全组合。

### P2 Step5 Sufficiency 诊断

已完成工程诊断补齐。

`compute_sufficiency` 现在输出：

```text
clean_score
corrupt_score
restored_score
clean_target
corrupt_target
restored_nodes
```

`restoration.py` 也避免对不等长 token positions 做不安全 remap。

注意：这只说明 sufficiency failure 现在可以定位；是否已经出现稳定 positive Suf，需要按 Step9 runbook 重跑真实 Step5。

### P3 mm_projector 支持

已完成 Step4/Step5 支持，Step7 已提供可选参数编辑支持。

Step4/Step5：

- `mm_projector` alias 会解析到真实模型模块。
- Qwen2.5-VL 中优先解析到 `model.visual.merger`。
- 对 projector 做 whole-vector activation cache / zero ablation / restoration。
- `vision_text` toy path 单测已覆盖 4 个节点全部 patchable。

Step7：

- 默认 `--masked_rmisu_projector_edit_mode qwen_merger_mlp`。
- Qwen2.5-VL 的 `mm_projector` 会映射到：

```text
model.visual.merger.mlp.0
```

- 对 projector 使用输出侧 activation objective，确保 projector 参数能收到梯度。
- 可通过 `--masked_rmisu_projector_edit_mode skip` 显式跳过 projector 编辑。

边界：

```text
当前不是 attention path 支持，也不是 projector 所有线性层全量编辑。
这是针对 Qwen2.5-VL visual.merger 的 MVP projector parameter editing。
```

### P4 重跑 Step6 分类

已完成可执行 runbook，真实重跑仍待执行。

新增：

- `docs/STEP9_CAUSAL_SELECTIVE_EDITING_CLOSURE_RUNBOOK.md`

主分类输出目录固定为：

```text
mip_workspace/outputs/paths/step6_bound_suf_projector_v1/
```

必须先用 bound+projector Step5 生成：

```text
mip_workspace/outputs/scores/path_scores_bound_projector_v1.jsonl
```

再由 Step6 分类生成新的 `P_forget/P_shared`。旧的 `step6_v1` 和 `step6_v1_quantile75` 不应继续作为最终主输入。

### P5 Step7 forget objective

已完成工程入口。

新增 CLI：

```bash
--masked_rmisu_forget_objective activation_random|ce_ascent|activation_random_ce
--masked_rmisu_forget_ce_alpha <float>
--masked_rmisu_projector_edit_mode qwen_merger_mlp|skip
```

语义：

```text
activation_random     原 RMisU random-direction activation objective
ce_ascent             只最大化 forget CE
activation_random_ce  random-direction activation + forget CE ascent
```

推荐下一轮主实验：

```text
activation_random_ce + forget_ce_alpha=0.1 + qwen_merger_mlp
```

验收仍以 Step8 为准：

```text
forget_clean name_hit_rate < PEFT baseline 0.6111
retain 指标不明显低于 baseline
```

### P6 Step8 主协议

已完成工程固定。

新增：

- `causal_mip/evaluation/step8_protocol.py`
- `causal_mip/test_step8_protocol.py`

现在 Step8 不再只是“看一个 pair-based name_hit_rate”，而是固定为三段：

```text
1. pair-based Step8 quick screen
2. Full CLEAR remote judge main protocol
3. step8_protocol.py 生成最终 pass / do_not_claim_success 判定
```

主协议门控规则：

```text
pair screen:
  forget_clean name_hit_rate 必须低于 PEFT baseline
  hard_retain / counterfactual_retain name_hit_rate 下降不超过 0.05

Full CLEAR main:
  forget classification/generation remote acc 必须低于 baseline
  retain classification/generation remote acc 下降不超过 0.05
  clf_forget / clf_retain / gen_forget / gen_retain 必须全部完成 remote scoring
```

只有 `step8_protocol.py` 输出：

```json
{"status": "pass", "can_claim_success": true}
```

才允许声明最终 Step8 主协议通过。

同时已修正 `full_clear_remote_eval.py`：

```bash
--peft_checkpoint
--output
```

现在 Step9 runbook 中的 Full CLEAR 命令可以直接加载 PEFT/checkpoint，并额外写出一个固定 summary JSON。

### 当前新增单元测试覆盖

新增或扩展覆盖：

```text
test_step3_path_pair_bindings.py
test_step4_interventions.py: projector cache/ablate/restore
test_step5_causal_scores.py: vision_text projector path num_patchable_nodes=4
test_step7_masked_rmisu.py:
  projector skip mode
  projector parameter wrapping
  projector output objective updates only selected row
  ce_ascent forget objective smoke
test_step8_protocol.py:
  pair screen + Full CLEAR 均通过时允许声明成功
  pair forget 未下降时拒绝声明成功
  缺少 Full CLEAR summary 时拒绝声明成功
  Full CLEAR task 未完成 remote scoring 时拒绝声明成功
```

### 当前最新状态判定

按工程能力：

```text
P0-P6 已基本补齐，Step9 闭环可执行入口已具备。
```

按科学结论：

```text
仍不能直接宣称三大科学问题已经完整回答。
```

还缺少新的真实实验产物：

```text
path_scores_bound_projector_v1.jsonl
step6_bound_suf_projector_v1/
masked_rmisu_step9_bound_projector_ce01 checkpoint
step8_eval_step9_bound_projector_ce01_val.json
full_clear_step9_bound_projector_ce01.json
step8_protocol_step9_bound_projector_ce01.json
```
