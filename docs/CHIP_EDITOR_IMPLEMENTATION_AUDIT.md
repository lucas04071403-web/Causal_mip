# CHIP-Editor Implementation Audit

更新时间：2026-05-29

本文合并旧实现审计和技术路线图中的工程状态，记录当前代码能力、已补齐项目、测试覆盖和剩余工程缺口。路线总览见 `CHIP_EDITOR_ROUTE_STATUS.md`，实验结果见 `CHIP_EDITOR_EXPERIMENTS_SUMMARY.md`。

## 1. 总体工程状态

当前工程状态：

```text
P0-P6 已基本补齐，Step9 闭环可执行入口已具备。
```

但要区分：

```text
代码能力已具备 != 真实 CUDA 实验已成功
```

当前还不能宣称三个科学问题完整回答，因为仍缺少稳定的因果路径选择和最终 Step8 通过结果。

## 2. P0-P6 工程补齐

### P0 workspace 路径

已完成。

新增统一路径入口：

```text
causal_mip/project_paths.py
```

默认 workspace：

```text
<repo_parent>/mip_workspace
```

可通过环境变量覆盖：

```bash
export MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
```

已更新 `main.py`、`full_clear_remote_eval.py`、`test_remote_scoring.py` 等入口，避免继续硬编码旧目录。

### P1 path 与 pair 绑定

已完成。

新增：

```text
causal_mip/data_pairs/bind_paths_to_pairs.py
causal_mip/test_step3_path_pair_bindings.py
```

当前已生成：

```text
mip_workspace/outputs/paths/P_cand_bound_train.jsonl = 1360
mip_workspace/outputs/paths/P_cand_bound_val.jsonl = 144
```

`Step5 build_scores.py` 已支持：

```bash
--path_pair_bindings
```

用途：只计算 pair 对应的 candidate paths，而不是 pair/path 全组合。

### P2 Step5 sufficiency 诊断

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

`restoration.py` 避免对不等长 token positions 做不安全 remap。

后续 Science Q1 又进一步加入：

```text
all_nodes_patchable
contains_projector
projector_patchable
resolved_nodes
skipped_nodes
sufficiency_positive
target_answer_text
answer/name token positions
```

### P3 mm_projector 支持

已完成 Step4/Step5 支持，Step7 已提供可选参数编辑支持。

Step4/Step5：

```text
mm_projector alias 会解析到真实模型模块
Qwen2.5-VL 中优先解析到 model.visual.merger
支持 projector whole-vector activation cache / zero ablation / restoration
vision_text toy path 单测覆盖 4 个节点全部 patchable
```

Step7：

```text
--masked_rmisu_projector_edit_mode qwen_merger_mlp
```

Qwen2.5-VL 的 `mm_projector` 映射到：

```text
model.visual.merger.mlp.0
```

也可显式跳过：

```bash
--masked_rmisu_projector_edit_mode skip
```

边界：

```text
这不是 attention path 支持，也不是 projector 所有线性层全量编辑。
这是针对 Qwen2.5-VL visual.merger 的 MVP projector parameter editing。
```

### P4 重跑 Step6 分类入口

已完成可执行 runbook 和分类入口。

主分类输出目录曾固定为：

```text
mip_workspace/outputs/paths/step6_bound_suf_projector_v1/
```

但 Science Q1 后续路线已升级为 retain-aware saliency / node / dim 分类，最新应优先参考：

```text
docs/SCIENCE_Q1_MAIN_REPORT.md
docs/SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md
```

### P5 Step7 forget objective

已完成工程入口。

基础 CLI：

```bash
--masked_rmisu_forget_objective activation_random|ce_ascent|activation_random_ce
--masked_rmisu_forget_ce_alpha <float>
--masked_rmisu_projector_edit_mode qwen_merger_mlp|skip
```

后续 Step7 probe 又新增 scoped objectives：

```text
answer_ce_ascent
name_ce_ascent
activation_random_answer_ce
activation_random_name_ce
```

配置：

```text
target_ce_scope = all | answer | name
```

CLEAR collate 已增加：

```text
answer_token_positions
name_token_positions
```

显存修正：

```text
preserve forward 默认不传 labels；
只在 scoped CE 中用 logits 手动计算目标 token loss。
```

### P6 Step8 主协议

已完成工程固定。

新增：

```text
causal_mip/evaluation/step8_protocol.py
causal_mip/test_step8_protocol.py
```

Step8 现在固定为三段：

```text
1. pair-based Step8 quick screen
2. Full CLEAR remote judge main protocol
3. step8_protocol.py 生成最终 pass / do_not_claim_success 判定
```

完整 remote scoring 要求：

```text
clf_forget
clf_retain
gen_forget
gen_retain
```

四个文件都必须完整 scored，不允许 partial 结果支持成功声明。

## 3. Step7 projector / scoped objective 审计

已验证能力：

1. projector / cross-modal path 能进入 Step7 编辑集合。
2. `visual.merger.mlp.0` 可作为 Qwen projector 编辑模块。
3. `P_projector_probe` 中的 projector neuron 能加入 trainable set。
4. answer/name token scoped CE 能定位真实 CLEAR 样本中的姓名 token。
5. Step7 preserve forward 显存问题已修复。

真实 CLEAR/Qwen processor sanity check：

| 样本 | name | answer tokens | name tokens |
| --- | --- | ---: | ---: |
| 0 | Ursula Schmidt | 30 | 4 |
| 1 | Giorgi Meladze | 32 | 6 |
| 2 | Gabriela Carrasco | 24 | 5 |
| 3 | Jaime Vasquez | 31 | 4 |

当前无效或不足：

1. 单纯增强 projector probe 强度无效。
2. `activation_random + name CE` 中 activation random loss 量级远大于 name CE，容易主导错误更新。
3. 单纯 `name_ce_ascent` 尚未低于 PEFT baseline。
4. 继续提高 name CE alpha 会带来 counterfactual retain 下降风险。

结论：

```text
Step7 的路径编辑机制可用；
但 forget objective 需要从 activation random / CE ascent 转向 NPO 或 replacement-style objective。
```

## 4. 当前测试覆盖

新增或扩展覆盖：

```text
test_step3_path_pair_bindings.py
test_step4_interventions.py
test_step5_causal_scores.py
test_step6_classify_paths.py
test_step7_masked_rmisu.py
test_step8_protocol.py
test_retain_aware_localization.py
```

覆盖点：

```text
projector cache/ablate/restore
vision_text projector path num_patchable_nodes=4
projector skip mode
projector parameter wrapping
projector output objective updates selected row
ce_ascent forget objective smoke
pair screen + Full CLEAR 均通过时允许声明成功
pair forget 未下降时拒绝声明成功
缺少 Full CLEAR summary 时拒绝声明成功
Full CLEAR task 未完成 remote scoring 时拒绝声明成功
retain-aware node/projector-dim localization
```

近期通过的测试：

```bash
PYTHONPATH=. python causal_mip/test_retain_aware_localization.py
PYTHONPATH=. python causal_mip/test_step5_causal_scores.py
PYTHONPATH=. python causal_mip/test_step6_classify_paths.py
```

历史 Step7 测试：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile causal_mip/editing/masked_rmisu.py clear.py ours.py main.py causal_mip/test_step7_masked_rmisu.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step7_masked_rmisu.py
```

结果：

```text
Step 7 masked RMisU tests passed.
```

## 5. 剩余工程缺口

仍需补强：

1. Step8 probability / scoped CE diagnostic，避免 name_hit 在 baseline 已为 0 时不可判定。
2. NPO / replacement / refusal objective。
3. KL-constrained preference editing。
4. pair-level / node-level / dim-level path classification 常规化。
5. projector subspace-level selection。
6. 更严格的 run metadata，避免 smoke 与正式 Step7 混用。

不建议近期优先做：

```text
attention head patching
pyvene backend
jailbreak / OCR trigger
safe-anchor 完整语义空间
```

原因：主闭环里的 causal path quality 和 forget objective 尚未稳定。

## 6. 工程状态结论

当前工程结论：

```text
管线可运行，关键入口已补齐；
但已有实验显示，仅打通 plumbing 不足以得到可靠遗忘。
```

下一步工程应服务两个目标：

1. 让路径定位更精细：pair-level、node-level、dim-level、subspace-level。
2. 让行为诊断更细：target probability / scoped CE / preference objective，而不是只看 name_hit。

