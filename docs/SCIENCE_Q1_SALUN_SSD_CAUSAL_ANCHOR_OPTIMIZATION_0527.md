# Science Q1: SalUn/SSD + Causal Tracing + Retain Anchor 优化记录

更新时间：2026-05-27

本文记录对科学问题 1 的第二批实现优化：

```text
科学问题 1：
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分 harmful identity path 和 benign/shared topic path？
```

本轮采用的原则：

```text
SalUn/SSD 负责找 forget-specific candidate path
CaMU/causal tracing 负责证明 path 是否真有因果作用
SMFA/retain anchor 负责区分 harmful identity path 和 benign/shared topic path
```

这里的借鉴不是直接替换当前 CHIP-Editor，而是把 machine unlearning 中成熟的 saliency / Fisher / retain-anchor 思想接入现有 Step5/Step6 path selection。

参考：

- Survey repo: `https://github.com/tamlhp/awesome-machine-unlearning`
- SalUn: `https://arxiv.org/abs/2310.12508`
- SSD: `https://arxiv.org/abs/2308.07707`
- CaMU: `https://arxiv.org/abs/2401.17504`
- SMFA: `https://arxiv.org/abs/2511.20196`
- 当前项目结构：`docs/PROJECT_STRUCTURE.md`
- 第一批 Science Q1 guard：`docs/SCIENCE_Q1_STEP5_STEP6_FIX_0527.md`
- 历史失败实验：`docs/EXPERIMENT_RESULT_STEP9_BOUND_PROJECTOR_0526_214028.md`

---

## 1. 本轮解决的问题

第一批修正已经确认：

```text
历史 Step9:
  Step5 positive Suf = 0
  旧 Step6 P_forget = 100
  新 guard 后 P_forget = 0
```

说明旧 `P_forget` 实际不满足 “Nec + Suf” 因果证据，不能回答科学问题 1。

但只加 `Suf > 0` 仍不够，因为它只能排除因果证据不足的路径，不能主动回答：

```text
哪些路径是 forget-specific？
哪些路径虽然影响 forget，但也承载 retain / benign topic？
```

因此本轮加入 SalUn/SSD 风格的 path saliency specificity。

---

## 2. 新增模块

新增文件：

```text
causal_mip/causal_scores/saliency_specificity.py
```

新增能力：

```text
compute_batch_path_saliency(...)
compute_path_saliency_specificity(...)
```

对每条 candidate path 计算：

```text
forget_saliency
retain_anchor_saliency
saliency_specificity_margin = forget_saliency - gamma * retain_anchor_saliency
saliency_specificity_ratio  = forget_saliency / (retain_anchor_saliency + eps)

forget_fisher_saliency
retain_anchor_fisher_saliency
fisher_specificity_margin = forget_fisher_saliency - gamma * retain_anchor_fisher_saliency
fisher_specificity_ratio  = forget_fisher_saliency / (retain_anchor_fisher_saliency + eps)
```

其中：

```text
forget_saliency:
  forget_clean 上 target answer/name token logprob 对 path activation 的 |activation * gradient|

forget_fisher_saliency:
  forget_clean 上 target answer/name token logprob 对 path activation 的 gradient^2

retain_anchor_saliency:
  same_topic / same_reasoning / counterfactual_retain retain anchors 上相同 path 的平均 saliency
```

实现细节：

- 优先从正常 forward graph 中读取 activation gradient。
- 如果 frozen/eval/PEFT 场景拿不到 activation gradient，则对目标 module input 创建临时 leaf tensor，计算局部 path saliency。
- projector 节点沿用 Step4/5 的 whole-vector patch 口径，因此可以对 `mm_projector` / `visual.merger` 类节点计算整向量 saliency。

---

## 3. Step5 新增输出字段

修改文件：

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/build_scores.py
causal_mip/causal_scores/__init__.py
```

`build_scores.py` 新增参数：

```bash
--compute_saliency_specificity
--saliency_gamma 1.0
--saliency_eps 1e-6
```

打开后，每条 Step5 score record 新增：

```text
saliency_status
forget_saliency
retain_anchor_saliency
saliency_specificity_margin
saliency_specificity_ratio
forget_fisher_saliency
retain_anchor_fisher_saliency
fisher_specificity_margin
fisher_specificity_ratio
saliency_gamma
saliency_eps
saliency_specificity
```

注意：

```text
该计算默认关闭。
```

原因是它需要对 forget 和 retain anchors 做额外 backward，完整实验会明显变慢。正式 Science Q1 跑法需要显式打开。

---

## 4. Step6 新增分类逻辑

修改文件：

```text
causal_mip/causal_scores/classify_paths.py
```

聚合路径时新增聚合字段：

```text
forget_saliency
retain_anchor_saliency
saliency_specificity_margin
saliency_specificity_ratio
forget_fisher_saliency
retain_anchor_fisher_saliency
fisher_specificity_margin
fisher_specificity_ratio
```

`classify_paths.py` 新增参数：

```bash
--require_saliency_specificity
--saliency_specificity_key saliency_specificity_margin
--min_saliency_specificity 0.0
--max_retain_anchor_saliency <float>
--demote_high_retain_anchor_to_irrelevant
```

默认科学规则更新为：

```text
P_forget 必须满足：
  1. forget_effect 高
  2. retain_impact 低
  3. Suf > min_forget_sufficiency
  4. all_score_records_fully_patchable = true
  5. 如果启用 --require_saliency_specificity，则 specificity > min_saliency_specificity
  6. 如果设置 max_retain_anchor_saliency，则 retain_anchor_saliency 不得过高
```

如果一个路径 causal forget effect 高，但 retain anchor saliency 也高：

```text
默认转入 P_shared
```

这对应科学解释：

```text
它不是纯 harmful identity path，而是 harmful identity 与 benign/shared topic 共用的路径。
```

如果调试时不希望转入 `P_shared`，可加：

```bash
--demote_high_retain_anchor_to_irrelevant
```

---

## 5. Science Q1 当前判定协议

当前可以回答科学问题 1 的口径应改为：

```text
forget-specific candidate:
  SalUn/SSD score 高
  即 forget_saliency 高，retain_anchor_saliency 低，
  saliency/fisher specificity margin 或 ratio 高

causal harmful identity path:
  在 forget-specific candidate 的基础上，
  Nec > threshold 且 Suf > threshold
  并且节点完整 patchable

benign/shared topic path:
  forget causal effect 高，
  但 Ret 高或 retain_anchor_saliency 高
```

建议分类公式：

```text
P_forget:
  forget_effect > tau_f
  retain_impact <= tau_r
  Suf > 0
  saliency_specificity_margin > tau_s
  retain_anchor_saliency <= tau_a
  all_score_records_fully_patchable = true

P_shared:
  forget_effect > tau_f
  and (
    retain_impact > tau_r
    or retain_anchor_saliency > tau_a
  )

P_retain:
  forget_effect <= tau_f
  retain_impact > tau_r

P_irrelevant:
  others
```

其中：

```text
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

---

## 6. 推荐下一次小规模跑法

不要直接跑完整 Step7/Step8。先小规模跑 Step5/Step6：

```bash
export MIP_WORKSPACE_ROOT=/home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/mip_workspace
cd /home/lucas/Desktop/CurrentReacher/MIP_fusion7_5.22/MIP_fusion7/MIP-Editor

/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/causal_scores/build_scores.py \
  --llm_directory /path/to/model/root/ \
  --model Qwen2.5-VL-3B-Instruct \
  --output_file_path "$MIP_WORKSPACE_ROOT/outputs/step5_saliency_dummy.log" \
  --pairs_path "$MIP_WORKSPACE_ROOT/outputs/pairs/causal_pairs_train.jsonl" \
  --candidate_paths_path "$MIP_WORKSPACE_ROOT/outputs/paths/P_cand.jsonl" \
  --path_pair_bindings "$MIP_WORKSPACE_ROOT/outputs/pairs/path_pair_bindings_train.json" \
  --output "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_scienceq1_saliency_probe_0527.jsonl" \
  --num_pairs 2 \
  --num_paths 20 \
  --compute_saliency_specificity \
  --saliency_gamma 1.0
```

然后分类：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/causal_scores/classify_paths.py \
  --scores_path "$MIP_WORKSPACE_ROOT/outputs/scores/path_scores_scienceq1_saliency_probe_0527.jsonl" \
  --output_dir "$MIP_WORKSPACE_ROOT/outputs/paths/step6_scienceq1_saliency_probe_0527" \
  --alpha 1.0 \
  --forget_quantile 0.75 \
  --retain_quantile 0.75 \
  --min_forget_effect 0.0 \
  --min_retain_impact 0.0 \
  --require_saliency_specificity \
  --saliency_specificity_key saliency_specificity_margin \
  --min_saliency_specificity 0.0
```

如果小规模结果里：

```text
P_forget > 0
positive Suf > 0
P_forget 中 contains_projector/projector_patchable 有正样本
P_shared 能吸收 high retain-anchor path
```

再扩大到全量 Step5/6。

---

## 7. 当前科学状态

本轮之后，科学问题 1 的回答不再是：

```text
只看 Nec/Ret 阈值分类。
```

而是：

```text
先用 SalUn/SSD 找 forget-specific candidate，
再用 CaMU/causal tracing 的 Nec/Suf/Ret 做因果验证，
最后用 SMFA/retain anchor 把 harmful identity path 和 benign/shared topic path 分开。
```

但注意：

```text
当前代码已经支持这个协议；
历史 Step9 结果仍不能被重新解释为成功。
```

原因是历史 Step9 的 Step5 没有计算 saliency specificity，且 `Suf=0`。下一步必须重新跑带 saliency specificity 的 Step5/Step6 小规模验证。

---

## 8. 测试

已运行：

```bash
/home/lucas/miniconda3/envs/mip-editor/bin/python -m py_compile \
  causal_mip/causal_scores/saliency_specificity.py \
  causal_mip/causal_scores/metrics.py \
  causal_mip/causal_scores/build_scores.py \
  causal_mip/causal_scores/classify_paths.py \
  causal_mip/test_step5_causal_scores.py \
  causal_mip/test_step6_classify_paths.py

/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step5_causal_scores.py
/home/lucas/miniconda3/envs/mip-editor/bin/python causal_mip/test_step6_classify_paths.py
```

结果：

```text
Step 5 causal-score tests passed.
Step 6 path-classification tests passed.
```
