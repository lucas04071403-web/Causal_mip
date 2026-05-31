# Science Q1 Next Steps

更新时间：2026-05-29

本文记录 Science Q1 的下一步优化路线。当前结论见 `SCIENCE_Q1_MAIN_REPORT.md`，实现与实验记录见 `SCIENCE_Q1_IMPLEMENTATION_AND_EXPERIMENTS.md`。

## 1. 当前瓶颈

### 1.1 P_forget recall 低

20-pair path-level strict result：

```text
P_forget = 4
covered pairs = 2/20
modality = vision-only
```

0529 retain-aware node/projector-dim result：

```text
merged candidates = 19
stable candidates = 1
covered pairs = 1/20
```

这说明当前 protocol 已能找到 positive case，但还不是稳定定位方法。

### 1.2 projector 是 shared trunk

代表性现象：

```text
vision_text / projector path:
  Suf 很强
  Ret 也很强
  当前分类为 P_shared
```

解释：

```text
整条 projector path 同时承载 identity signal 和 benign grounding/topic signal。
直接编辑 whole path 很可能破坏 retain。
```

因此核心任务不是加强 Step7，而是继续拆分 projector path 内部的 node/dim/subspace。

### 1.3 path-level specificity 太粗

当前 path-level scoring：

```text
score(path) =
  mean over nodes of forget_saliency(node)
  - gamma * mean over nodes of retain_anchor_saliency(node)
```

问题：

```text
长 path 内部混有 harmful identity nodes 和 benign/shared topic nodes。
path-level 平均会稀释 identity-specific signal。
```

需要推进到：

```text
pair-level -> node-level -> dim-level -> subspace-level
```

### 1.4 Step7 行为证据不足

0529 Step7 smoke 只证明：

```text
formal stable 单候选能进入 Step7；
projector-dim path 能解析到 visual.merger.mlp.0；
5-step 最小编辑能保存 checkpoint。
```

它没有证明：

```text
forget target 被进一步压低；
retain 全局不退化；
Science Q1 已闭环。
```

Step8 pair diagnostic 的关键问题是 baseline 对 `pair_000089` 的 target name-hit 已经是 0，所以 name-hit 无法判断新增遗忘效果。

## 2. 优化原则

优先级：

```text
路径定位精度 > 行为诊断精度 > Step7 训练强度 > 大规模实验
```

当前不应直接做：

1. 大规模 Step7。
2. 总体超参优化。
3. whole-projector editing。
4. 用 single-candidate smoke 声称成功。

当前应做：

1. 对 existing positives 做稳定性确认。
2. 对 projector/shared path 做 node/dim/subspace 拆分。
3. 对 Step7 smoke 增加更细行为指标。
4. 再小规模扩大到多个 stable candidates。

## 3. 具体执行计划

### Step A: 补 exact target probability / scoped CE diagnostic

目的：

```text
解决 name_hit 在 baseline 已经为 0 时不可判定的问题。
```

对 baseline 与 smoke checkpoint 计算：

```text
forget_clean:
  target full-name token logprob
  name-scoped CE
  target-vs-generated-name margin

same_topic:
  scene/caption quality
  不把目标姓名 name_hit 当作主要 retain 指标

same_reasoning / counterfactual_retain:
  retain target name token logprob
  retain name-scoped CE
```

判定：

```text
有效 forget-side 信号:
  forget target name probability / logprob 下降
  或 name-scoped CE 上升

retain 不退化:
  same_reasoning / counterfactual target probability 不下降
  或 CE 不上升
```

优先级：最高。原因是当前 Step7 行为层最缺的是可判定 forget-side 信号。

### Step B: 对 0529 stable candidate 做稍长 smoke

条件：先完成 Step A 的 probability diagnostic。

配置建议：

```text
candidate = projdimRA20d_p000003
max_steps = 20 或 50
objective = name_ce_ascent
target_ce_scope = name
仍标注为 small diagnostic / single-candidate smoke
```

必须固定同一 diagnostic 集：

```text
pair_000089 forget_clean
same_topic
same_reasoning
counterfactual_retain
```

观察：

```text
forget target probability 是否下降；
counterfactual retain 是否保持 Ismail Jengo；
same_reasoning 是否不进一步恶化。
```

如果 forget target probability 不下降，则不应扩大 Step7，应回到 candidate/objective 设计。

### Step C: 扩大 retain-aware stability 到小批量

目标：

```text
从 1 条 stable candidate 扩到 3-5 条 stable candidates，
但仍然保持 small diagnostic，不进入大规模 Step7。
```

候选来源：

```text
P_cand_retainaware_node_projdim_20pairs_0529.jsonl
```

筛选：

```text
repeat pass rate >= 0.8
Suf.mean > 0
Ret.max <= retain_threshold
min_anchor_margin > 0
prefer pair-level stable candidates
```

如果当前 19 条只有 1 条稳定，应扩大 candidate generation，而不是降低阈值。

### Step D: pair-level classification 常规化

当前问题：

```text
record-level strict forget-like = 8
aggregated P_forget = 4
```

说明 path-level 聚合会压弱 pair-specific signal。

建议输出两个层级：

```text
P_forget_pair:
  aggregate by (pair_id, path_id)

P_forget_global:
  aggregate by path_id across pairs
```

用途：

```text
pair-level 用于定位目标身份相关路径；
global-level 用于发现可泛化编辑对象。
```

成功信号：

```text
pair-level P_forget coverage > path-level coverage
且 repeat validation 后仍保持 low Ret。
```

### Step E: node-level specificity 压缩长 path

对 text / vision path：

```text
score(node) =
  forget_saliency(node)
  - gamma * retain_anchor_saliency(node)
```

从每条 path 中选择：

```text
top-k positive-specific nodes
```

生成 compact candidates：

```text
P_cand_node_specific_*.jsonl
P_cand_node_specific_bound_*.jsonl
```

验证：

```text
compact path 的 Suf 仍 > 0
Ret 下降
margin 更稳定
```

进一步可以做 leave-one-layer-out：

```text
delta_suf_when_removed
delta_ret_when_removed
node_specificity_margin
```

目标：减少编辑范围，降低 retain 风险。

### Step F: projector dim/subspace specificity

对 projector / `visual.merger` whole-vector 节点：

```text
score(dim) =
  forget_saliency(dim)
  - gamma * retain_anchor_saliency(dim)
```

选择：

```text
top-k dims with positive worst-anchor margin
```

验证：

```text
dim-level patch:
  Suf > 0
  Ret <= retain_threshold
  repeat pass rate >= 0.8
```

0529 的 `projdimRA20d_p000003` 已经证明这条路线可行，但样本量只有 1。下一步应提高 stable dim-level candidate 数量，而不是直接编辑更多 shared dims。

### Step G: 小规模多候选 Step7

进入条件：

```text
stable candidates >= 3
每条 candidate:
  repeat pass rate >= 0.8
  Suf.mean > 0
  Ret.max <= retain_threshold
  min_anchor_margin > 0
Step A probability diagnostic 可判定
```

Step7 标注：

```text
small stable multi-candidate diagnostic
```

仍不能叫大规模 Step7。

观察：

```text
forget target probability mean 下降
hard/counterfactual retain probability 不下降
name_hit 不作为唯一指标
```

## 4. 成功标准

20-pair 定位层成功标准：

```text
stable P_forget coverage >= 6/20
至少出现一些 text 或 vision_text/projector dim-level P_forget
P_forget mean Suf > 0
P_forget mean Ret <= retain_threshold
P_shared projector path 可被拆出 low-Ret subpath/dims
```

Step7 diagnostic 成功标准：

```text
forget-side:
  target name probability 下降
  或 name-scoped CE 上升

retain-side:
  same_reasoning / counterfactual retain target probability 不下降
  或 CE 不上升

scope:
  编辑 modules/dims 与 stable candidates 对齐
  不出现 skipped_modules 或意外整层扩散
```

只有同时满足定位层和行为层标准，才可以推进到更大规模 Step7/Step8。

## 5. 失败分支

如果 exact target probability 没有下降：

```text
不要扩大 Step7；
检查 objective、learning rate、steps、candidate dim 是否真对应 target name token。
```

如果 forget 下降但 retain 退化：

```text
引入非空 P_shared；
提高 retain/shared penalty；
收紧 dim selection；
加入 retain CE / KL guard。
```

如果 stable candidates 仍只有 1 条：

```text
扩大 node/dim candidate generation；
保留 strict stability gate；
不要通过放松阈值制造候选。
```

如果 projector dims 仍 mostly high-Ret：

```text
考虑 subspace-level 方法，而不是单 dim top-k；
例如按 Fisher/saliency covariance 聚类，找 identity-specific low-rank direction。
```

## 6. 下一次建议执行

推荐最小下一步：

```text
1. 写一个 Step8 probability diagnostic：
   baseline vs step7_single_stable_smoke_retainaware_0529_152017
   pair_000089 only
   输出 target name CE/logprob。

2. 如果 forget target CE 上升且 retain CE 不坏：
   对同一 candidate 跑 20-step 或 50-step single-candidate smoke。

3. 如果没有 forget-side 信号：
   回到 projector dim candidate generation，扩大候选并重新做 repeat stability。
```

这个顺序的理由：

```text
当前最大的不可判定点是 forget-side 行为指标；
先补指标比盲目加训练步数更有信息量。
```

