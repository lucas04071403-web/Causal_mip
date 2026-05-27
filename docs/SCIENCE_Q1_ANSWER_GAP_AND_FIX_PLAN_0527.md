# Science Q1 Answer Gap and Fix Plan

更新时间：2026-05-27

基于：

```text
docs/SCIENCE_Q1_CONSOLIDATED_0527.md
```

本文回答三个问题：

```text
1. 当前结果是否回答了科学问题 1？
2. 目前还存在哪些问题？
3. 下一步怎么改，以及为什么这些改法会有效？
```

---

## 1. 是否回答了科学问题 1？

科学问题 1 是：

```text
在多模态大模型中，哪些路径是真正承载目标知识的因果路径，
如何区分 harmful identity path 和 benign/shared topic path？
```

当前结论：

```text
已经回答了“如何判定”的框架；
还没有稳定回答“哪些路径”。
```

更准确地说：

```text
Science Q1 当前完成度约为 60%。
```

已完成的部分：

```text
1. 已证明旧 Step9 的 P_forget 不可靠：
   历史 100 条 P_forget 全部缺少 positive Suf。

2. 已修正判定标准：
   P_forget 不能只靠 Nec，必须同时要求 Suf > 0。

3. 已加入 retain anchor：
   Suf > 0 但 Ret 高的 path 不能叫 harmful identity path，应进入 P_shared。

4. 已加入 saliency specificity：
   forget_saliency - retain_anchor_saliency 可以过滤 retain-biased path。

5. 已在 20-pair 上产生少量严格 P_forget：
   P_forget = 4，覆盖 pair_000185 和 pair_000067。
```

仍未完成的部分：

```text
1. P_forget 覆盖率太低，只覆盖 2/20 个 pair。
2. P_forget 全部是 vision-only，没有 text 或 vision_text/projector。
3. projector path 虽然 Suf 很强，但 Ret 也很强，仍被判为 P_shared。
4. 当前 path-level 方法无法拆开 shared path 内部的 harmful identity 子成分。
5. 已发现的 P_forget saliency margin 很小，稳定性尚未验证。
```

因此当前不能说：

```text
我们已经定位了多模态大模型中的目标知识因果路径。
```

只能说：

```text
我们已经建立了一个比旧 Step9 更严格的判定协议，
并在 20-pair 上找到少量符合该协议的候选路径。
```

---

## 2. 当前证据链的强点

## 2.1 旧 P_forget 被正确推翻

历史 Step9：

| metric | value |
| --- | ---: |
| records | 2720 |
| positive Suf | 0 |
| old P_forget | 100 |

加上 `Suf > 0` guard 后：

| category | count |
| --- | ---: |
| P_forget | 0 |
| demoted_from_forget | 100 |

说明：

```text
旧 P_forget 只是 Nec 驱动，不能证明路径足以恢复目标知识。
```

这个修正是必要的，因为：

```text
Nec 只能说明 ablate 后输出变化；
Suf 才能说明 restore 后目标知识被路径重新带回来。
```

## 2.2 Suf 修复后路径确实有因果信号

修复后的 Step5：

| metric | value |
| --- | ---: |
| records | 2720 |
| positive Suf | 1807 |
| vision_text positive Suf | 1360 / 1360 |

说明：

```text
原先 Suf=0 不是模型中没有充分因果路径，
而是 Step5 计分/patch 口径没有正确暴露它们。
```

这使后续 Science Q1 有了可验证基础。

## 2.3 retain anchor 已经能阻止误判

pair_000002 + pair_000064：

| setting | P_forget |
| --- | ---: |
| no saliency gate | 5 |
| with saliency gate | 0 |

这些候选：

```text
Suf > 0
Ret 低
但 retain_anchor_saliency > forget_saliency
```

说明：

```text
仅靠 Suf/Ret 仍会误判；
saliency specificity 能识别 retain-biased path。
```

这部分已经实质回答了 Q1 中的“如何区分 shared topic path”。

## 2.4 20-pair 中首次出现严格 P_forget

20-pair strict Step6：

| category | count |
| --- | ---: |
| P_forget | 4 |
| P_shared | 12 |
| P_retain | 2 |
| P_irrelevant | 16 |

严格 P_forget：

| path_id | pair | modality | Suf | Ret | saliency margin |
| --- | --- | --- | ---: | ---: | ---: |
| `saliency20_p000027` | `pair_000185` | vision | 0.0234375 | -0.0169271 | 0.0000020433 |
| `saliency20_p000028` | `pair_000185` | vision | 0.0234375 | -0.0182292 | 0.0000020036 |
| `saliency20_p000031` | `pair_000067` | vision | 0.0156250 | -0.0130208 | 0.0000009385 |
| `saliency20_p000033` | `pair_000067` | vision | 0.0312500 | -0.0039063 | 0.0000000438 |

这说明：

```text
当前 protocol 不是只会拒绝候选；
它确实能找到少量 harmful-identity-like causal path。
```

---

## 3. 当前主要问题

## 3.1 P_forget recall 太低

现象：

```text
20 个 pair 中，严格 P_forget 只覆盖 pair_000185 和 pair_000067。
```

问题：

```text
如果一个方法只能在 10% pair 上找到 P_forget，
它还不能作为稳定的目标知识路径定位方法。
```

根因判断：

```text
当前候选生成仍是 path-level ranking。
一条长 path 内部包含很多节点，真正有用的 identity 节点可能只占少数。
平均到整条 path 后，forget-specific signal 被稀释。
```

## 3.2 P_forget 全部是 vision-only

现象：

```text
P_forget = 4
modality = vision-only
projector / vision_text P_forget = 0
```

问题：

```text
Science Q1 面向的是多模态大模型。
如果最终只找到 vision branch 的路径，还不足以解释跨模态目标知识如何进入语言输出。
```

根因判断：

```text
当前 projector path 以 whole-vector / whole-path 方式计分。
这会把 identity-specific dims 和 general grounding/topic dims 混在一起。
```

## 3.3 Projector path 被判为 P_shared

代表性结果：

| path_id | modality | Suf | Ret | saliency margin | category |
| --- | --- | ---: | ---: | ---: | --- |
| `saliency20_p000000` | vision_text | 2.15625 | 0.892578 | 0.0000706877 | P_shared |
| `saliency20_p000004` | vision_text | 2.62305 | 0.840495 | 0.0000195429 | P_shared |
| `saliency20_p000007` | vision_text | 2.52344 | 0.727865 | 0.0000283855 | P_shared |

解释：

```text
这些 path 的 Suf 非常强，说明它们确实承载目标相关信息；
但 Ret 也非常强，说明它们同时承载 retain / shared topic 信息。
```

这不是失败信号，而是重要发现：

```text
projector path 很可能是 shared trunk。
真正 harmful identity signal 不是整条 projector path，
而是 projector path 内部的某些 dims / nodes / subspace。
```

## 3.4 Saliency margin 太小

严格 P_forget 的 margin：

```text
0.0000020433
0.0000020036
0.0000009385
0.0000000438
```

问题：

```text
这些 margin 都很小，尤其 saliency20_p000033 几乎贴近 0。
如果不做 repeat validation，可能把数值噪声当成 specificity。
```

## 3.5 path_id 聚合可能压弱 pair-specific signal

当前 Step6 按 `path_id` 聚合：

```text
同一 path 如果绑定多个 pair，会先聚合 Nec/Suf/Ret/saliency，再分类。
```

问题：

```text
目标知识往往是 pair-specific 的。
一个 path 对 pair A 是 P_forget，对 pair B 是 P_shared 或 P_irrelevant，
聚合后可能被平均掉。
```

20-pair 中已经出现这种迹象：

```text
record-level strict forget-like = 8
最终 aggregated P_forget = 4
```

这说明有一半 pair-record 级别信号没有变成最终 path 级 P_forget。

---

## 4. 解决办法

## 4.1 先做 repeat validation，确认 4 条 P_forget 是否稳定

建议：

```text
对当前 4 条 P_forget 重新运行 causal tracing：
  1. 重复 3-5 次；
  2. retain anchor 换不同 sampled anchors；
  3. corrupt source 换不同 negative pair；
  4. 使用 mean 和 median 两种聚合；
  5. 输出 Suf/Ret/saliency margin 的均值、方差和置信区间。
```

判定标准：

```text
稳定 P_forget:
  mean(Suf) > 0
  mean(Ret) <= retain_threshold
  mean(saliency_margin) > 0
  lower confidence bound of margin > 0 or pass rate >= 80%
```

为什么有效：

```text
当前 4 条 P_forget 的 margin 很小。
重复验证可以区分真实 forget-specific signal 和一次性数值波动。
如果一条 path 在不同 retain anchors / corrupt sources 下仍保持 Suf > 0 且 Ret 低，
它才更接近真正 causal harmful identity path。
```

优先级：

```text
最高。
```

原因：

```text
如果这 4 条都不稳定，就不能基于它们继续 Step7；
如果它们稳定，则可以作为后续 node-level / dim-level 方法的 positive control。
```

## 4.2 把 path-level specificity 改成 node-level specificity

当前：

```text
score(path) =
  mean over nodes of forget_saliency(node)
  - gamma * mean over nodes of retain_anchor_saliency(node)
```

建议改为：

```text
score(node) =
  forget_saliency(node)
  - gamma * retain_anchor_saliency(node)
```

然后在每条 path 内选：

```text
top-k positive-specific nodes
```

再构造新的 compact path：

```text
P_cand_node_specific =
  only selected nodes from original path
```

为什么有效：

```text
长 path 内部很可能混合了两类节点：
  harmful identity nodes
  benign/shared topic nodes

path-level 平均会把两者混在一起。
node-level scoring 可以把 identity-specific 节点从 shared path 中拆出来。
```

对当前问题的直接作用：

```text
1. 提高 P_forget recall。
2. 降低 Ret，因为 shared-topic nodes 会被剔除。
3. 让 text/vision 中的边界候选更容易超过 forget threshold。
```

建议新增模块：

```text
causal_mip/path_localization/node_specific_export.py
```

输入：

```text
Step5 score records with node_scores
原始 P_cand.jsonl
```

输出：

```text
P_cand_node_specific_*.jsonl
P_cand_node_specific_bound_*.jsonl
```

## 4.3 对 projector / vision_text 做 dim-level specificity

当前 projector path 的问题：

```text
Suf 很强，但 Ret 也很强。
```

建议：

```text
对 projector / visual.merger 的 whole-vector 节点，不再只算一个整向量 saliency。
改为计算每个 hidden dim 的 specificity：

score(dim) =
  forget_saliency(dim)
  - gamma * retain_anchor_saliency(dim)
```

然后选择：

```text
top-k dims with positive score
```

干预方式从 whole-vector patch 改成 dim-mask patch：

```text
只 patch selected dims，
不 patch shared dims。
```

为什么有效：

```text
projector 是跨模态共享通道。
整向量 patch 会同时恢复 identity 信息和 benign grounding 信息，
所以 Suf 和 Ret 都高。

dim-level patch 可以只恢复 forget-specific dims，
理论上应保持 Suf > 0，同时显著降低 Ret。
```

这正好对应当前最主要瓶颈：

```text
把 P_shared projector path 拆成：
  P_forget projector dims
  P_shared projector dims
```

建议新增能力：

```text
1. activation_cache 支持 dim mask patch。
2. saliency_specificity 支持 whole-vector node 的 per-dim saliency。
3. CandidatePath schema 增加 dim_indices 或 mask metadata。
```

## 4.4 Step6 增加 pair-level classification

当前：

```text
aggregate by path_id
```

建议增加：

```text
aggregate by (pair_id, path_id)
```

输出两个层级：

```text
P_forget_pair:
  pair-specific forget path

P_forget_global:
  across-pair stable forget path
```

为什么有效：

```text
目标身份知识本身是 pair-specific。
先按 path_id 全局聚合，会把某个 pair 的强 P_forget 信号和其他 pair 的弱信号平均掉。

pair-level classification 能提高 recall，
global-level classification 再负责筛出可泛化路径。
```

当前数据支持这个判断：

```text
record-level strict forget-like = 8
aggregated P_forget = 4
```

说明 pair-level 和 path-level 之间已经有信息损失。

## 4.5 saliency specificity 不应只用 margin > 0

当前规则：

```text
saliency_specificity_margin > 0
```

问题：

```text
margin 接近 0 时不稳定。
```

建议改成联合门控：

```text
saliency_specificity_margin > tau_margin
saliency_specificity_ratio > tau_ratio
retain_anchor_saliency < tau_anchor
```

推荐初始值：

```text
tau_margin:
  用 20-pair positive margin 的 25% 或 50% 分位数，而不是 0。

tau_ratio:
  > 1.05 或 > 1.10

tau_anchor:
  用 P_shared projector retain_anchor_saliency 分布的低分位数。
```

为什么有效：

```text
margin > 0 只能说明 forget saliency 略大于 retain saliency。
ratio 和 anchor cap 能排除“几乎相等”或“retain 也很高”的路径。

这样可以提高 P_forget precision，
避免把 shared path 或数值噪声放进编辑集合。
```

注意：

```text
这会降低 recall。
因此应在 node/dim-level specificity 之后使用，而不是现在直接全局收紧。
```

## 4.6 对 text / vision 长 path 做 leave-one-layer-out 和 top-k compression

当前 vision P_forget 每条 path 约 32 个 visual blocks 节点。

建议：

```text
1. 对每条 P_forget path 做 leave-one-layer-out：
   每次移除一个 node/layer，重算 Suf/Ret。

2. 计算 node contribution：
   delta_suf_when_removed
   delta_ret_when_removed
   node_specificity_margin

3. 保留少量关键 nodes：
   top-k by delta_suf - lambda * delta_ret
```

为什么有效：

```text
长 path 会带来两个问题：
  干预范围大，容易影响 retain；
  真实 causal contribution 被低贡献节点稀释。

压缩 path 后，若 Suf 保持正而 Ret 降低，
说明我们更接近真正的 harmful identity causal subpath。
```

这也能服务后续 Step7：

```text
编辑更少节点，retain 破坏风险更低。
```

---

## 5. 推荐执行顺序

不要马上进入 Step7 大规模编辑。

推荐顺序：

```text
Step A: validate current positives
  对 4 条 P_forget 做 repeat validation。

Step B: pair-level classification
  输出 P_forget_pair 和 P_forget_global，
  确认 record-level 8 条 forget-like 中哪些是真实 pair-specific signal。

Step C: node-level specificity
  对 text / vision path 生成 node-specific compact candidates。

Step D: projector dim-level specificity
  对 high-Suf high-Ret projector path 做 dim/subspace 拆分。

Step E: rerun Step5/Step6
  在 20-pair 上重新验证：
    P_forget coverage
    projector P_forget 是否出现
    P_shared 是否下降或被拆分

Step F: only then Step7
  只有当 P_forget 稳定且 Ret 低，才进入编辑实验。
```

成功标准：

```text
20-pair 上：
  stable P_forget coverage >= 6/20
  at least some text or vision_text/projector subpaths enter P_forget
  P_forget mean Suf > 0
  P_forget mean Ret <= retain_threshold
  P_shared projector path 可被拆出低-Ret subpath
```

---

## 6. 为什么不能现在做总体优化？

当前如果直接总体优化 Step7，有两个风险：

```text
1. 编辑 projector P_shared：
   会破坏 retain，因为 projector path Ret 很高。

2. 编辑不稳定 P_forget：
   可能只是 margin 噪声或 pair-specific 偶然信号。
```

当前真正的问题不是训练强度不足，而是：

```text
编辑对象还不够精确。
```

所以优先级应是：

```text
路径定位精度 > 编辑损失设计 > 总体超参优化
```

---

## 7. Final Assessment

对 Science Q1 的最终判断：

```text
当前工作已经回答了 Q1 的判定原则：
  harmful identity path 必须同时满足 causal effect、low retain impact、
  saliency specificity 和 full patchability。

当前工作还没有充分回答 Q1 的定位结果：
  P_forget 覆盖率低，且没有跨模态 projector P_forget。
```

下一步最关键的改法：

```text
把 path-level specificity 推进到 node-level / dim-level specificity。
```

这会有效的原因：

```text
当前最大瓶颈不是没有 causal signal，
而是 causal signal 和 shared signal 混在同一条 path 里。

只有拆到 node/dim/subspace 级别，
才可能在保留 Suf 的同时降低 Ret，
从而把 P_shared projector path 拆出真正的 P_forget 子路径。
```

