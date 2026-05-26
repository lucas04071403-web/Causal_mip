# CHIP-Editor Step2-Step8 代码实现审查报告

本文档对当前 `MIP-Editor/causal_mip` 中 Step2 到 Step8 的实现进行代码审查，重点检查：

- 每一步是否与 `CHIP_EDITOR_TECHNICAL_ROADMAP.md` 对应。
- 每一步输入输出是否能贯通。
- 数据 schema、hook/patch 位置、causal score、path 分类、masked RMisU 和 final evaluation 是否逻辑一致。
- 哪些实现是合理 MVP，哪些地方与路线图存在偏差。

参考文档：

```text
docs/STEP2_TO_STEP8_IMPLEMENTATION_SUMMARY.md
PROJECT_STRUCTURE.md
CHIP_EDITOR_TECHNICAL_ROADMAP.md
```

核心代码目录：

```text
causal_mip/path_localization/
causal_mip/data_pairs/
causal_mip/interventions/
causal_mip/causal_scores/
causal_mip/editing/
causal_mip/evaluation/
```

## 1. 总体结论

当前 Step2-Step8 的工程链路已经基本跑通：

```text
Step2 candidate paths
-> Step3 causal pairs
-> Step4 activation cache / patching
-> Step5 Nec / Suf / Ret causal scores
-> Step6 path classification
-> Step7 masked RMisU editing
-> Step8 final evaluation
```

从测试和真实产物看，代码能够正常执行，主要 schema 能串起来。

但严格来说，当前实现并不是每一步都和路线图完全一一对应。最大偏差集中在：

1. cross-modal path 目前只是 endpoint-based MVP，不是真正完整跨模态桥接路径。
2. Step5 sufficiency 的 clean-to-corrupt restoration 存在 token position 对齐风险。
3. Step7 masked RMisU 没有继承 Step4/5 的 token-level path 语义，实际退化为 module-neuron 级别训练。
4. Step7 只训练 `up_proj/gate_proj`，没有训练 `down_proj` 对应列，编辑强度偏保守。
5. Step6 path-level mean 聚合会丢失 sample-level identity path 结构。

因此，当前状态更准确地说是：

```text
Step2-Step8 工程闭环成立；
因果 path editing 的 MVP 成立；
但 cross-modal causal path 和 token-level selective editing 还没有完整实现。
```

## 2. 已执行的轻量测试

已运行以下测试：

```bash
python causal_mip/test_step2.py
python causal_mip/test_step3_data_pairs.py
python causal_mip/test_step4_interventions.py
python causal_mip/test_step5_causal_scores.py
python causal_mip/test_step6_classify_paths.py
python causal_mip/test_step7_masked_rmisu.py
```

结果：

```text
Step2 tests passed.
Step3 data-pair tests passed.
Step4 intervention tests passed.
Step5 causal-score tests passed.
Step6 path-classification tests passed.
Step7 masked RMisU tests passed.
```

注意：

这些测试主要是 toy model / smoke test，能够证明接口、基础维度、基本逻辑可跑通，但不能完全覆盖真实 Qwen2.5-VL 中的所有 token 对齐和跨模态路径语义。

## 3. 真实产物抽查

当前真实产物能对上文档中的主要数量关系。

### 3.1 Step2 `P_cand.jsonl`

路径：

```text
mip_workspace/outputs/paths/P_cand.jsonl
```

抽查结果：

```text
total paths = 2312
text paths = 1552
vision paths = 752
vision_text paths = 8
```

按节点数统计：

```text
text path: 36 nodes
vision path: 32 nodes
vision_text path: 4 nodes
```

其中 `vision_text` path 包含 `mm_projector` 节点，但后续 Step4/5/7 会跳过该节点。

### 3.2 Step3 causal pairs

路径：

```text
mip_workspace/outputs/causal_pairs_train.jsonl
mip_workspace/outputs/causal_pairs_val.jsonl
```

抽查结果：

```text
train pairs = 170
val pairs = 18
hard_retain per pair = 2
counterfactual_retain missing = 0
corruption_type = explicit_perturbed_pair
```

这与文档中 `CLEAR + forget_ratio=5` 的规模对应。

### 3.3 Step5 scores

抽查文件：

```text
path_scores_real_text_20x50_cuda.jsonl
path_scores_real_vision_text_20x8_cuda.jsonl
```

结果：

```text
text records = 1000
status = ok
num_patchable_nodes = 36

vision_text records = 160
status = ok
num_patchable_nodes = 2
```

注意：

`vision_text` 原始 path 有 4 个节点，但实际只有 2 个节点被 patch。说明 cross-modal path 中至少 `mm_projector` 被跳过，并且实际验证的是部分 endpoint。

### 3.4 Step6 classification

`step6_v1_quantile75`：

```text
num_input_score_records = 1160
num_classified_paths = 58
P_forget = 8
P_shared = 3
P_retain = 12
P_irrelevant = 35
```

其中：

```text
P_forget 全部为 text path
P_shared 全部为 text path
```

这说明当前最终编辑集合偏 text，vision / cross-modal 没有进入最终 `P_forget`。

### 3.5 Step7 masked RMisU

`masked_rmisu_step7_quantile75_ce01_full1epoch_0522.json`：

```text
mask modules = 18
editable neurons = 45
skipped modules = 18
loss records = 1790
```

这个结果和 Step6 quantile75 的保守 mask 一致。

## 4. Step-by-Step 审查

## 4.1 Step2: Candidate Path Localization

### 目标

将原始 MIP-Editor 的 single greedy path 扩展为 top-k candidate paths：

```text
text IGI paths
vision Fisher / IFI paths
cross-modal bridge paths
```

### 关键代码

```text
causal_mip/path_localization/path_schema.py
causal_mip/path_localization/mip_topk_wrapper.py
causal_mip/path_localization/beam_search_paths.py
causal_mip/path_localization/cross_modal_path_builder.py
causal_mip/path_localization/cached_path_export.py
```

### 实现是否合理

整体合理。`CandidatePath` / `PathNode` schema 清晰，`P_cand.jsonl` 可被 Step4/5/7 消费。

每条 text path 对应 36 层，每层一个 neuron。每条 vision path 对应 32 层，每层一个 neuron。这和文档中“每层只保留一个 neuron”的逻辑一致。

### 主要问题

#### 问题 1：cross-modal path 只是 MVP 组合，不是真实搜索路径

`cross_modal_path_builder.py` 当前通过高分 vision path 和 text path 组合：

```text
vision node
-> mm_projector placeholder
-> text early node
-> text late node
```

这不是基于完整计算图搜索得到的跨模态因果路径，而是 endpoint 组合。

#### 问题 2：`mm_projector` 不被后续 patch/edit

`mm_projector` 节点会进入 `P_cand.jsonl`，但 Step4/5/7 都不支持该模块，因此实际会跳过。

结果是：

```text
vision_text path schema = 4 nodes
actual patchable nodes = 2
```

### 判断

```text
Step2 作为候选路径导出是合理的；
但 cross-modal path 的科学表述应降级为 endpoint-based MVP。
```

## 4.2 Step3: Causal Pair Construction

### 目标

构造 causal validation 所需的：

```text
forget_clean
forget_corrupt
hard_retain
counterfactual_retain
```

### 关键代码

```text
causal_mip/data_pairs/build_pairs.py
causal_mip/data_pairs/text_corruption.py
causal_mip/data_pairs/image_corruption.py
causal_mip/data_pairs/hard_retain_builder.py
```

### 实现是否合理

整体合理。

当前实现能够：

- 从 forget set 构造 `forget_clean`。
- 优先使用 explicit perturbed pair 构造 `forget_corrupt`。
- fallback 到 semantic minimal image corruption。
- 再 fallback 到 text corruption。
- 构造 `same_topic` hard retain。
- 构造 `same_reasoning` hard retain。
- 构造 `counterfactual_retain`。
- 保留 `image_ref.dataset_path + row_idx`，供 Step4/5/8 恢复图像。

### 主要问题

#### 问题 1：`forget_corrupt` 的语义依赖数据集质量

当前 CLEAR 中全部使用 `explicit_perturbed_pair`。如果 perturbed pair 不是严格“同问题、不同身份/图像”的最小反事实，则 Step5 sufficiency 的解释会变弱。

#### 问题 2：`same_topic` answer 是规则生成

`same_topic` 的 answer 来自 caption redaction：

```text
Describe visible scene without identifying person.
```

这是合理 MVP，但它不一定等价于模型真实 retain knowledge，只能作为 hard retain 的近似。

### 判断

```text
Step3 与路线图基本一一对应；
当前实现可用，主要风险来自 pair 质量，而不是代码结构。
```

## 4.3 Step4: Activation Cache And Path Patching

### 目标

对 candidate path 执行：

```text
activation cache
zero ablation
noise patch
clean-to-corrupt restoration
```

### 关键代码

```text
causal_mip/interventions/hooks.py
causal_mip/interventions/activation_cache.py
causal_mip/interventions/patching.py
causal_mip/interventions/ablation.py
causal_mip/interventions/restoration.py
```

### 实现是否合理

大体合理，尤其是一个关键修正是正确的：

```text
PathNode.neuron 是 FFN intermediate neuron，
所以 patch 应作用在 mlp.down_proj 输入侧，
而不是 down_proj 输出侧。
```

代码中通过 `forward_pre_hook` 修改 `down_proj` input，这是正确方向。

### 主要问题

#### 问题 1：语言侧 `image_tokens` 语义较弱

Step2 中早期 text layer 使用：

```text
token_selector = image_tokens
```

Step4 中语言侧 `image_tokens` 解析为 input ids 中的 image token position：

```python
input_ids_1d == image_token_id
```

但在 Qwen2.5-VL 中，语言模型里真正承载视觉信息的 token 可能不是简单一个 image placeholder token，而是经过视觉编码和 merge 后的一组视觉 token/embedding。

因此：

```text
text path + image_tokens
```

在 Step4 中可能只 patch 到占位 token，而不是完整视觉 token span。

#### 问题 2：vision path 默认 patch 所有视觉 token

对于 vision module：

```text
image_tokens -> [-1] -> expand to all visual tokens
```

这个 MVP 合理，但它与 text-side `image_tokens` 的语义不对称。

#### 问题 3：unsupported node 默认跳过

`mm_projector`、attention path 等不支持时默认跳过。这样能保证流程跑通，但会让某些 path 的实际 patch 节点少于 schema 节点。

### 判断

```text
Step4 的 down_proj input patch 逻辑是对的；
但 token_selector 的语义还比较粗，尤其是语言侧 image_tokens。
```

## 4.4 Step5: Causal Score Computation

### 目标

基于 Step4 计算：

```text
Nec(P) = clean_score - ablated_score
Suf(P) = restored_score - corrupt_score
Ret(P) = retain_baseline_score - retain_ablated_score
```

统一 score：

```text
target answer log-prob
```

### 关键代码

```text
causal_mip/causal_scores/metrics.py
causal_mip/causal_scores/necessity.py
causal_mip/causal_scores/sufficiency.py
causal_mip/causal_scores/retain_impact.py
causal_mip/causal_scores/build_scores.py
```

### 实现是否合理

公式与文档一致。

`forget_corrupt_target_clean_answer` 的设计也是对的：corrupt 输入上仍然计算 clean answer 的 log-prob，这符合 sufficiency 的定义。

### 主要问题

#### 问题 1：sufficiency 的 clean-to-corrupt restoration 有 token 对齐风险

流程是：

```text
clean_batch cache activation
corrupt_batch forward
restore clean activation into corrupt forward
```

但 `restoration.py` 中对语言节点会用 corrupt batch 重新解析 token positions：

```python
resolve_token_positions(prepared_batch, node.token_selector)
```

同时 restore values 仍来自 clean batch。

如果 clean 和 corrupt 的 answer token 长度、prompt length、image token position 不一致，就可能出现：

- shape mismatch。
- restore 到语义不等价位置。
- silently 得到不可信 Suf。

当前 CLEAR explicit perturbed pair 中 question 多数相同，所以风险不容易暴露。

#### 问题 2：retain impact 只用第一个 retain batch 判断 path 是否 patchable

`retain_impact.py` 先用第一个 retain batch 做：

```python
resolved_nodes = resolve_candidate_path_targets(candidate_path, first_batch)
```

后续对每个 retain batch 再执行 ablation。

如果不同 retain batch 的 token selector 可解析性不同，状态判断可能不完全准确。

#### 问题 3：Step5 仍然只用 log-prob，不覆盖自由生成行为

这与 MVP 文档一致，但要注意 Step8/Full CLEAR 评估最终看生成和 remote judge。log-prob causal score 与最终行为之间可能有 gap。

### 判断

```text
Step5 公式实现对应路线图；
但 clean/corrupt token alignment 是需要优先加固的点。
```

## 4.5 Step6: Path Classification

### 目标

将 Step5 的 `(pair_id, path_id)` 记录聚合成 path-level 分类：

```text
P_forget
P_shared
P_retain
P_irrelevant
```

### 关键代码

```text
causal_mip/causal_scores/classify_paths.py
```

### 实现是否合理

整体合理。

默认逻辑：

```text
Nec = mean over pairs
Suf = mean over pairs
Ret = mean over pairs
forget_effect = max(0, Nec) + alpha * max(0, Suf)
retain_impact = max(0, Ret)
```

再按阈值分类。

### 主要问题

#### 问题 1：mean 聚合会稀释 sample-specific identity path

身份遗忘中的路径可能只对部分人物/样本有效。直接 mean 可能把强作用 path 平均掉。

当前真实结果也显示：

```text
step6_v1_quantile75:
P_forget = 8
P_forget 全部为 text
```

这说明分类结果可能过窄。

#### 问题 2：阈值策略对结果影响很大

`step6_v1` 使用 0 阈值得到：

```text
P_forget = 27
P_shared = 12
```

`step6_v1_quantile75` 得到：

```text
P_forget = 8
P_shared = 3
```

这说明分类结果对 threshold/quantile 极敏感。

#### 问题 3：没有显式保留 coverage / stability 信息

当前 Step6 输出没有包含：

```text
forget_effect_coverage
retain_impact_coverage
positive_effect_pair_count
stable_under_corruption
```

这些信息对于判断 path 是否真的可编辑很重要。

### 判断

```text
Step6 与文档一一对应；
但分类质量依赖聚合策略，当前过于粗糙。
```

## 4.6 Step7: Masked RMisU Editing

### 目标

使用 Step6 的：

```text
P_forget
P_shared
```

构造 neuron mask，并只编辑 `P_forget - P_shared`。

### 关键代码

```text
causal_mip/editing/masked_rmisu.py
partial_linear.py
ours.py
main.py
```

### 实现是否合理

mask 构建逻辑基本合理：

```text
读取 P_forget/P_shared path_id
-> 回查 P_cand
-> 聚合 module/neuron
-> editable_neurons = forget_neurons - shared_neurons
```

参数冻结和局部训练也能工作。

### 主要问题

#### 问题 1：Step7 没有继承 token_selector

Step4/5 的 causal path 是 token-specific：

```text
image_tokens
answer_tokens
all_tokens
```

但 Step7 的 tracer 只按 module/neuron 抽取：

```python
selected = tensor[..., neurons]
```

也就是说，它训练的是：

```text
某 neuron 在所有 token 上的 activation
```

而不是：

```text
某 path 在指定 token selector 上的 activation
```

这会导致 Step5 的 causal path 语义在 Step7 被弱化。

#### 问题 2：只训练 `up_proj/gate_proj`，没有训练 `down_proj`

Step4 patch 的对象是：

```text
down_proj input side FFN intermediate neuron
```

Step7 当前包装的是：

```text
up_proj
gate_proj
```

这能改变 intermediate neuron 的产生过程，是合理的。但没有训练 `down_proj` 对应输入列，因此没有完整编辑该 neuron 对 hidden state 的投影影响。

#### 问题 3：`PartialLinear` 命名为 `trainable_cols`，实际是 weight row

在 PyTorch `nn.Linear` 中：

```text
weight shape = [out_features, in_features]
```

代码中：

```python
original_linear.weight[self.trainable_cols, :]
```

实际选择的是 output rows，不是 columns。功能上对应 intermediate neuron 是对的，但命名会误导。

#### 问题 4：forget loss 与最终输出指标不直接对齐

当前 forget loss 是：

```text
editable neuron activation -> random direction
```

这不保证目标答案 log-prob 或生成中的 name_hit 下降。

后续加入的 `forget_ce_alpha` 以负号加入：

```text
- forget_ce_alpha * forget_ce_loss
```

理论上是 CE ascent，但如果权重很小，主要训练信号仍然是 activation random target。

### 判断

```text
Step7 的 mask 接入是合理的；
但从 causal path editing 角度看，它目前退化为 neuron-level masked RMU，
还不是完整 token-aware path editing。
```

## 4.7 Step8: Final Evaluation

### 目标

评估编辑后模型是否：

```text
forget_clean 上遗忘目标知识
hard_retain 上保持能力
counterfactual_retain 上保持泛化
```

### 关键代码

```text
causal_mip/evaluation/step8_final_eval.py
causal_mip/evaluation/full_clear_remote_eval.py
```

### 实现是否合理

Step8 pair-based evaluation 基本合理：

- 读取 Step3 pair 文件。
- 构造 `forget_clean`、`hard_retain`、`counterfactual_retain` examples。
- 生成模型输出。
- 计算 `name_hit_rate`、BLEU、ROUGE。

Full CLEAR remote eval 也提供了更接近原 baseline 的远程评分协议。

### 主要问题

#### 问题 1：Step8 没有评估 `forget_corrupt`

Step3/5 中 `forget_corrupt` 是 sufficiency 的核心，但 Step8 最终 eval 不看 corrupt 行为。

这不是代码 bug，但说明 Step8 不是完整 counterfactual behavior validation。

#### 问题 2：`name_hit_rate` 是启发式指标

`name_hit` 当前基于字符串包含和姓名 parts 匹配。它适合快速判断身份泄露，但可能出现：

- paraphrase 漏判。
- partial name 误判。
- retain 样本中姓名出现不一定代表错误。

#### 问题 3：Step8 与 Full CLEAR 指标不是同一个协议

Step8 是 pair val 上的本地生成指标；Full CLEAR remote eval 是全量任务 + 远程 judge。两者结论应分开解释。

### 判断

```text
Step8 与路线图基本对应；
但它是 final behavior smoke/eval，不是完整验证 Step3 causal pair 的所有反事实设计。
```

## 5. 每一步对应性总表

| Step | 文档目标 | 当前实现 | 对应性判断 |
|---|---|---|---|
| Step2 | 导出 top-k candidate paths | text / vision / vision_text paths 已导出 | 基本对应，cross-modal 是 MVP |
| Step3 | 构造 clean/corrupt/hard retain/counterfactual retain | schema 和数据均可用 | 对应 |
| Step4 | activation cache + patching | down_proj input pre-hook 已实现 | 基本对应，token selector 粗 |
| Step5 | 计算 Nec/Suf/Ret | 公式实现一致 | 对应，但 restoration 对齐有风险 |
| Step6 | 四分类路径集合 | mean/median + threshold 已实现 | 对应，但聚合策略粗 |
| Step7 | masked RMisU selective editing | module-neuron mask 可训练 | 部分对应，未 token-aware |
| Step8 | final eval | pair eval + full CLEAR remote eval | 基本对应 |

## 6. 关键风险清单

### 高优先级

#### 风险 1：Step7 丢失 token-level path 语义

表现：

```text
Step5 验证的是 token-specific path；
Step7 训练的是 all-token neuron activation。
```

影响：

- causal score 与编辑目标不完全一致。
- 可能削弱 forget 效果。
- 可能增加 retain 副作用。

建议：

```text
让 Step7 的 DownProjInputTracer 支持 token_selector / token_positions。
```

#### 风险 2：Step5 sufficiency clean/corrupt token 对齐不稳

表现：

```text
clean activation values
restore 到 corrupt token positions
```

当 clean/corrupt token 数量不同，可能出错。

建议：

```text
保存 clean token positions 与 corrupt token positions；
如果长度不同，显式跳过、截断、padding 或做 token-level alignment；
不要 silent restore。
```

#### 风险 3：cross-modal path 实际没有完整桥接验证

表现：

```text
vision_text path schema 有 4 nodes；
Step5 实际 patchable nodes 只有 2。
```

建议：

```text
要么实现 mm_projector patch/edit；
要么在文档和论文表述中明确当前是 endpoint-based cross-modal MVP。
```

### 中优先级

#### 风险 4：Step6 mean 聚合过粗

建议增加：

```text
forget_effect_coverage
retain_impact_coverage
positive_pair_count
sample-level stability
```

#### 风险 5：Step7 编辑强度偏保守

建议：

```text
加入 down_proj selected column 编辑；
或加入 selective dampening warm start。
```

#### 风险 6：Step8 不验证 forget_corrupt 行为

建议：

```text
增加 forget_corrupt eval_set；
统计 corrupt_target 与 clean_target 的相对概率或生成行为。
```

## 7. 推荐修正路线

### 7.1 第一优先级：修 Step7 token-aware masked RMisU

目标：

```text
让 Step7 的训练目标与 Step5 的 causal path 语义一致。
```

需要做：

1. 从 `P_cand` 中保留每个 editable neuron 对应的 token_selector。
2. 在训练 batch 中解析 token positions。
3. `DownProjInputTracer` 只抽取对应 token positions 上的 neuron activation。
4. preserve loss 同样按 token selector 保护。

预期收益：

```text
减少无关 token 上的扰动；
增强目标 answer/image token 上的编辑信号；
提高 Step5 -> Step7 的逻辑一致性。
```

### 7.2 第二优先级：修 Step5 restoration alignment

目标：

```text
保证 Suff(P) 真实表示 clean path restoration 到 corrupt 输入。
```

需要做：

1. 在 `CachedNodeActivation` 中保存 clean token positions 和 values shape。
2. restore 时解析 corrupt token positions。
3. 如果 clean/corrupt positions 数量不同，记录 `status=token_alignment_mismatch`。
4. Step6 默认跳过 mismatch score。

预期收益：

```text
提高 Suf 分数可信度；
避免 text corruption 或 prompt 变化时污染 causal score。
```

### 7.3 第三优先级：增强 cross-modal path 支持

两个选择：

#### 选择 A：实现真正 `mm_projector` patch

需要：

```text
定位 Qwen2.5-VL 中视觉 embedding 到 language embedding 的 projector module；
支持 projector input/output activation cache；
支持 projector neuron patch。
```

#### 选择 B：明确降级为 endpoint-based MVP

文档中改写：

```text
cross-modal path 当前表示 vision endpoint + language endpoint 组合，
不是完整 bridge path。
```

如果当前目标是尽快优化实验，建议先选 B，避免科学表述过强。

### 7.4 第四优先级：改 Step6 聚合

新增 path-level 统计：

```text
mean_forget_effect
median_forget_effect
max_forget_effect
forget_effect_coverage
mean_retain_impact
retain_impact_coverage
num_positive_nec_pairs
num_positive_suf_pairs
```

新的 `P_forget` 选择：

```text
forget_effect 高
forget coverage 高
retain impact 低
retain coverage 低
```

### 7.5 第五优先级：增强 Step7 编辑强度

可选方案：

1. 训练 `down_proj` selected input columns。
2. selective dampening warm start。
3. output-level forget loss。
4. retain KD preserve。

这部分与 `MACHINE_UNLEARNING_SURVEY_OPTIMIZATION_ROUTE.md` 中的优化路线一致。

## 8. 文档需要修正的地方

### 8.1 Cross-modal 表述

当前文档中：

```text
cross-modal path 由 vision path 和 text path 组合生成，当前通过 mm_projector 作为 bridge 节点。
```

建议改为：

```text
cross-modal path 当前是 endpoint-based MVP：
由 vision endpoint 和 text endpoint 组合，并插入 mm_projector placeholder。
当前 Step4/5/7 不 patch/edit mm_projector，因此该 path 不能视为完整 bridge path。
```

### 8.2 Step7 表述

当前文档中：

```text
P_forget: 执行 random-direction misdirection
P_shared: preserve / freeze
```

建议补充：

```text
当前 Step7 是 module-neuron-level masked RMisU，
尚未按 Step4/5 的 token_selector 执行 token-aware path editing。
```

### 8.3 Step5 表述

建议补充：

```text
Sufficiency restoration 当前假设 clean/corrupt 对应 token positions 可对齐。
对于 text corruption 或 prompt length 变化，需要显式 alignment 检查。
```

## 9. 最终判断

当前实现可以作为 CHIP-Editor 的 causal MIP MVP：

```text
候选路径导出可用；
反事实 pair 可用；
down_proj input patching 可用；
Nec/Suf/Ret score 可用；
四分类接口可用；
masked RMisU 可训练；
final evaluation 可跑。
```

但如果目标是支撑更强的论文结论，需要修正以下三个核心缺口：

```text
1. cross-modal bridge path 没有完整实现；
2. Step5 clean/corrupt restoration 没有显式 token alignment；
3. Step7 没有 token-aware path-level editing。
```

建议下一阶段优先做：

```text
Step7 token-aware masked RMisU
-> Step5 restoration alignment
-> Step6 coverage-based classification
-> cross-modal 表述修正或 mm_projector patch 支持
```

这样才能让整条链路从“工程上跑通”进一步变成“逻辑上一一对应且实验上更可能有效”。
