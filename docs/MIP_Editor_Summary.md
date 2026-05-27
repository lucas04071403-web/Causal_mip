
# Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models (arXiv:2511.06793) - Detailed Summary

## 1. 研究背景

### 1.1 多模态大语言模型（MLLM）与遗忘需求
多模态大语言模型（MLLMs）如 Qwen-VL、LLaVA 等能处理文本与图像输入，并执行问答、生成等任务。这类模型容量大、知识广，但也带来问题：

- **隐私泄露**（记住敏感信息）  
- **版权和有害内容风险**  
- **需要选择性遗忘特定知识**（比如从模型中移除某些敏感概念）  

**Machine Unlearning（机器遗忘）**的目标是：  
> 在不重训练整个模型的前提下，让模型“忘掉”特定知识（forget-set），同时保持其原有的通用能力（retain-set）。

在多模态场景下存在两个核心难题：

1. **跨模态遗忘不一致**：已有基于单个神经元评分的方法不能捕获跨层信息流，导致某些模态遗忘不足。  
2. **通用能力退化**：直接剪掉敏感神经元往往破坏了重要的推理路径，使模型泛化性能下降。

## 2. 核心目标

> 精确定位负责存储特定知识的神经元路径，并编辑这些路径，使模型忘记目标知识，同时最大限度保留通用能力。

## 3. 方法概述：MIP-Editor

MIP-Editor（Multimodal Influential Neuron Path Editor）核心思想：

- 定位 **影响力神经元路径（Influential Neuron Path）**，即跨层信息流  
- 对路径进行编辑，实现遗忘目标知识，保留通用能力

编辑流程分两个阶段：

1. **路径定位（Localization）**  
2. **路径编辑（Editing）**

## 4. 路径定位（Localization）

路径定义：

\[
\mathcal{P} = \{ w^1, w^2, ..., w^L \}
\]

每层选择一个重要神经元，形成跨层路径。

### 4.1 文本模态：跨层梯度积分（IGI）
- 对候选路径每层神经元激活从 0 插值到真实值  
- 计算梯度积分，衡量路径贡献：

\[
\text{IGI}(\mathbf{w}) \approx \sum_{l=1}^{L} \sum_{k=1}^{m} \frac{\partial F_{\text{text}}}{\partial w_l}\Big|_{scaled}
\]

### 4.2 视觉模态：跨层 Fisher 积分（IFI）
- 针对视觉信号高维特性  
- 计算梯度平方近似二阶重要性：

\[
\text{IFI}(\mathbf{z}) \approx \sum_{l=1}^{L}\sum_{k=1}^{m} \left(\frac{\partial G}{\partial z^l}\right)^2
\]

### 4.3 路径搜索算法
- 贪心搜索，每层选贡献最高神经元，形成路径
- 计算复杂度约 O(C_{grad} * m * L * sum |w_l|)

## 5. 路径编辑（Editing）

### 5.1 剪枝
- 将路径神经元激活置零，阻断信息流：

\[
w^l \gets \mathbf{0}
\]

### 5.2 表征误导遗忘（RMisU）
- **遗忘集合**：中间表示偏向随机方向  
- **保留集合**：保持表示不偏离原模型

随机方向向量：

\[
\mathbf{v}_f = \lambda \cdot \| h^{(l)}(x_f) \|_2 \cdot \mathbf{u}
\]

### 5.3 损失函数设计

1. 保留集交叉熵：

\[
\mathcal{L}_{\text{retain}} = \text{CE}(model(X_r), Y_r)
\]

2. Forget RMisU Loss：

\[
\mathcal{L}_{\text{RMisU}}^f = \sum_{x_f\in D^f} \|h^{(l)}(x_f)-\mathbf{v}_f\|^2
\]

3. Retain RMisU Loss：

\[
\mathcal{L}_{\text{RMisU}}^r = \sum_{x_r\in D^r} \|h^{(l)}(x_r)-h^{\text{orig}}(x_r)\|^2
\]

总损失：

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{retain}} + \mathcal{L}_{\text{RMisU}}^f + \gamma \cdot \mathcal{L}_{\text{RMisU}}^r
\]

## 6. 算法伪代码

```
Input: pretrained MLLM, forget-set D^f, retain-set D^r
Output: unlearned MLLM

# 1) Localization
for modality in {text, vision}:
    for all FFN layers l:
        compute contribution scores IGI or IFI on neurons
    greedy select path P_modality

# 2) Editing
zero_out(P_text), zero_out(P_vision)
freeze all parameters except neurons in P_text, P_vision

for epoch in training_epochs:
    for batch_f, batch_r in (D^f, D^r):
        compute L_retain, L_RMisU^f, L_RMisU^r
        loss = L_retain + L_RMisU^f + gamma * L_RMisU^r
        gradient_update(neurons_in_paths)

return model
```

## 7. 实验与结果

### 数据集
- MLLMU-Bench、CLEAR  
- Metrics: Forget Performance（越低越好）、Retain Performance（越高越好）

### 主要结果示例

| 方法 | Forget ↓ | Retain ↑ |
|------|-----------|-----------|
| Vanilla | 39.20% | 37.72% |
| GA_Diff | 32.00% | 32.80% |
| KL_Min | 33.60% | 27.59% |
| NPO | 37.60% | 36.20% |
| MANU | 36.00% | 34.47% |
| **MIP-Editor** | **4.80%** | **58.19%** |

### Ablation 分析
- 无 IGI/IFI → 忘记不彻底  
- 无 RMisU → 保留性能差  
- 全模型 RMisU 无路径定位 → 泛化性能下降  
- 点式替代路径 → 遗忘率高，泛化消失  

## 8. 核心技术总结

| 技术 | 作用 |
|------|------|
| 跨层路径定位 | 捕获层间信息流 |
| IGI（文本） / IFI（视觉） | 分别适应不同模态特性 |
| Greedy搜索路径 | 近似最优路径 |
| RMisU编辑机制 | 遗忘目标语义同时保存泛化能力 |
| 冻结其他参数 | 防止遗忘破坏全局知识 |

## 9. 优缺点与未来方向

**优点**：
- 跨模态一致遗忘  
- 保留泛化能力显著优于现有方法  
- 支持大规模模型实际需求（隐私、版权）  

**局限**：
- 大模型（7B+）上效果下降  
- 路径搜索为 Greedy，可能非最优  
- 计算量较点式方法大  
- 忘记比例越高时性能下降明显
