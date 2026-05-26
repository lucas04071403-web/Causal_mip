# RMU 显存问题解决方案说明

## 背景

在当前 `MIP-Editor` 实现中，`our()` 路径会在进入 `adaptive_rmu_finetune()` 前构造一份冻结参考模型：

```python
model_frozen = deepcopy(model).eval().to(args.device)
```

这会导致本地 GPU 上同时常驻：

1. 一份可训练的 `Qwen2.5-VL-3B`
2. 一份冻结的 `Qwen2.5-VL-3B`
3. `freeze_parameters()` / `PartialLinear` 带来的额外权重拷贝
4. 训练阶段的梯度、激活、优化器状态

因此即使远程评分模型 `Qwen2.5-VL-7B` 部署在超算服务器，本地依然可能因为 RMU 训练阶段的额外显存占用而 OOM。

## 目标

在**不影响现有实验定义和损失计算方式**的前提下，减少 RMU 阶段的峰值显存占用。

要求：

- 不改变 `retain_loss` 的数学定义
- 不改变 `unlearn_loss` 的数学定义
- 不改变数据顺序和训练流程
- 只优化实现方式，避免 GPU 上常驻第二份整模型

## 方案概述

将“冻结模型在线前向”替换为“编辑前原模型的参考激活预缓存”。

具体做法：

1. 在执行 `freeze_parameters()` 和路径编辑之前
2. 使用**原始未编辑模型**
3. 对 `retain_loader` 做一次前向
4. 通过 hook 抓取 `args.rmu_layer_id` 对应层的输出激活
5. 将这些激活缓存到 CPU
6. 在 `adaptive_rmu_finetune()` 中直接读取缓存，计算 `retain_loss`

这样可以去掉 GPU 上那份 `deepcopy(model)` 冻结副本。

## 为什么不影响实验

原始实现的 `retain_loss` 是：

```python
retain_loss = MSE(
    updated_retain_activations,
    frozen_retain_activations
)
```

其中 `frozen_retain_activations` 来自：

- 编辑前的原始模型
- `eval()` 模式
- `no_grad()` 前向

新的实现中，缓存值同样来自：

- 编辑前的原始模型
- `eval()` 模式
- `no_grad()` 前向

因此只要 `retain_loader` 的批次顺序不变，训练时取出的参考激活与原实现完全等价。

当前代码里 `retain_loader` 没有开启 `shuffle`，因此这个前提成立。

## 实现细节

### 1. 新增公共函数

文件：

- `MIP-Editor/train_eval.py`

新增了以下函数：

- `get_model_device(model)`
- `forward_with_cache(model, inputs, module, no_grad=True)`
- `get_rmu_layer_module(model, layer_id, model_role)`
- `precompute_rmu_retain_activations(model, loader, layer_id)`

作用：

- 统一获取模型所在设备
- 通过 forward hook 抓取指定层激活
- 找到 RMU 使用的语言层
- 将 `retain_loader` 的参考激活预先缓存到 CPU

### 2. 在 `our()` 中前移缓存时机

文件：

- `MIP-Editor/ours.py`

原本流程是：

1. 计算路径
2. 构造 `deepcopy(model)` 作为冻结模型
3. 执行路径编辑
4. 调用 `adaptive_rmu_finetune()`

现在改为：

1. 计算路径
2. 在原始模型上预缓存 `retain` 参考激活
3. 执行路径编辑
4. 调用 `adaptive_rmu_finetune()`，传入缓存而不是整模型副本

### 3. `adaptive_rmu_finetune()` 保持兼容

文件：

- `MIP-Editor/train_eval.py`

当前实现兼容两种输入：

1. 旧模式：`frozen_model` 是冻结模型对象
2. 新模式：`frozen_model` 是激活缓存列表

因此这次改动是实现层面的优化，不会破坏原函数接口的基本用途。

## 关键收益

### 显存收益

最大收益来自移除：

```python
deepcopy(model).eval().to(args.device)
```

这通常会直接节省一整份 3B 模型在 GPU 上的显存占用。

### 行为一致性

保留了原有：

- `retain_loss`
- `unlearn_loss`
- `rmu_layer_id`
- `rmu_alpha / rmu_beta / rmu_coeffs`
- `retain_loader` / `forget_loader` 训练逻辑

所以实验目标和对比口径不变。

## 适用前提

该方案成立的前提是：

1. `retain_loader` 的 batch 顺序固定
2. 缓存参考激活时使用的是编辑前原模型
3. 训练时读取缓存的顺序与 `retain_loader` 顺序一致

当前项目满足这些前提。

## 新增保护机制

为了避免后续代码调整后出现“缓存与 batch 错配但不易察觉”的问题，当前实现额外加入了两层保护。

### 1. 检查 `retain_loader` 是否为随机采样

缓存方案要求 `retain_loader` 在每个 epoch 中的遍历顺序保持一致。

因此在预缓存参考激活时，代码会检查：

- `retain_loader.sampler` 是否为 `RandomSampler`

如果是，则直接报错并提示关闭 `shuffle` / 随机采样，而不是继续训练。

这样可以避免：

- 缓存来自顺序 A
- 训练时 batch 顺序变成顺序 B
- 最终 `retain_loss` 使用了错误参考激活

### 2. 检查缓存长度与 `retain_loader` 长度是否一致

实现中增加了长度校验：

- 预缓存完成后，检查 `len(cached_activations) == len(retain_loader)`
- 进入 `adaptive_rmu_finetune()` 后，如果传入的是缓存列表，也会再次检查缓存长度是否等于 `len(retain_loader)`

如果不一致，则直接报错。

这样可以避免：

- 缓存不完整
- loader 长度变化
- 后续访问缓存时出现隐藏错误或错配

## 关于 `finetune_epochs > 1`

当 `finetune_epochs > 1` 时，当前实现不会因为重复训练 epoch 而导致缓存越界。

原因是训练循环写法为：

```python
for epoch in range(args.finetune_epochs):
    for idx, (batch_r, batch_f) in enumerate(zip(retain_loader, forget_loader)):
```

这里的 `idx` 会在**每个 epoch 内重新从 0 开始计数**，因此不会累加到下一轮 epoch。

也就是说：

- 第 1 轮 epoch 访问缓存 `0 ... len(retain_loader)-1`
- 第 2 轮 epoch 仍然访问缓存 `0 ... len(retain_loader)-1`

所以只要 `retain_loader` 顺序稳定、长度不变，缓存可以安全地在多个 epoch 中重复使用。

## 已修改文件

- `MIP-Editor/ours.py`
- `MIP-Editor/train_eval.py`

## 后续建议

如果显存仍然紧张，可以继续做以下优化，但这些属于下一层优化，不属于这次“完全不改变实验定义”的最小方案：

1. 优化 `PartialLinear`，减少额外权重 clone
2. 开启 gradient checkpointing
3. 降低训练时临时激活峰值
4. 仅对更少层做路径编辑

这些优化都需要进一步评估是否会影响速度、实现复杂度或训练稳定性。

## 总结

本方案的本质是：

- **保留 RMU 的实验定义**
- **去掉 GPU 上冻结参考整模型副本**
- **改为预缓存编辑前原模型的参考激活**

这是一种实现优化，而不是实验设定修改，因此适合作为当前 OOM 问题的首选解决方案。
