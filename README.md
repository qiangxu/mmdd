# 强化学习在做市中的应用

本项目利用强化学习技术开发做市策略。

## 概述

- 创建了包含执行和市场数据延迟的交易模拟器，为各种做市策略提供更真实的测试环境。
- 实现了 Avellaneda-Stoikov 策略作为基准，并设计了基于 Actor-Critic (A2C) 算法的深度强化学习策略。
- 在高频数据上进行大量实验，验证了强化学习方法的有效性，并指出了其局限性。

## 基准策略

- 简单策略：每个时间步在最佳价格水平下达买卖订单。
- Avellaneda-Stoikov 策略（[论文链接](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)）

## 强化学习策略

### Environment State Space

- Price Level Distance to Midpoint
- Cumulative Notional Value at Price Level
- Notional Imbalances
- Order Flow Imbalance
- Custom RSI
- Spread

### Agent State Space

- Inventory Ratio
- Total PnL

### Action State Space

![Action State Space](images/readme/action_space.png)

### Reward Function and Training Method

- Positional PnL with inventory penalty ![Positional PnL](images/readme/reward_func.png)
- Advantage Actor-Critic (A2C). The A2C update is calculated as ![The A2C Update](images/readme/a2c_update.png)

### Function Approximator

![NN Architecture](images/readme/nn_architecture.png)


## 实验

### 环境

- 执行延迟：10ms
- 市场数据延迟：10ms
- 做市手续费：-0.004%
- 所有订单均为仅挂单类型

### 方法

- 数据：
  - BTC/USDT：2023年1月9日-20日的高频交易数据（约250万条快照）
- 训练测试集划分：
  - BTC/USDT：训练集为前三小时数据，测试集为剩余21小时数据

### 结果

- TODO

## 结论

- 开发了一种基于强化学习的做市策略。
- 将该策略与简单策略和 Avellaneda-Stoikov 策略在真实数据上进行比较，评估了其有效性。
- 强化学习策略的局限性包括：算法训练需大量时间和计算资源，超参数较多且需调优，推理速度受限。




