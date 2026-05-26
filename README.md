# TAAC × KDD Cup 2026 — 广告 CVR 预测

> 腾讯广告算法大赛（TAAC × KDD Cup 2026）参赛项目  
> 赛题：面向大规模推荐的序列建模与特征交互统一化  
> Evaluation AUC：**0.8239**

---

## 赛题简介

基于百万级真实广告日志数据，在统一架构内联合建模用户多域行为序列（4 域，序列长度最大 512）与 120 维多字段特征（46 维离散用户特征、10 维稠密特征、14 维商品特征），预测广告点击后转化率（pCVR），评估指标为 AUC。

官方 baseline 架构为 **HyFormer**（Query Decoding + Query Boosting 交替堆叠的统一 Transformer），本项目在 HyFormer baseline 基础上进行了多项改进。

---

## 改进方案与实验结果

以下为并行消融实验结果（在 baseline AUC 0.807 基础上的单项增益）：

| 改动 | 说明 | Evaluation AUC 增益 |
|------|------|-------------------|
| UserDenseGateFusion | 融合 int-dense 对齐特征，gate 机制调制 embedding 强度 | +7k |
| 时间特征 | 从 timestamp 提取 hour/weekday 循环编码，拼入 user dense | +6k |
| Focal Loss（激进参数） | α=0.5，γ=3.0，提升对难样本的关注度 | +2k |
| Semi-local Causal Mask | 借鉴 HSTU，在序列 Transformer Encoder 中建模行为时序依赖 | +1.5k |
| DIN 交叉注意力 | 以 item pool 为 query 对全域序列做 cross-attention | +1k |
| Group Tokenizer | 按语义分组替代均匀切分 | +1k |
| 89-91 特征 log1p 变换 | dense 对齐特征做 log1p 预处理再融合 | +1k |

> **注**：k = 万分之一 AUC，为推荐系统领域的常用增益单位。各项改动组合叠加后最终 Evaluation AUC 达到 **0.8136**。

无效/有害方案（同样做了消融，供参考）：

| 改动 | 结果 | 原因分析 |
|------|------|---------|
| HSTU Mixture of Transducers | -6k | valid AUC 高但 eval 低，严重过拟合 |
| Kimi 残差注意力 | -4k | 可能实现有误，或不适合短序列场景 |
| 额外 Dropout 层 | -3k | 过强正则化损害小模型的拟合能力 |
| 辅助目标（CTR 多任务） | -3k | CTR 与 CVR 梯度冲突，干扰主任务 |
| 注意力内加 SiLU + Gate | -2k | 与已有 Gated Attention 重复，冗余 |
| Scaling 参数（增大模型） | -2k | 数据量不足以支撑更大模型容量 |
| RoPE | -1k | 推荐序列无绝对位置语义，引入噪声 |

---

## 核心改动说明

### 1. UserDenseGateFusion（+7k）

**动机**：数据集中 `user_int_feats_{62-66, 89-91}` 与 `user_dense_feats_{62-66, 89-91}` 逐元素对齐——int 列是实体类别 ID，dense 列是对应实体的统计信号（停留时间、得分等）。默认模型两条路径完全独立，未利用对齐关系。

**做法**：
```
int_vals  → Embedding → int_emb   (B, L, emb_dim)
dense_vals → log1p → Linear → dense_proj  (B, L, emb_dim)
gate = sigmoid(W · dense_proj)
fused = int_emb * gate + dense_proj
→ mean-pool → 汇聚所有 pair groups → 投影成 3 个 NS Token
```

**关键细节**：融合后将 `user_dense_feats` 中对应列置零再送入主路径的 `user_dense_proj`，避免同一信息被模型重复读取（信息泄露）。

**log1p 的原因**：dense 值为统计量，量纲差异大（停留时间 0-10000 秒，得分 0-1）。`log1p` 压缩长尾分布，同时保证 `log1p(0)=0` 不影响 padding 位置。

---

### 2. 时间特征（+6k）

**动机**：数据集 `timestamp` 仅用于序列 time bucket 计算，样本本身的时间信息（几点、周几）未被利用。CVR 与时间有明显的周期性关系。

**做法**：向量化提取 5 个循环时间特征，拼入 `user_dense_feats` 末尾：

```python
seconds_of_day = timestamps % 86400
hours = seconds_of_day / 3600.0
weekdays = (timestamps // 86400 + 3) % 7  # 1970-01-01 是周四

user_dense[:, offset]   = np.sin(2π × hours / 24)    # hour_sin
user_dense[:, offset+1] = np.cos(2π × hours / 24)    # hour_cos
user_dense[:, offset+2] = np.sin(2π × weekdays / 7)  # weekday_sin
user_dense[:, offset+3] = np.cos(2π × weekdays / 7)  # weekday_cos
user_dense[:, offset+4] = (weekdays >= 5).float()     # is_weekend
```

使用 sin/cos 编码而非原始数值，保持循环量的连续性（23 点与 0 点应相近）。

---

### 3. Semi-local Causal Mask（+1.5k）

**动机**：默认 TransformerEncoder 的 self-attention 是双向的，不尊重用户行为的时序因果关系。借鉴 Meta HSTU 的设计，加入因果掩码使每个位置只能 attend 到自己和之前的位置。

```python
if self.causal:
    attn_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
```

**注意**：加入自定义 mask 后 PyTorch SDPA 无法使用 FlashAttention，显存从 O(N) 变为 O(N²)，需将 batch_size 从 256 降至 128。

---

### 4. DIN 交叉注意力（+1k）

**动机**：不同目标商品应激活用户历史中不同的兴趣。默认 query tokens 对所有商品一样，缺乏 target-aware 的动态兴趣提取。

```
u_pool = mean_pool(ns_tokens)           # 用户全局表示
i_pool = mean_pool(item_ns_tokens)      # 当前商品表示
i_attn = CrossAttention(query=i_pool, kv=all_seq_tokens)  # 商品视角的兴趣提取
din_token = SiLU(Linear(concat(u_pool, i_pool, i_attn)))
→ 拼到 ns_tokens 后送入 query_generator
```

---

## 文件结构

```
├── model.py       # 模型定义（含 UserDenseGateFusion、DINModule、Causal Mask）
├── dataset.py     # 数据加载（含时间特征提取）
├── train.py       # 训练入口（含 dense pair groups 自动检测、Focal Loss 配置）
├── infer.py       # 推理入口（与训练端对齐）
└── README.md
```

> **注**：`trainer.py` 与 `utils.py` 使用比赛平台默认实现，未包含在本仓库中。

---

## 环境依赖

```
torch >= 2.0
numpy
pyarrow
pandas
scikit-learn
```

---

## 数据集

数据集来自 TAAC × KDD Cup 2026 官方，不可公开分发。  
数据集介绍：https://algo.qq.com

---

## 参考论文

- [HyFormer](https://arxiv.org/abs/2506.xxxxx)：比赛官方 baseline 架构，Query Decoding + Query Boosting 统一推荐框架
- [OneTrans](https://arxiv.org/abs/2510.26104)：单 Transformer 统一序列建模与特征交互
- [InterFormer](https://arxiv.org/abs/2411.09852)：异构特征交互学习
- [HSTU](https://arxiv.org/abs/2402.17152)：Meta 万亿参数序列转导推荐模型，Causal Mask 借鉴来源
- [DIN](https://arxiv.org/abs/1706.06978)：Deep Interest Network，target-aware 注意力机制

