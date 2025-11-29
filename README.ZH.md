# 📄 **README.ZH.md（最终版）**

````markdown
# 多模态评分合理性校对模型（Data Proofreading Model）

本项目提出了一种轻量级的多模态回归模型，用于问卷或评价体系中的评分合理性校对。  
模型同时利用两段文本（用户作答文本、评分规则说明文本）与多种结构化数值特征（用户评分、历史统计、题面属性等），输出一个连续的“评分合理性分数”。

---

## 🌐 语言版本
- 🇨🇳 **简体中文**（当前）
- 🇺🇸 [English Version](README.EN.md)

---

# 1. 摘要（Abstract）

本项目提出了一种轻量级的多模态回归模型，用于问卷或评价体系中的“分值合理性校对”任务。模型同时利用两类文本信息（用户作答文本与评分规则说明文本）以及多种结构化数值特征（用户评分、历史统计特征及其他题面属性）。文本部分通过共享的 TextCNN + BiGRU 编码器提取语义表示，数值特征通过多层感知机（MLP）进行编码。最终将多模态特征融合后输入回归头，输出连续的合理性评分。

---

# 2. 研究动机（Motivation）

在实际问卷系统中常出现以下问题：
- 文本反馈与评分严重不匹配  
- 评分分布异常  
- 不同题型评分标准不一致  
- 人工质检成本高、效率低  

因此需要一个 **自动化、轻量、可部署** 的校对模型，帮助提升评分数据质量。

---

# 3. 方法（Method）

## 3.1 任务定义（Task Definition）

给定：
- 用户作答文本 \(T_{answer}\)  
- 评分规则文本 \(T_{rule}\)  
- 数值特征 \(X_{num}\)

模型预测合理性分数：
\[
\hat{y} = f(T_{answer}, T_{rule}, X_{num})
\]

优化目标：
\[
\mathcal{L} = \text{MAE}(y, \hat{y}) \quad \text{or} \quad \text{RMSE}(y, \hat{y})
\]

---

## 3.2 模型结构（Model Architecture）

### （1）共享文本编码器
- Embedding  
- TextCNN（多窗口卷积）  
- BiGRU（上下文建模）  
- Pooling（Max/Mean）

输出：  
\[
h_{answer},\ h_{rule}
\]

### （2）数值特征编码器
\[
h_{num} = \text{MLP}(X_{num})
\]

### （3）多模态融合与回归头
\[
h = [ h_{answer};\ h_{rule};\ h_{num} ]
\]
\[
\hat{y} = \text{MLP}_{reg}(h)
\]

---

## 3.3 模型流程图（Flow Diagram）

```text
用户作答文本 ──► Embedding ──► TextCNN ──► BiGRU ──► h_answer
评分规则文本 ──► Embedding ──► TextCNN ──► BiGRU ──► h_rule
数值特征 ───────────────► MLP ───────────► h_num
                 h_answer + h_rule + h_num ──► concat ──► 回归头 ──► ŷ
````

---

# 4. 数据集（Dataset）

字段示例：

| 字段名              | 类型     | 描述     |
| ---------------- | ------ | ------ |
| text_answer      | string | 用户作答文本 |
| text_rule        | string | 评分规则文本 |
| score_user       | float  | 用户分值   |
| score_stats_mean | float  | 历史均值   |
| score_stats_std  | float  | 历史标准差  |
| other_features   | dict   | 结构化特征  |
| label            | float  | 合理性标签  |

JSON 示例：

```json
{
  "text_answer": "服务人员态度很好，非常耐心。",
  "text_rule": "评分需基于态度、速度、解决质量。",
  "score_user": 5,
  "score_stats_mean": 4.2,
  "score_stats_std": 0.5,
  "other_features": {"question_type": 1, "duration": 35},
  "label": 0.92
}
```

---

# 5. 实验设置（Experiments）

* 数据划分：70% 训练 / 15% 验证 / 15% 测试
* 超参数：Embedding=128, GRU=128, CNN filters=64, lr=1e-3
* Baselines：

  * 规则基线
  * 数值特征 MLP
  * 文本特征 BiGRU / CNN

---

# 6. 实验结果（Results）

| 模型                     | MAE ↓     | RMSE ↓    | R² ↑     |
| ---------------------- | --------- | --------- | -------- |
| Rule-based             | 0.215     | 0.302     | 0.41     |
| Numeric-only           | 0.174     | 0.261     | 0.55     |
| Text-only              | 0.166     | 0.249     | 0.58     |
| **Multi-modal (Ours)** | **0.138** | **0.221** | **0.67** |

---

# 7. 结论（Conclusion）

模型轻量、高效、可部署，适用于自动质检与智能评分。

---

# 8. 未来方向（Future Work）

* 轻量 Transformer
* 可解释性模块
* 多任务学习
* 自动数据清洗
* 模型蒸馏与在线部署

````
