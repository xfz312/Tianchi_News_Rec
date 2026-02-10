# 📰 Tianchi News Recommendation - ItemCF Baseline

> 阿里云天池“新闻推荐场景下的用户行为预测”挑战赛 - 召回阶段基线方案

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Status](https://img.shields.io/badge/Status-Baseline-green)

## 📖 项目背景 (Background)
本项目致力于解决新闻推荐场景下的**冷启动**与**个性化召回**问题。基于大规模用户点击日志（Click Logs），构建用户兴趣画像，并预测用户未来可能点击的新闻文章。

当前版本实现了 **基于物品的协同过滤 (Item-based Collaborative Filtering)** 算法，作为整个推荐系统的召回（Recall）基线。

## 🛠️ 核心算法 (Algorithm)
本项目没有直接调用现成库，而是**手写实现**了带有工业界常用 Trick 的 ItemCF 算法：

### 1. 相似度计算 (Similarity Calculation)
我们采用 **余弦相似度 (Cosine Similarity)** 衡量物品间的共现关系，并引入了以下优化：

* **用户活跃度惩罚 (IUF - Inverse User Frequency)**:
    降低活跃用户（点击狂魔）对物品相似度的贡献。公式如下：
    $$w_{ij} = \frac{\sum_{u \in U(i) \cap U(j)} \frac{1}{\log(1+|N(u)|)}}{\sqrt{|N(i)| \cdot |N(j)|}}$$
    
* **归一化 (Normalization)**: 
    通过分母 $\sqrt{|N(i)| \cdot |N(j)|}$ 消除热门物品的偏差。

### 2. 推荐策略 (Recommendation)
* **个性化召回**: 基于用户历史行为序列进行相似物品推荐。
* **冷启动处理 (Cold Start)**: 实现了 **全局热门补全 (Most Popular Fallback)** 策略，解决新用户或历史行为稀疏用户的推荐问题。

## 📂 项目结构 (Structure)
```text
Tianchi_News_Rec/
├── code/                   # 核心代码目录
│   ├── config.py           # [配置] 路径与全局参数配置
│   ├── data_loader.py      # [数据] 数据读取、清洗与预处理
│   ├── itemcf.py           # [核心] ItemCF 算法实现 (相似度计算 & 推荐)
│   ├── submit.py           # [工具] 提交文件生成
│   ├── utils.py            # [工具] 内存压缩等通用工具
│   └── main.py             # [入口] 主程序
├── tcdata/                 # (Git忽略) 原始数据集
├── user_data/              # (Git忽略) 中间结果，如 pickle 缓存
├── prediction_result/      # (Git忽略) 最终生成的 result.csv
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明
```

## 🚀 快速运行 (Quick Start)

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据准备
请前往 阿里云天池官网 下载数据集，解压并放入 tcdata/ 目录，确保结构如下：
tcdata/train_click_log.csv
tcdata/testA_click_log.csv
tcdata/articles.csv

### 3. 运行代码
```bash
python code/main.py
```

程序将自动执行以下流程：
· 读取数据并进行内存压缩（Memory Reduction）。
· 构建 User-Item-Time 倒排索引。
· 计算 Item-Item 相似度矩阵（计算后会自动保存为 .pkl 文件，下次运行直接加载）。
· 为测试集用户生成 Top-5 推荐。
· 在 prediction_result/ 下生成 result.csv。`

## 📊 实验结果 (Results)
ItemCF (Baseline) + Log-weighted : 0.1026