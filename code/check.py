import sys
import os
import pandas as pd

# 1. 动态设置路径（无论你在哪里运行，都能找到 tcdata）
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(curr_path)
data_path = os.path.join(root_path, 'tcdata')

# 2. 定义文件名
train_file = os.path.join(data_path, 'train_click_log.csv')
article_file = os.path.join(data_path, 'articles.csv')

# 3. 尝试读取前 5 行 (用 nrows=5 防止内存爆炸)
print(f"正在读取数据: {train_file}")
try:
    df_train = pd.read_csv(train_file, nrows=5)
    print("✅ 训练集读取成功！前5行数据如下：")
    print(df_train)
except FileNotFoundError:
    print(f"❌ 错误：找不到文件。请确认你把csv放到了 {data_path} 目录下")

print("-" * 30)

print(f"正在读取文章数据: {article_file}")
try:
    df_article = pd.read_csv(article_file, nrows=5)
    print("✅ 文章表读取成功！")
    print(df_article.columns.tolist()) # 看看有哪些列
except FileNotFoundError:
    print("❌ 文章表未找到")