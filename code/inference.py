import torch
import torch.nn as nn
import pandas as pd
import os
from config import MODEL_FILE, RESULT_PATH

# 必须重新定义模型结构才能加载权重 (或者从 model.py import)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

def inference():
    print(">>> 开始预测...")
    
    # --- 1. 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print("模型加载成功")
    else:
        print("错误：未找到模型文件，请先运行训练！")
        return

    # --- 2. 模拟预测 ---
    # 实际比赛中：读取 testB_click_log.csv -> 预处理 -> model(x)
    print("正在生成预测结果...")
    
    # 伪造一个结果 DataFrame
    # 假设提交格式是 user_id, article_id
    df_result = pd.DataFrame({
        'user_id': ['user_1', 'user_2'],
        'article_id': ['art_1', 'art_2']
    })
    
    # --- 3. 保存结果 ---
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    save_path = os.path.join(RESULT_PATH, 'result.csv')
    df_result.to_csv(save_path, index=False)
    print(f"预测结果已保存至: {save_path}")

if __name__ == "__main__":
    inference()