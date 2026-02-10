import os

# --- 1. 动态获取项目根目录 ---
# 解释：__file__ 是当前文件的路径，dirname 取父目录。
# 这样无论你的项目在 D盘 还是 Linux 的 /home 下，都能自动找到位置。
curr_path = os.path.dirname(os.path.abspath(__file__)) # D:\pytorch_stu\Tianchi_News_Rec\code
ROOT_PATH = os.path.dirname(curr_path)                 # D:\pytorch_stu\Tianchi_News_Rec

# --- 2. 定义各文件夹路径 ---
DATA_PATH = os.path.join(ROOT_PATH, 'tcdata')
USER_DATA_PATH = os.path.join(ROOT_PATH, 'user_data')
MODEL_PATH = os.path.join(USER_DATA_PATH, 'model_data')
TMP_PATH = os.path.join(USER_DATA_PATH, 'tmp_data')
RESULT_PATH = os.path.join(ROOT_PATH, 'prediction_result')

# --- 3. 定义具体文件路径 (方便调用) ---
TRAIN_FILE = os.path.join(DATA_PATH, 'train_click_log.csv')
TEST_FILE = os.path.join(DATA_PATH, 'testA_click_log.csv')
ARTICLES_FILE = os.path.join(DATA_PATH, 'articles.csv')

# 定义模型保存路径 (ItemCF 的相似度矩阵 pickle 文件)
ITEMCF_SIM_PKL = os.path.join(MODEL_PATH, 'itemcf_i2i_sim.pkl')