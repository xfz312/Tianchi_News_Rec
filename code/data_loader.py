import pandas as pd
import numpy as np
from utils import reduce_mem

# 这里的 data_path 参数，我们在 main 里面会传入 config.DATA_PATH
def get_all_click_df(data_path, offline=True):
    """
    读取并合并数据
    offline=True: 线下验证模式，只读取部分数据快速跑通（Debug用）
    offline=False: 线上提交模式，读取全量数据
    """
    if offline:
        print(">>> [Offline Mode] Loading partial data for debugging...")
        # 调试时只读前 20000 行，节省时间
        all_click = pd.read_csv(data_path + '/train_click_log.csv', nrows=20000)
    else:
        print(">>> [Online Mode] Loading full data...")
        trn_click = pd.read_csv(data_path + '/train_click_log.csv')
        tst_click = pd.read_csv(data_path + '/testA_click_log.csv')
        # 把训练集和测试集拼起来，统一进行编码
        all_click = pd.concat([trn_click, tst_click])

    # 去重：同一个用户在同一时间点击同一篇文章，可能是误触，去掉
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    
    # 瘦身：减少内存
    all_click = reduce_mem(all_click)
    return all_click

def get_user_item_time(click_df):
    """
    【面试考点】
    把 DataFrame 转换成字典： {user_id: [(item_id, time), (item_id, time)...]}
    这是 ItemCF 算法计算共现矩阵的输入格式。
    """
    print(">>> Constructing User-Item-Time dictionary...")
    # 按照时间排序，这对于序列推荐很重要
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    # groupby 操作：按用户分组，把该用户的所有文章和时间打包成一个列表
    user_item_time_df = (
        click_df.groupby('user_id')[['click_article_id', 'click_timestamp']]
        .apply(lambda x: make_item_time_pair(x))
        .reset_index()
        .rename(columns={0: 'item_time_list'})
    )
    
    # 转成字典
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict

def get_item_topk_click(click_df, k):
    """
    获取全局最热门的 K 篇文章（用于冷启动补全）
    """
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click