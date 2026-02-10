import math
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

def itemcf_sim(user_item_time_dict):
    """
    计算物品相似度矩阵
    :param user_item_time_dict: {user_id:[(item_id,time),...]}
    :return: i2i_sim : {item_i:{item_j,score,...}}
    """
    #---1.初始化 ---
    #i2i_sim[i][j] 存物品i，j的相似度
    i2i_sim = {}
    #item_cnt[i] 统计物品i被多少用户点击过（分母）
    item_cnt = defaultdict(int)

    print(">>> Calculating Item-Item Similarity Matrix...")

    # --- 2. 遍历每个用户的历史行为，构建共现矩阵 ---
    # tqdm 是进度条，因为数据量大，不加这个你会以为程序死机了
    for user,item_time_list in tqdm(user_item_time_dict.items()):

        #遍历用户看过的每一个物品i
        for i,_ in item_time_list:
            item_cnt[i]+=1
            i2i_sim.setdefault(i,{})

            # 再次遍历用户看过的每一个物品 j (计算 i 和 j 的关系)
            for j,_ in item_time_list:
                if i == j:
                    continue
                i2i_sim[i].setdefault(j,0)
                weight = 1 / math.log(len(item_time_list) + 1)
                i2i_sim[i][j] += weight
            
    # --- 3. 归一化 (计算最终相似度) ---
    print(">>> Normalizing...")
    for i,related_items in i2i_sim.items():
        for j,wij in related_items.items():
            i2i_sim[i][j] = wij/math.sqrt(item_cnt[i]*item_cnt[j])
    return i2i_sim

def item_based_recommend(user_id,user_item_time_dict,i2i_sim,sim_item_topk,recall_item_num,item_topk_click):
    """
    item_base_recommend 的 Docstring
    基于物品的协同过滤
    :param sim_item_topk: 选取最相似的k个物品进行计算
    :param recall_item_num: 最终召回多少个物品
    :param item_topk_click: 全局热门物品（用于补全）
    """
    #1.获取用户历史交互过的物品
    # 这里的 user_item_time_dict 是我们在 data_loader 里搞出来的那个字典
    user_hist_items = user_item_time_dict.get(user_id,[])
    user_hist_set = {item for item,_ in user_hist_items}
    item_rank = {}

    #2.遍历用户看过的每个物品i
    for i,click_time in user_hist_items:
        #从相似度矩阵拿出和i最相似的前k个物品：
        #sorted从大到小排序
        for j,wij in sorted(i2i_sim.get(i,{}).items(),key = lambda x : x[1],reverse=True)[:sim_item_topk]:
            if j in user_hist_set:
                continue
            #计算推荐分数
            item_rank.setdefault(j,0)
            item_rank[j] += wij
        
    #3.热门补全
    if len(item_rank) < recall_item_num:
        for item in item_topk_click:
            if item not in item_rank and item not in user_hist_set:
                item_rank[item] = -999
                if len(item_rank) == recall_item_num:
                    break
    # 4. 排序并返回前 recall_item_num 个
    # 返回格式: [(item_id, score), (item_id, score)...]
    return sorted(item_rank.items(),key= lambda x : x[1],reverse= True)[:recall_item_num]