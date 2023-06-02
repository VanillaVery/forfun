
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""# 데이터 불러오기"""
data=pd.read_csv(r"C:\Users\윤유진\Downloads\[넥슨] 플랫폼분석팀_과제\serviceUsageMonthly.csv")
data['month']=data['monthCode'].apply(lambda x: str(x)[-2:])
data['time_per_visit'] = data['timeSpent']/data['numberOfDays']

#%%
"""유저-서비스 행렬 만들기"""
def set_temp(x): 
    return set(x)

user_anal=pd.pivot_table(data,
                index='userId',
               columns='serviceId',
               values='month',
               aggfunc=set_temp)	

user_anal[~user_anal.isna()] = 1.0
user_anal.fillna(0,inplace=True)
# 행렬 완성 

#%%
"""행렬분해"""
#유저-아이템 메트리스를 유저 매트릭스, 아이템 메트릭스로 분해 

import scipy

def get_svd_prediction(user_item_matrix, k):
    u,sig,vh = scipy.sparse.linalg.svds(user_item_matrix.to_numpy(), k=k)
    preds = np.dot(np.dot(u,np.diag(sig)),vh)

    preds = pd.DataFrame(preds,columns = user_item_matrix.columns, index=user_item_matrix.index)
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    return preds

predictions = get_svd_prediction(user_anal, k=10)
#예측 끝 
#%%

user_id = 42
user_service_ids = data[data.userId == user_id].serviceId

#복원된 행렬에서 유저 row만 가져온 뒤 내림차순으로 정리
user_predictions = predictions.loc[user_id].sort_values(ascending=False)
#이미 유저가 본 영화는 제외 
user_predictions = user_predictions[~user_predictions.index.isin(set(user_service_ids))]
set(user_service_ids)