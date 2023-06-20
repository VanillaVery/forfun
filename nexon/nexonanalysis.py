
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
"""사업현황"""
data.groupby('serviceId')['userId'].nunique() # 상위 3개 서비스가 전체의 약 60퍼센트 차지

# 월별로 본 유저 수 
user_anal=pd.pivot_table(data,
                index='serviceId',
               columns='month',
               values='userId',
               aggfunc='nunique')	
user_anal.to_csv("월별유저수.csv")

# 시계열 클러스터링을 이용한 유저 수 추이 클러스터링 
#%%
"""서비스 별 특이 사항"""
# """접속일vs접속시간 """
data.groupby('serviceId')['timeSpent','numberOfDays'].median()
data.groupby('serviceId')['time_per_visit'].median()

data[['timeSpent','numberOfDays']].median()

user_anal=pd.pivot_table(data,
                index='serviceId',
               columns='month',
               values='numberOfDays',
               aggfunc='median')	
# user_anal.to_csv("aa.csv")

# 4,12,19,22 번의 유저들은 다른 서비스의 유저보다 헤비 유저가 많을까?
a = data[data['serviceId'].isin([4,12,19,22])]\
    .groupby('userId')['timeSpent','numberOfDays'].median()

b = data[~data['serviceId'].isin([4,12,19,22])]\
    .groupby('userId')['timeSpent','numberOfDays'].median()

#헤비 유저의 정의 : 상위 30%의 시간(분/달), 접속 횟수(일)를 보유 
# # 시간: 달에 392분, 하루에 6번 이상 들어오는 사람
is_heavyuser=(data['timeSpent']>=np.percentile(data['timeSpent'].fillna(0),70)) &\
            (data['numberOfDays']>=np.percentile(data['numberOfDays'].fillna(0),70))

# 4,12,19,22 번을 하면서 헤비유저인 사람
len(data.loc[(data['serviceId'].isin([4,12,19,22])) & is_heavyuser,'userId'].unique()) / len(data.loc[data['serviceId'].isin([4,12,19,22]),'userId'].unique())
#64% ... 호도도...
# 나머지 게임을 하면서 헤비유저인 사람
len(data.loc[~(data['serviceId'].isin([4,12,19,22])) & is_heavyuser,'userId'].unique()) / len(data.loc[~data['serviceId'].isin([4,12,19,22]),'userId'].unique())
#39% 

#19번 게임을 하면서 헤비유저인 사람
data['month'] = data['month'].apply(lambda x: int(x))
for i in range(1,13):
    print(round(len(data.loc[(data['serviceId'].isin([19])) & 
                 (data['month'] == i) & 
                 is_heavyuser,'userId'].unique()) / 
        len(data.loc[data['serviceId'].isin([19]) &
                    (data['month'] == i) 
                    ,'userId'].unique()),2))

#가설 2: 1번서비스의 유저 흐름은 보합세이지만, 꾸준히 하는 사람은 없을 것 같다.
data_copy=data.copy()
data_copy['serviceId'] = data_copy['serviceId'].apply(lambda x: str(x))

def set_temp(x): 
    return set(x)
    

data_copy = pd.pivot_table(data_copy,
                          index = 'userId',
                          columns = 'month',
                          values = 'serviceId',
                          aggfunc = set_temp)
data_copy.to_csv('유저분석.csv')

#%%
"""by 승환"""
# 게임별로 12월 내내 하는 유저가 얼마나 있는지 알아보자 
data_copy_ = data_copy.apply(lambda s: s.fillna({i: {99} for i in data_copy.index}))

# service='1'
service = [str(i) for i in range(1, 23)]
service_num_dict = {}
for s_num in tqdm(service):
    count = np.zeros(len(data_copy_), )
    for col in data_copy_.columns: #달 
        count += data_copy_[col].apply(lambda x: 1 if s_num in x else 0).to_numpy()
    service_num_dict[s_num] = count

for month in range(1,13):
    for s_num in service:
        print((service_num_dict[s_num] == month).sum(), end= ' ')
    print()

#%%
# 유저마다 몇 개의 게임을 즐기고 있을까?
data_copy=data.copy()
data_copy['serviceId'] = data_copy['serviceId'].apply(lambda x: str(x))

def set_temp(x): 
    return set(x)
    

data_copy = pd.pivot_table(data_copy,
                          index = 'userId',
                        #   columns = 'month',
                          values = 'serviceId',
                          aggfunc = set_temp)

#게임을 몇개 해?
data_copy['count']=data_copy['serviceId'].apply(lambda x: len(x))
#게임 8을 하는 사람들은 게임을 평균 몇 개 해? (중위수)
data_copy.loc[data_copy['serviceId'].apply(lambda x:'18' in x),'count'].median()

data_copy['count'].median()

#%%
"""유저 분석"""
#1. 유저들은 평균 몇 개의 서비스를 사용하는가?
user_anal=data.groupby('userId')['serviceId'].nunique()
user_anal.describe()
np.percentile(user_anal,90)
#2. 유저들은 평균 어느 정도의 시간을 서비스에 사용하는가?
user_anal=data.groupby('userId')['timeSpent'].mean()
user_anal.describe()
np.percentile(user_anal.fillna(0),90)
data.columns
#3. 유저들은 평균 몇 일을 서비스에 사용하는가?
user_anal=data.groupby('userId')['numberOfDays'].mean()
user_anal.describe()
np.percentile(user_anal,90)

"""1년의 절반 이상 서비스를 방문하는 유저의 비율"""
# 1년의 절반 이상 서비스를 방문하는 유저의 비율

data_copy = pd.pivot_table(data_copy,
                          index = 'userId',
                          columns = 'month',
                          values = 'serviceId',
                          aggfunc = set_temp)	

user_anal['active'] = 12 - user_anal.isna().sum(axis=1) #이용 기간 구함

user_anal['active'].hist(bins=12)

len(user_anal[user_anal['active']>=12])/len(user_anal)

#꽤나... 12월 내내 이용하는 유저의 비율이 높음 

"""유저 마의 n개월 분석"""
"""12개월 내내 하지 않는 유저들은 연속 몇 달 접속 후 이탈하는가?"""
# 연속으로 몇 달을 게임에 접속한 유저들의 정보 추출
data_copy = data_copy.apply(lambda s: s.fillna({i: {99} for i in data_copy.index}))

for col in data_copy.columns: #달 
        data_copy[col] = data_copy[col].apply(lambda x: 0 if 99 in x else 1)

row = data_copy.iloc[3,:]
def count_continuous_months(row):
    counts = row.cumsum()  # 누적 접속 개월수 계산
    max_continuous_months = counts - counts.shift(1).fillna(0)
    maxcount = 0
    for i,c in enumerate(max_continuous_months):
        if c == 0:
            continue
        if c == 1:
            maxcount += 1
            if max_continuous_months[i+1] == 0:
                break
            else: continue
    return maxcount

data_copy['max_continuous_months'] = data_copy.apply(count_continuous_months, axis=1)

data_copy['max_continuous_months'].hist(bins=12, weights=np.ones(len(data_copy)) / len(data_copy))
#%%
"""유저 여정 분석"""
data['quarter']=data['month'].apply(lambda x: 1 if int(x) in [1,2,3] 
                                    else 2 if int(x) in [4,5,6]
                                    else 3 if int(x) in [7,8,9]
                                    else 4 )
data_copy=data.copy()
data_copy['serviceId'] = data_copy['serviceId'].apply(lambda x: str(x))

def set_temp(x): 
    return set(x)
    

data_copy = pd.pivot_table(data_copy,
                          index = 'userId',
                          columns = 'quarter',
                          values = 'serviceId',
                          aggfunc = set_temp)

data_copy.to_csv("유저여정분석.csv")
#%%
"""11번 서비스"""
data_copy.columns=['1','2','3','4']
data_copy = data_copy.apply(lambda s: s.fillna({i: {99} for i in data_copy.index}))

data1_11 = data_copy.loc[data_copy['1'].apply(lambda x: '11' in x)]
#1분기에 서비스 11을 쓰는 사람들

#함께 쓰는 서비스 
set1=[]
data1_11['1'].apply(lambda x: set1.extend(x))

from collections import Counter
Counter(set1)

# 11번 서비스를 쓰는 사람들은 2분기에 어떻게 변화하는가 
data2_11 = data1_11.loc[data_copy['2'].apply(lambda x: '11' in x)]

data2_not11 = data1_11.loc[data_copy['2'].apply(lambda x: '11' not in x)]

set1=[]
data2_not11['2'].apply(lambda x: set1.extend(x))
Counter(set1)

# 11번 서비스를 쓰는 사람들은 3분기에 어떻게 변화하는가
data3_11 = data2_11.loc[data_copy['3'].apply(lambda x: '11' in x)]
data3_not11 = data2_11.loc[data_copy['3'].apply(lambda x: '11' not in x)]

set1=[]
data3_not11['3'].apply(lambda x: set1.extend(x))
Counter(set1)

# 11번 서비스를 쓰는 사람들은 4분기에 어떻게 변화하는가
data4_11 = data3_11.loc[data_copy['4'].apply(lambda x: '11' in x)]
data4_not11 = data3_11.loc[data_copy['4'].apply(lambda x: '11' not in x)]
data4_not11.shape
set1=[]
data4_not11['4'].apply(lambda x: set1.extend(x))
Counter(set1)
#%%
"""17번 서비스"""
data_copy.columns=['1','2','3','4']
data_copy = data_copy.apply(lambda s: s.fillna({i: {99} for i in data_copy.index}))

data1_11 = data_copy.loc[data_copy['1'].apply(lambda x: '17' in x)]
#1분기에 서비스 11을 쓰는 사람들

#함께 쓰는 서비스 
set1=[]
data1_11['1'].apply(lambda x: set1.extend(x))

from collections import Counter
Counter(set1)

# 11번 서비스를 쓰는 사람들은 2분기에 어떻게 변화하는가 
data2_11 = data1_11.loc[data_copy['2'].apply(lambda x: '17' in x)]

data2_not11 = data1_11.loc[data_copy['2'].apply(lambda x: '17' not in x)]

set1=[]
data2_not11['2'].apply(lambda x: set1.extend(x))
Counter(set1)

# 11번 서비스를 쓰는 사람들은 3분기에 어떻게 변화하는가
data3_11 = data2_11.loc[data_copy['3'].apply(lambda x: '17' in x)]
data3_not11 = data2_11.loc[data_copy['3'].apply(lambda x: '17' not in x)]

set1=[]
data3_not11['3'].apply(lambda x: set1.extend(x))
Counter(set1).most_common()

# 11번 서비스를 쓰는 사람들은 4분기에 어떻게 변화하는가
data4_11 = data3_11.loc[data_copy['4'].apply(lambda x: '17' in x)]
data4_not11 = data3_11.loc[data_copy['4'].apply(lambda x: '17' not in x)]
data4_not11.shape
set1=[]
data4_not11['4'].apply(lambda x: set1.extend(x))
Counter(set1).most_common()
#%%
"""19번 서비스 유저유지율"""
data1_19 = data_copy.loc[data_copy['1'].apply(lambda x: '19' in x)]
#1분기에 서비스 19을 쓰는 사람들

# 19번 서비스를 쓰는 사람들은 2분기에 어떻게 변화하는가 
data2_19 = data1_19.loc[data_copy['2'].apply(lambda x: '19' in x)]


# 19번 서비스를 쓰는 사람들은 3분기에 어떻게 변화하는가
data3_19 = data2_19.loc[data_copy['3'].apply(lambda x: '19' in x)]

# 11번 서비스를 쓰는 사람들은 4분기에 어떻게 변화하는가
data4_19 = data3_19.loc[data_copy['4'].apply(lambda x: '19' in x)]

