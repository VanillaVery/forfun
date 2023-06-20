import pickle 
import geopandas as gpd
from geopandas import GeoDataFrame
import shapely
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd

"""데이터불러오기"""
#gv_info (미완성)
with open('gv_info.pickle', 'rb') as f:
    gv_info = pickle.load(f)

#dong_pop(송파구 한정) 
with open('dong_pop.pickle', 'rb') as f:
    dong_pop = pickle.load(f)

#traffic(송파구 한정)
with open('traffic.pickle', 'rb') as f:
    traffic = pickle.load(f)

hangjeongdong = pd.read_csv('hangjeongdong.csv')
#%%

gv_info = gv_info[gv_info['addr'].str.contains('송파구')]

gv_info = pd.merge(gv_info,hangjeongdong, how='left',on='statId')

gv_info.groupby('dong')['statId'].nunique()

#%%
gv_info.columns
import datetime
gv_info['lastTsdt'] = gv_info['lastTsdt'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d%H%M%S' if x!='' else ''))
gv_info['lastTedt'] = gv_info['lastTedt'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d%H%M%S' if x!='' else ''))
gv_info['nowTsdt'] = gv_info['nowTsdt'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d%H%M%S' if x!='' else ''))

#6월 15일 데이터가 최신 

"""#그룹화 """
gv_info_dong = gv_info.groupby('dong')['lastTsdt'].count()/gv_info.groupby('dong')['statId'].nunique() 
# 데이터에서 1개 충전소당 평균 충전 횟수 
gv_info_dong = pd.DataFrame(gv_info_dong).reset_index().rename(columns={0:'charge_per_stat'})
#%%
dong_pop = dong_pop.iloc[:,:4]
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710510','dong']='풍납1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710520','dong']='풍납2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710531','dong']='거여1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710532','dong']='거여2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710540','dong']='마천1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710550','dong']='마천2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710561','dong']='방이1동'

dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710562','dong']='방이2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710566','dong']='오륜동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710570','dong']='오금동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710580','dong']='송파1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710590','dong']='송파2동'

dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710600','dong']='석촌동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710610','dong']='삼전동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710620','dong']='가락본동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710631','dong']='가락1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710632','dong']='가락2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710641','dong']='문정1동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710642','dong']='문정2동'

dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710650','dong']='잠실본동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710690','dong']='잠실4동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710710','dong']='잠실6동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710720','dong']='잠실7동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710670','dong']='잠실2동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710680','dong']='잠실3동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710646','dong']='장지동'
dong_pop.loc[dong_pop['ADSTRD_CODE_SE']=='11710647','dong']='위례동'

dong_pop['TOT_LVPOP_CO']=dong_pop['TOT_LVPOP_CO'].astype(float)
dong_pop_dong = dong_pop.groupby('dong')['TOT_LVPOP_CO'].mean()
dong_pop_dong = pd.DataFrame(dong_pop_dong).reset_index().rename(columns={0:'average_pop'})
#%%
traffic.loc[traffic['spot_num']=='B-12','dong']='위례동'
traffic.loc[traffic['spot_num']=='C-18','dong']='잠실6동'
traffic.loc[traffic['spot_num']=='C-20','dong']='풍납1동'
traffic.loc[traffic['spot_num']=='D-44','dong']='잠실2동'
traffic.loc[traffic['spot_num']=='D-45','dong']='가락1동'

traffic['vol']=traffic['vol'].astype(float)
traffic_dong = traffic.groupby('dong')['vol'].mean()
traffic_dong = pd.DataFrame(traffic_dong).reset_index().rename(columns={0:'average_traffic'})

# 나머지 동들은 삼각형의 평균으로 치환 

a=pd.merge(gv_info_dong, dong_pop_dong,how='outer',on='dong')
total=pd.merge(a,traffic_dong,how='left',on='dong')

total.loc[total['dong'].isin(['잠실7동','잠실본동','삼전동','석촌동','잠실3동']),'vol']=(308+215+468)/3
total.loc[total['dong'].isin(['풍납2동','잠실4동','방이2동','송파1동','송파2동']),'vol']=(308+456+468)/3
total.loc[total['dong'].isin(['문정2동','가락본동','문정1동','가락2동','장지동']),'vol']=(308+149)/2
total.loc[total['dong'].isin(['거여1동','거여2동','마천1동','마천2동']),'vol']=149
total.loc[total['dong'].isin(['오륜동','오금동','방이1동']),'vol']=(456+149)/2
#%%
total['charge_per_stat'].hist(bins=10) #5 로 나누기로 

total.loc[total['charge_per_stat']>=5,'label']=1
total.loc[total['charge_per_stat']<5,'label']=0

total['label'].value_counts()
#%%
"""표준화 및 모델링"""
#잠실3동, 잠실4동, 잠실7동,잠실본동, 풍납2동 에 추가로 충전소를 지어도 될 것인지?
test=total.loc[total['dong'].isin(['잠실3동', '잠실4동', '잠실7동','잠실본동','풍납2동'])]
train=total.loc[~total['dong'].isin(['잠실3동', '잠실4동', '잠실7동','잠실본동','풍납2동'])]

train[['charge_per_stat','TOT_LVPOP_CO','vol']]=\
    (train[['charge_per_stat','TOT_LVPOP_CO','vol']] - train[['charge_per_stat','TOT_LVPOP_CO','vol']].mean(axis=0))/train[['charge_per_stat','TOT_LVPOP_CO','vol']].std(axis=0)

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(objective='binary', random_state=5)
lgbm.fit(train.iloc[:,1:4],train.iloc[:,4])
y_pred = lgbm.predict(test.iloc[:,1:4])


#%%
"""시각화 """
# 데이터를 좌표 데이터?로 변환

gv_info = gpd.GeoDataFrame(gv_info, 
                           geometry=gpd.points_from_xy(gv_info['lat'], gv_info['lng']))

gv_info.set_crs(epsg = 4326, inplace = True)
geometry = [Point(xy) for xy in zip(gv_info['lng'],gv_info['lat'])]
gdf = GeoDataFrame(gv_info, geometry=geometry, crs= 32610) 
gdf.crs = "EPSG:4326"

gv_info_temp = gv_info.drop_duplicates('statId')
gv_info_temp['lng'] = gv_info_temp['lng'].astype(float)
gv_info_temp['lat'] = gv_info_temp['lat'].astype(float)

gv_info_temp.drop(gv_info_temp[gv_info_temp['lng']==125].index,inplace=True)
gv_info_temp.drop(gv_info_temp[gv_info_temp['lng']<=127].index,inplace=True)

gv_info_temp.to_csv("충전기정보.csv",encoding='utf-8-sig')

#시각화
# plt.scatter(gv_info_temp['lng'],gv_info_temp['lat'],
#             color= "#bfff00",
#             linewidths = 1)

#%%
"""시각화"""
# 송파구 shp 데이터 
songpa_map = gpd.read_file('LSMD_CONT_LDREG_11710.shp')
songpa_map.set_crs(epsg=5186, inplace= True)
songpa_map = songpa_map.to_crs(epsg=4326)
songpa_map.plot()

# 둘을 결합하여 플랏으로 나타내기 
# crs = {'init':'EPSG:4326'}
fig, ax = plt.subplots(figsize = (10,10))
songpa_map.plot(ax=ax, color='lightgrey')
gv_info_temp.plot(ax=ax,aspect=1)
ax.set_title('송파구 내 전기차충전소 위치')
#%%

