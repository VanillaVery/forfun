from sklearn.preprocessing import StandardScaler  # 표준화 패키지 라이브러리 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

user_anal = data.groupby('serviceId').agg({'timeSpent':'median','numberOfDays':'median','userId':'count'})
user_anal.reset_index(inplace=True)

x=user_anal.iloc[:,1:]
y=user_anal.iloc[:,1]

x = StandardScaler().fit_transform(x) # x객체에 x를 표준화한 데이터를 저장

features = ['timeSpent','numberOfDays','userId']
pd.DataFrame(x, columns=features).head()

#%%
pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

pca.explained_variance_ratio_
#주성분 1개로도 분산을 84% 설명할 수 있다 

data_principal = pd.concat([user_anal,principalDf],axis=1)
#%%
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize=20)

targets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
colors = ['black','silver','rosybrown','firebrick','red',
          'lightsalmon','saddlebrown','darkorange','tan','orange',
          'gold','darkkhaki','olive','olivedrab','greenyellow',
          'limegreen','seagreen','aquamarine','darkslategray','aqua',
          'royalblue','purple']

for target, color in zip(targets,colors):
    indicesToKeep = data_principal['serviceId'] == target
    ax.scatter(data_principal.loc[indicesToKeep, 'principal component1']
               , data_principal.loc[indicesToKeep, 'principal component2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#%%
abs(pca.components_)

