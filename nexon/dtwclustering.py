"""시계열 클러스터링"""
# 서비스를 클러스터링
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv(r"C:\Users\윤유진\Downloads\[넥슨] 플랫폼분석팀_과제\serviceUsageMonthly.csv")
seed=0
user_anal=pd.pivot_table(data,
                index='serviceId',
               columns='month',
               values='userId',
               aggfunc='count').fillna(0)


np.random.seed(seed)
X_train = user_anal 
# X_train = X_train[y_train < 4]  # Keep first 3 classes
# np.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
# Make time series shorter
X_train = TimeSeriesResampler(sz=12).fit_transform(X_train)
sz = X_train.shape[1]

dt = np.reshape(X_train,(22,12))
pd.DataFrame().to_csv('dtw_cluster.csv')

#%%
"""트렌드 필터링"""
#  복잡한 노이즈를 없애기 위해 쓰는 거고 트렌드가 단순할 경우 쓸 필요 없음

import cvxpy 
import scipy
import cvxopt 
import matplotlib.pyplot as plt

arr = np.empty((0,12),float)

i=0
for i in range(22):
    y = dt[i]
    n = y.size
    lambda_value = 2
    ones_row = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
    solver = cvxpy.CVXOPT
    reg_norm = 2

    x = cvxpy.Variable(shape=n) 
    # x is the filtered trend that we initialize
    objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
                + lambda_value * cvxpy.norm(D@x, reg_norm))
    # Note: D@x is syntax for matrix multiplication
    problem = cvxpy.Problem(objective)
    problem.solve(solver=solver, verbose=False)

    arr = np.append(arr,x.value[None,:],axis=0)

for i in range(12):
    pd.DataFrame(arr[i]).plot()

pd.DataFrame(arr).transpose().plot()

#%%
"""# Soft-DTW-k-means"""

print("DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=7,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=0)
y_pred = sdtw_km.fit_predict(X_train)

for yi in range(7):
    plt.figure(figsize=(10,30))
    plt.subplot(7, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DTW $k$-means")

plt.tight_layout()
plt.show()


sdtw_km.labels_
sdtw_km.cluster_centers_[0].ravel()
sdtw_km.cluster_centers_[1].ravel()
sdtw_km.cluster_centers_[2].ravel()



#%%
