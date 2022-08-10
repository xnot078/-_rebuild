import pandas as pd
import numpy as np
import tqdm

"""alg."""
from sklearn import decomposition, neighbors
from sklearn import cluster

import matplotlib.pyplot as plt

# X = df_recent.iloc[:, 1:]
def anomaly_detection_PCA(X:pd.DataFrame, exp_var_ratio=0.8, *args, **kwargs):
    drop_ratio=kwargs.get('drop_ratio', 0.05)
    print(drop_ratio)
    """
    以PCA的重構誤差法排除離群值。

    args:
        X: pd.DataFrame; shape=(實例, 特徵)的特徵dataframe
        exp_var_ratio = 重構誤差時，要用多少的n_components
        drop_ratio = 要篩去多少比例的高重構誤差樣本
    return:
        np.array, 篩去離群值後的X'
    """
    mask = pd.Series(True, index=X.index)
    if len(X) < 1:
        return mask # 輸入實例數=0，根本不用做

    if isinstance(X, pd.DataFrame):
        X.fillna(0, inplace=True)
        X_ft = X
    if 'fassured' in X.columns:
        X_ft = X_ft.drop('fassured', axis=1)

        # X_ft = X_ft.loc[X['fassured']==0]

    # 先確認要用多少個n_components能解釋到exp_var_ratio比例的變異
    test_n_components = min(X_ft.shape)
    pca = decomposition.PCA(n_components=test_n_components)
    pca.fit(X_ft)
    expVar = pca.explained_variance_ratio_
    n_components = np.argmax(expVar.cumsum()>=exp_var_ratio)
    n_components = max(n_components, 1) # 預防n_components=0的情況
    # 建立pca並找異常
    pca = decomposition.PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_ft)
    X_pca_inverse = pca.inverse_transform(X_pca)
    loss = np.sum( (X_pca_inverse - X_ft.values)**2, axis=1 )
    loss = (loss - loss.min()) / (loss.max() - loss.min())
    loss = pd.Series(loss, index=X_ft.index)
    thresh = np.quantile(loss, 1-drop_ratio)
    mask.loc[loss.index] = loss <= thresh

    print(f'({exp_var_ratio=:.1%})n_components: {test_n_components}->{n_components}\n({drop_ratio=:.1%})異常數: {sum(mask):,.0f}({sum(mask)/len(mask):.1%})')
    return mask

def anomaly_detection_LOF(X, n_neighbors=10, drop_ratio=0.2, *args, **kwargs):
    """
    警告：雖然比較準，但實例一多就要跑非常久。
    以LOF法排除離群值。
    args:
        X: pd.DataFrame; shape=(實例, 特徵)的特徵dataframe
        n_neighbors: LOF參數，控制用來檢視密度的鄰居數量
        drop_ratio = 要篩去多少比例的低LOF分數實例(低密度分數)
    return:
        np.array, 篩去離群值後的X'
    """
    mask = pd.Series(True, index=X.index)

    # 如果樣本過少，直接跳出
    if len(X) < n_neighbors:
        return mask

    if isinstance(X, pd.DataFrame):
        X.fillna(0, inplace=True)
        X_ft = X
    if 'fassured' in X.columns:
        X_ft = X_ft.drop('fassured', axis=1)
        # X_ft = X_ft.loc[X['fassured']==0]

    lof = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors)
    label = lof.fit_predict(X_ft)
    factor = -lof.negative_outlier_factor_
    factor = pd.Series(factor, index=X_ft.index)
    thresh = np.quantile(factor, 1-drop_ratio)
    mask.loc[factor.index] = factor <= thresh

    return mask

def anomaly_detection_PCA2(X:pd.DataFrame, use_clustering=False, exp_var_ratio=0.8, drop_ratio=0.05, **kwargs):
    if not use_clustering:
        mask = pd.Series(True, index=X.index)
        if len(X) < 1:
            return mask # 輸入實例數=0，根本不用做

        if isinstance(X, pd.DataFrame):
            X.fillna(0, inplace=True)
            X_ft = X
        if 'fassured' in X.columns:
            X_ft = X_ft.drop('fassured', axis=1)

            # X_ft = X_ft.loc[X['fassured']==0]

        # 先確認要用多少個n_components能解釋到exp_var_ratio比例的變異
        test_n_components = min(X_ft.shape)
        pca = decomposition.PCA(n_components=test_n_components)
        pca.fit(X_ft)
        expVar = pca.explained_variance_ratio_
        n_components = np.argmax(expVar.cumsum()>=exp_var_ratio)
        n_components = max(n_components, 1) # 預防n_components=0的情況
        # 建立pca並找異常
        pca = decomposition.PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_ft)
        X_pca_inverse = pca.inverse_transform(X_pca)
        loss = np.sum( (X_pca_inverse - X_ft.values)**2, axis=1 )
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        loss = pd.Series(loss, index=X_ft.index)
        thresh = np.quantile(loss, 1-drop_ratio)
        thresh = 1e-3 if thresh==0 else thresh # 避免thresh=0的情況
        mask.loc[loss.index] = loss <= thresh

        print(f'({exp_var_ratio=:.1%})n_components: {test_n_components}->{n_components}\n({drop_ratio=:.1%})異常數: {sum(mask):,.0f}({sum(mask)/len(mask):.1%})')
        return mask
    else:
        method = kwargs.get('method', cluster.KMeans(n_clusters=10))
        cluster_labels = clustering(X, method=method)
        mask = anomaly_detection_in_clusters(X, cluster_labels, anomaly_detection_PCA2, drop_ratio=drop_ratio)
        return mask

m = anomaly_detection_PCA2(X.iloc[:1000], use_clustering=True, drop_ratio=0.2)



def anomaly_detection_PCA2_0(X:pd.DataFrame, use_clustering=False, exp_var_ratio=0.8, drop_ratio=0.05, **kwargs):
    """
    以PCA的重構誤差法排除離群值。

    args:
        X: pd.DataFrame; shape=(實例, 特徵)的特徵dataframe
        exp_var_ratio = 重構誤差時，要用多少的n_components
        drop_ratio = 要篩去多少比例的高重構誤差樣本
    return:
        np.array, 篩去離群值後的X'
    """
    mask = pd.Series(True, index=X.index)
    if len(X) < 1:
        return mask # 輸入實例數=0，根本不用做

    if isinstance(X, pd.DataFrame):
        X.fillna(0, inplace=True)
        X_ft = X
    if 'fassured' in X.columns:
        X_ft = X_ft.drop('fassured', axis=1)

        # X_ft = X_ft.loc[X['fassured']==0]

    # 先確認要用多少個n_components能解釋到exp_var_ratio比例的變異
    test_n_components = min(X_ft.shape)
    pca = decomposition.PCA(n_components=test_n_components)
    pca.fit(X_ft)
    expVar = pca.explained_variance_ratio_
    n_components = np.argmax(expVar.cumsum()>=exp_var_ratio)
    # 建立pca並找異常
    pca = decomposition.PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_ft)
    X_pca_inverse = pca.inverse_transform(X_pca)
    loss = np.sum( (X_pca_inverse - X_ft.values)**2, axis=1 )
    loss = (loss - loss.min()) / (loss.max() - loss.min())
    loss = pd.Series(loss, index=X_ft.index)
    thresh = np.quantile(loss, 1-drop_ratio)
    mask.loc[loss.index] = loss <= thresh

    print(f'({exp_var_ratio=:.1%})n_components: {test_n_components}->{n_components}\n({drop_ratio=:.1%})異常數: {sum(mask):,.0f}({sum(mask)/len(mask):.1%})')

    if use_clustering==False:
        return mask
    else:
        method = kwargs.get('method', cluster.KMeans(n_clusters=10))
        cluster_labels = clustering(X, method=method)
        mask = anomaly_detection_in_clusters(X, cluster_labels, anomaly_detection_PCA2, drop_ratio=0.05)
        return mask
# method = cluster.KMeans(n_clusters=10)

def clustering(X:pd.DataFrame, method=cluster.KMeans(n_clusters=10)):
    """
    輸入資料 & 分群的方法(預設為KMeans)，回傳X的labels
    """
    if isinstance(X, pd.DataFrame):
        X.fillna(0, inplace=True)
        X_ft = X
    if 'fassured' in X.columns:
        X_ft = X.drop('fassured', axis=1)
    print(f'input size: {X_ft.shape}')
    print(f'clustering method: {method}')
    labels = method.fit_predict(X)
    labels = pd.Series(labels, index=X.index, name='label')
    return labels

def anomaly_detection_in_clusters(X:pd.DataFrame, labels:pd.Series, anomaly_method=anomaly_detection_PCA, drop_ratio=0.05, **kwargs):
    """
    輸入實例集 & 對應的標籤，依labels做groupby執行指定的異常偵測，最後整合成一個pd.Series(Bool)。
    """
    Xy = pd.concat([X, labels], axis=1)
    mask_all = pd.Series(False, index=X.index)
    for l, x_l in Xy.groupby(labels.name):
        print(f'label: {l}')
        if len(x_l) > 0:
            mask_l = anomaly_method(x_l, drop_ratio=drop_ratio, **kwargs)
            mask_all.loc[x_l.index] = mask_l
        else:
            print('len(x_l)=0')
    return mask_all



def anomaly_detection_kmeans(X:pd.DataFrame, K=10, drop_ratio=0.2, method_in_cluster='dist'):
    """
    先將實例分群，將每個簇中的離群值排除(排除方法可選用)。
    可用更泛用的clustering() + anomaly_detection_in_clusters()取代。

    args:
        X: pd.DataFrame; shape=(實例, 特徵)的特徵dataframe
        K: 重構誤差時，要用多少的n_components
        method_in_cluster:
            'dist'=以離簇心的距離做排除
            'pca'=以PCA重構誤差法做排除
            'lof'=以LOF法做排除
        drop_ratio = 要篩去多少比例的遠離各cluster中心距離的樣本
    return:
        np.array, 篩去離群值後的X'
    """
    mask = pd.Series(True, index=X.index)

    if isinstance(X, pd.DataFrame):
        X.fillna(0, inplace=True)
    if 'fassured' in X.columns:
        X_ft = X.drop('fassured', axis=1)
        # X_ft = X_ft.loc[X['fassured']==0]

    print('clustering...')
    kmeans = cluster.KMeans(n_clusters=K, n_jobs=-1)
    kmeans.fit(X_ft)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # mask = np.ones(len(labels), dtype=bool)
    # method_in_cluster = 'pca'
    print('find outlier for each cluster...')
    for i, center in tqdm.tqdm(enumerate(centers)):
        idxs = np.where(labels==i)[0]
        X_cluster = X.iloc[idxs]
        X_cluster_ft = X_cluster.drop('fassured', axis=1)
        if method_in_cluster == 'dist':
            dist = np.sqrt(np.sum((X_cluster_ft.values - center)**2, axis=1))
            stdDist = (dist - dist.min()) / (dist.max()-dist.min())
            stdDist = pd.Series(stdDist, index=X_cluster_ft.index)
            thresh = np.quantile(stdDist, 1-drop_ratio)
            mask_i = stdDist <= thresh
        elif method_in_cluster == 'pca':
            mask_i = anomaly_detection_PCA(X_cluster, drop_ratio=drop_ratio)
        elif method_in_cluster == 'lof':
            mask_i = anomaly_detection_LOF(X_cluster, drop_ratio=drop_ratio)
        else:
            print('"method_in_cluster" : "dist", "pca", "lof". ')
        mask.loc[mask_i.index] = mask_i
    sum(mask) / len(mask)
    return mask
#
# raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq')
#
# X = raw.drop('fassured', axis=1)#.iloc[:, :10]
# y = raw['fassured']

# labels = clustering(X.iloc[:, :50], method=cluster.KMeans(n_clusters=100))
# labels.value_counts()


# m = anomaly_detection_in_clusters(X.iloc[:, :50], labels, anomaly_detection_PCA, drop_ratio=0.05)
# sum(m) / len(m)
#
# @anomaly_detection_in_clusters_decorator(X.iloc[:, :50], labels)
# ad_PCA = lambda X: anomaly_detection_PCA(X, drop_ratio=0.05)
# anomaly_detection_in_clusters_decorator(X.iloc[:50], labels, anomaly_detec_in_cluster_PCA)
#
# def anomaly_detec_in_cluster_PCA(*args, **kwargs):
#     return anomaly_detection_PCA(X.iloc[:, :50], drop_ratio=0.05)
#
# anomaly_detec_in_cluster_PCA(X.iloc[:, :50])
