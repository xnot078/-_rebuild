import pandas as pd
import numpy as np
import tqdm

"""alg."""
from sklearn import decomposition, neighbors
from sklearn import cluster


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

def anomaly_detection_in_clusters(X:pd.DataFrame, labels:pd.Series, anomaly_method=None, drop_ratio=0.05, **kwargs):
    if anomaly_method is None:
        anomaly_method = anomaly_detection_PCA
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

def anomaly_detection_LOF(X, use_clustering=False, n_neighbors=10, drop_ratio=0.2, *args, **kwargs):
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
    if not use_clustering:
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
        print(f'({drop_ratio=:.1%})異常數: {sum(~mask):,.0f}({sum(~mask)/len(mask):.1%})')
        return mask
    else:
        method = kwargs.get('cluster_method', cluster.KMeans(n_clusters=10))
        cluster_labels = clustering(X, method=method)
        mask = anomaly_detection_in_clusters(X, cluster_labels, anomaly_detection_LOF, drop_ratio=drop_ratio)
        return mask

def anomaly_detection_PCA(X:pd.DataFrame, use_clustering=False, exp_var_ratio=0.8, drop_ratio=0.05, **kwargs):

    def n_components_for_rebuild(X):
        """要多少n_compoents就足夠解釋exp_var_ratio % 的變異，用在重構時的PCA"""
        test_n_components = min(X.shape)
        pca = decomposition.PCA(n_components=test_n_components)
        pca.fit(X)
        expVar = pca.explained_variance_ratio_
        n_components = np.argmax(expVar.cumsum()>=exp_var_ratio)
        n_components = max(n_components, 1) # 預防n_components=0的情況
        print(f'({exp_var_ratio=:.1%})n_components: {test_n_components}->{n_components}', end=' ')
        return n_components

    def rebuild_loss(X, n_components):
        pca = decomposition.PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        X_pca_inverse = pca.inverse_transform(X_pca)
        loss = np.sum( (X_pca_inverse - X.values)**2, axis=1 )
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        loss = pd.Series(loss, index=X.index)
        return loss

    if not use_clustering:
        mask = pd.Series(True, index=X.index)
        if len(X) < 1:
            return mask # 輸入實例數=0，根本不用做

        if isinstance(X, pd.DataFrame):
            X.fillna(0, inplace=True)
            X_ft = X
        if 'fassured' in X.columns:
            X_ft = X_ft.drop('fassured', axis=1)

        # 先確認要用多少個n_components能解釋到exp_var_ratio比例的變異
        n_components = n_components_for_rebuild(X_ft)
        # 建立pca並找異常
        loss = rebuild_loss(X_ft, n_components)
        loss = pd.Series(loss, index=X_ft.index)
        # 篩選重構誤差在標準以下的
        thresh = np.quantile(loss, 1-drop_ratio)
        thresh = 1e-3 if thresh==0 else thresh # 避免thresh=0的情況
        mask.loc[loss.index] = loss <= thresh

        print(f'\n({drop_ratio=:.1%})異常數: {sum(~mask):,.0f}({sum(~mask)/len(mask):.1%})')
        return mask
    else:
        method = kwargs.get('cluster_method', cluster.KMeans(n_clusters=10))
        cluster_labels = clustering(X, method=method)
        mask = anomaly_detection_in_clusters(X, cluster_labels, anomaly_detection_PCA, drop_ratio=drop_ratio)
        return mask

if __name__ == '__main__':
    raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq')

    X = raw.drop('fassured', axis=1)#.iloc[:, :10]
    y = raw['fassured']

    m = anomaly_detection_PCA(X, use_clustering=False, drop_ratio=0.05)
