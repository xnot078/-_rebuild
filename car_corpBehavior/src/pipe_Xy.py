import pandas as pd
from car_corpBehavior.preprocess import anomaly_detection as ad
from sklearn import cluster
from sklearn import base


class pipe_preprocess(base.BaseEstimator, base.TransformerMixin):
    """
    .fit(X, y=None, **kwargs):
        篩選使用欄位 -> 存到attribute: feature_name_
    .transform(X, y=None, use_clustering=False, **kwargs):
        篩選使用欄位(from feature_name_) -> 去除完全沒資料的實例 -> fill_na -> anomaly_detection
    """
    def __init__(self):
        self.feature_name_ = None
        pass
    def fit(self, X, y=None, **kwargs):
        """
        filter_columns to filter features to use.
        args:
            x: pd.DataFrame; raw data. 需要有以下欄位(組合):
                -依險別: 近五年  Avg.每車年均投入保費(&保單數)
                    iins_sum(count)_mean-name-year: name=險種類別、year=第幾個近五年
                -依車種: 近五年  Avg.每車年均投入保費(&保單數)
                    cartype_sum(count)_mean-name-year: name=車種、year=第幾個近五年
                -依業務來源: 近五年  Avg.每車年均投入保費(&保單數)
                    iroute_sum(count)_mean-name-year: name=車種、year=第幾個近五年
            not_use: list of str; 要排除的regex substring，欄位名稱有regex=(?!.*substring)的就排除
            use: list of str; 要使用的regex substring，欄位名稱有regex=(?=.*substring)的就使用
            labels: str; 標籤欄的名稱
        X =
        """
        mask_cols = filter_columns(X, **kwargs)
        self.feature_name_ = mask_cols[mask_cols].index
        return self

    def transform(self, X, y=None, use_clustering=False, **kwargs):
        """
        args:
            X: pd.DataFrame; 輸入特徵
            y: pd.Series; optional，instances的標籤。如果有提供，將把X依所屬標籤分群進行異常偵測。
            anomaly_method: str;
                'pca': 以PCA進行異常偵測。
                'lof': 以lof進行異常偵測。
            use_clustering: bool; 是否先把X分群後再逐簇進行異常偵測
            cluster_method: sklearn.cluster的分群演算法，必須有fit_predict方法。預設使用KMeans(n_clusters=10)。
        """
        try:
            X_ = X[self.feature_name_]
        except KeyError as e:
            raise Exception('Check columns of training data used in .fit() and columns of input data X.\n Or simply use fit_transform carefully.') from e

        # mask_recent = filter_recent(X, **kwargs)
        # X_ = X_[mask_recent] # 把缺少的必要欄位的instances丟掉
        # X_.fillna(0, inplace=True)

        if y is not None and len(X) == len(y):
            # y_ = y[mask_recent]
            y_ = y
            # if use_clustering and len(X) > 10:
            mask_anomaly = filter_not_anomaly(X_, y, use_clustering=use_clustering, **kwargs)
            X_ = X_[mask_anomaly]
            y_ = y_[mask_anomaly]
            Xy_ = pd.concat([X_, y_], axis=1)
            Xy_ = balanced_sample(Xy_, **kwargs)
            X_ = Xy_.drop(y_.name, axis=1)
            return X_
        else:
            return X_

def filter_columns(df:pd.DataFrame, not_use=['others', '其他', '-0'], use=['cartype.*mean', 'iins.*mean', 'iroute.*mean'], label='fassured', **kwargs):
    not_use = kwargs.get('not_use', not_use)
    use = kwargs.get('use', use)
    label = kwargs.get('label', label)
    """
    透過regex篩選要用的欄位("use")和要排除的欄位("not_use")
    args:
        not_use: list of str; 要排除的regex substring，欄位名稱有regex=(?!.*substring)的就排除
        use: list of str; 要使用的regex substring，欄位名稱有regex=(?=.*substring)的就使用
    return:
        pd.Series of bool.
    """
    # not_use = ['others', '其他', 'max', 'carQt', 'outlier']
    mat_str =  '^(?=.*' + '|.*'.join(use) + '.*|' + label + ')'
    if isinstance(not_use, list) and len(not_use)>0 and all(isinstance(i, str) for i in not_use):
        mat_str += '(?!.*' + '|.*'.join(not_use) + ').*'
    mask_cols = pd.Series(df.columns, index=df.columns).str.match(mat_str)
    return mask_cols

def filter_recent(df:pd.DataFrame, by='-1', **kwargs):
    by = kwargs.get('recent_by', by)
    mask = ~df.filter(regex=f'.*{by}').isna().all(axis=1)
    return mask

def filter_not_anomaly(X:pd.DataFrame, y:pd.Series=None, use_clustering=False, anomaly_method='pca', **kwargs):
    """排除極端值。
    args:
        X: pd.DataFrame; 輸入特徵
        y: pd.Series; optional，instances的標籤。如果有提供，將把X依所屬標籤分群進行異常偵測。
        anomaly_method: str;
            'pca': 以PCA進行異常偵測。
            'lof': 以lof進行異常偵測。
        use_clustering: bool; 是否先把X分群後再逐簇進行異常偵測
        cluster_method: sklearn.cluster的分群演算法，必須有fit_predict方法。預設使用KMeans(n_clusters=10)。
    """
    methods = {'pca': ad.anomaly_detection_PCA,
               'lof': ad.anomaly_detection_LOF
                }
    used_method = methods.get(anomaly_method)

    if anomaly_method not in methods.keys():
        print('anomaly_method must be one of  ["pca", "lof"].')
        print('use default method: "pca".')
        anomaly_method = 'pca'
    if isinstance(y, pd.Series) and len(y) == len(X):
        Xy = pd.concat([X, y], axis=1)
        mask = pd.Series(False, index=X.index)
        for k, g in Xy.groupby(y.name):
            if len(g) < 10:
                mask.loc[g.index] = True
            else:
                mask_g = used_method(g, use_clustering=use_clustering, **kwargs)
                mask.loc[g.index] = mask_g
    else:
        mask = used_method(X, use_clustering=use_clustering, **kwargs)
    return mask

def balanced_sample(df:pd.DataFrame, label='fassured', base_on=1, sample_rate=3, random_state=2018, **kwargs):
    label = kwargs.get('balanced_label', label)
    base_on = kwargs.get('balanced_base_on', base_on)
    sample_rate = kwargs.get('balanced_sample_rate', sample_rate)

    """
    目標是平衡標籤。
    依label欄位和base_on來辨別positive分類。以positive的數量為基礎，抽樣positive*sample_rate的negative實例合併之。
    當negative實例數<=positive*sample_rate時，自然人實例全部取用。
    """
    numPositive = sum(df[label]==base_on)
    print(sum(df[label]!=base_on), numPositive, sample_rate)
    numNatSamples = min(sum(df[label]!=base_on), numPositive*sample_rate)
    df_Nat = df[df[label]!=base_on]
    df_Nat_samples = df.sample(numNatSamples, random_state=random_state)
    df_Law = df[df[label]==base_on]
    df_con = pd.concat([df_Nat_samples, df_Law])
    df_con = df_con.sample(frac=1, random_state=random_state) # 打亂順序
    return df_con

if __name__ == '__main__':
    raw = pd.read_parquet('./car_corpBehavior/data/medium_pivot/input_data.parq')
    # mask_cols = filter_columns(raw)
    # raw = raw[mask_cols[mask_cols].index]
    # mask_recent = filter_recent(raw)
    # raw = raw[mask_recent]
    # raw.reset_index(inplace=True)
    X = raw.drop('fassured', axis=1)
    y = raw['fassured']

    p = pipe_preprocess()
    p.fit(X)
    X = p.transform(X, y, use_clustering=True, balanced_label='fassured', balanced_base_on='2')
    X
