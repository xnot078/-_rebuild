import pandas as pd
from preprocess import anomaly_detection as ad

def pipe_prepare(df:pd.DataFrame, is_train=True, sample_rate=3, use_anomaly_detect=True, anomaly_drop_ratio=0.05, anomaly_method='cluster', K=10, method_in_cluster='pca', fill_value=0, random_state=2018):
    """
    進入模型前，預處理資料的pipeline。
    輸入一個df，回傳:
    1. 可以訓練模型或預測的hot encoding特徵dataframe，即X。
    2. 訓練或檢視用的hot encoding標籤dataframe，即y。(此選項為選擇性的)
    args:
        df: pd.DataFrame; raw data. 需要有以下欄位(組合):
            -依險別: 近五年  Avg.每車年均投入保費(&保單數)
                iins_sum(count)_mean-name-year: name=險種類別、year=第幾個近五年
            -依車種: 近五年  Avg.每車年均投入保費(&保單數)
                cartype_sum(count)_mean-name-year: name=車種、year=第幾個近五年
            -依業務來源: 近五年  Avg.每車年均投入保費(&保單數)
                iroute_sum(count)_mean-name-year: name=車種、year=第幾個近五年
        is_train: bool; 是否為訓練集。若True，則進行平衡標籤，並建議同時設定use_anomaly_detect=True。
        sample_rate: int; 自然人實例數/法人實例數的比例，用來平衡標籤。
        use_anomaly_detect: bool; 是否排除異常值，建議訓練時使用。預測或測試時不使用。
        fill_value: number; 遇na值得填補常數。
        **kwargs請參考preprocess.anomaly_detection的各函式
    return:
        X: pd.DataFrame; (index, columns)=(被保人代號, 特徵)
        y: pd.DataFrame; (index, columns)=(被保人代號, 標籤)
        *** 如果df中沒有'fassured'，視為預測用資料，不會回傳y ***
    """
    df = df.filter(regex='^(?!.*others|.*其他|.*max|.*carQt|.*outlier).*') # 負向表列用不到的欄位

    mask_recent = filter_recent(df, by='1')
    df_recent = df[mask_recent].copy()

    if is_train and 'fassured' in df.columns:
        # for training set, 先排除異常。可以回傳X_train & y_train
        df_recent['fassured'] = df_recent['fassured'].apply(lambda x: 1 if x=='2' else 0) # 雖然是在處理df，但是為了用抽樣來平衡標籤比例，先保留標籤值，抽樣玩再丟掉
        if use_anomaly_detect:
            # 排除極端值: 如果有標籤，不同標籤應該分開排除
            data = []
            for k, g in df_recent.groupby('fassured'):
                mask_anomaly = filter_not_anomaly(g, anomaly_method=anomaly_method, method_in_cluster=method_in_cluster, K=K, drop_ratio=anomaly_drop_ratio)
                tem = g[mask_anomaly]
                data.append(tem)
            df_recent = pd.concat(data)
        # 平衡標籤
        df_res = balanced_sample(df_recent, sample_rate=sample_rate, random_state=random_state)
        X_ans, y_ans = df_res.drop('fassured', axis=1), df_res['fassured']
        X_ans.fillna(0, inplace=True)
        return X_ans, y_ans
    elif 'fassured' in df.columns:
        # for testing set (此時仍必須有標籤)，可以回傳X_test & y_test
        df_recent['fassured'] = df_recent['fassured'].apply(lambda x: 1 if x=='2' else 0) # 雖然是在處理df，但是為了用抽樣來平衡標籤比例，先保留標籤值，抽樣玩再丟掉
        X_ans, y_ans = df_recent.drop('fassured', axis=1), df_recent['fassured']
        X_ans.fillna(0, inplace=True)
        return X_ans, y_ans
    else:
        # for prediction, (預測時不會有標籤)，只會回傳 X_for_predict
        X_ans = df_recent
        X_ans.fillna(0, inplace=True)
        return X_ans

def filter_recent(df:pd.DataFrame, by='-1'):
    mask = ~df.filter(regex=f'.*{by}').isna().all(axis=1)
    return mask

def filter_not_anomaly(df:pd.DataFrame, anomaly_method='all', method_in_cluster='pca', K=10, drop_ratio=0.05):
    """排除極端值。
    args:
        anomaly_method: str;
            'all'=只對全部實例做一次異常偵測
            'cluster'=將實例利用kmeans分成K簇，逐簇做異常偵測。
        method_in_cluster: str; 進行異常偵測的方法。
            'pca'=以PCA重構誤差法做排除
            'lof'=以LOF法做排除
        K: anomaly_method='cluster'時才有用。
    """
    def choose_method_in_cluster(df, method_in_cluster='pca', drop_ratio=0.05):
        """當anomaly_method='all'時，判斷要用哪一種方法(method_in_cluster)進行異常偵測的if else"""
        if method_in_cluster == 'pca':
            mask = ad.anomaly_detection_PCA(df, drop_ratio=drop_ratio)
        elif method_in_cluster == 'lof':
            mask = ad.anomaly_detection_LOF(df, drop_ratio=drop_ratio)
        else:
            print('neither "method_in_cluster" is "pca" or "lof", use "pca".')
            mask = ad.anomaly_detection_PCA(df, drop_ratio=drop_ratio)
        return mask

    if anomaly_method == 'cluster':
        mask = ad.anomaly_detection_kmeans(df, K=K, drop_ratio=drop_ratio, method_in_cluster=method_in_cluster)
    elif anomaly_method == 'all':
        mask = choose_method_in_cluster(df, method_in_cluster=method_in_cluster, drop_ratio=0.05)
    else:
        print('neither "anomaly_method" is "cluster" or "all", use "all".')
        mask = choose_method_in_cluster(df, method_in_cluster=method_in_cluster, drop_ratio=0.05)
    return mask


def balanced_sample(df:pd.DataFrame, by='fassured', base_on=1, sample_rate=3, random_state=2018):
    """
    目標是平衡標籤。以法人實例數 N為基準，抽樣自然人實例共M = sample_rate*N，然後合併。當自然人實例數<=M時，自然人實例全部取用。
    """
    numPositive = sum(df[by]==base_on)
    numNatSamples = min(len(df[by]!=base_on), numPositive*sample_rate)
    df_Nat = df[df[by]!=base_on]
    df_Nat_samples = df.sample(numNatSamples, random_state=random_state)
    df_Law = df[df[by]==base_on]
    df_con = pd.concat([df_Nat_samples, df_Law])
    df_con = df_con.sample(frac=1, random_state=random_state) # 打亂順序
    return df_con
