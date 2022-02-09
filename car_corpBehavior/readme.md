###摘要

**Problem: 預測潛在車險法人。**
**Rationale: 從過去的法人行為學習，找出身分是自然人但是行為舉止非常接近法人的「被保險人」。**

******
###輸入:
>**三個面向的 "近五年  Avg.每車年均投入保費(&保單數)"**

**1. 依險別:** iins_sum/count_mean-name-year
> **iins** 表示"依險別"
> **sum/count** 前者表示保費，後者表示保單數
> **mean** 紀錄這是平均
> **name** 表示險別名稱
> (傷害險、機車相關、竊盜險、責任險、車體險)
> **year** 1表示近五年，0表示其他'


**2. 依車種:** cartype_sum/count_mean-name-year
> **cartype** 表示"依車種"
> **name** 表示車種名稱
> (大客車、大貨車、客貨兩用車、小客車、小貨車、機車)

**3. 依業務來源:** iroute_sum/count_mean-name-year
> **iroute** 表示"依業務來源"
> **name** 表示業務來源名稱 e.g.
> (大客車、大貨車、客貨兩用車、小客車、小貨車、機車)

註1：請以pandas.DataFrame格式輸入.
註2: 只取近五年(year=1)，將有40個欄位 (sum:20個欄位+count:20個欄位)

******

###使用

1. 使用預先訓練的模型:
```python
import pandas as pd
import pickle

raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq') #範例資料請另外要求
X, y = raw.drop('fassured', axis=1), raw['fassured']

with open('./trained_model/trained_cpl.pickle', 'rb') as f:
      trained_cpl = pickle.load(f)
y_pred_proba = trained_cpl.predict_proba(X)
```

2. 訓練模型:
```python
import pandas as pd
import pickle
from car_corpBehavior.src


```
3.




>1. **pre-process pipeline: 訓練前先選擇做異常偵測與否，並且對缺失值進行處理。**

>2. **estimator.fit(): 訓練過程。**
  目前不同模型的訓練是分散的func，希望有一個類似interface的class能更飯用的輸入不同的模型、訓練、存模型。
>- 利用cross-validation做訓練: n_splits要能設定
>- 有test method ".socres()": 紀錄precision, recall
>- 有attribute ".pred_Kfold": 預想為pd.DataFrame([預估機率, 真實標籤], index=instance_index)

>3. **estimator.predict(): 預估過程。**
>- 要有predict_proba()
>- 要有fit_predict()
