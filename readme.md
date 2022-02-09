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

raw = pd.read_parquet('./car_corpBehavior/data/medium_pivot/ptByPerson_險別車種分開_v5.parq') #範例資料請另外要求
X, y = raw.drop('fassured', axis=1), raw['fassured']

with open('./trained_model/trained_cpl.pickle', 'rb') as f:
      trained_cpl = pickle.load(f)
y_pred_proba = trained_cpl.predict_proba(X)
```

2. 訓練模型:
```python
import pandas as pd
import pickle
from car_corpBehavior.src import model

#pos_label: positive label, '2' means Crops and '1' means Natures in this case.
"""
Using default model:
cpl = model.car_potential_legal(pos_label='2', neg_label='1')
"""
cpl =  model.car_potential_legal(model=used_model, pos_label='2', neg_label='1') # init_model
cpl.fit(X_train, y_train, K=10) # fit

cpl.model.feature_name_ #check feature_name_

#show scores
for k, v in cpl.scores.items():
    print(k, end='')
    if isinstance(v, dict):
        print()
        for thresh, score in v.items():
            print(f'\t{thresh=}: {score:.1%}')
    else:
        print(f': {v:.1%}')

#Save trained model.
with open('./car_corpBehavior/trained_model/trained_cpl.pickle', 'wb') as f:
    pickle.dump(cpl, f)


```
