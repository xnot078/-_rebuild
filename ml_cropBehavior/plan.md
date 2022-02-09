**目標: 更泛用、精簡的sklearn-style estimator**

>1. **pre-process pipeline: 訓練前先選擇做異常偵測與否，並且對缺失值進行處理。**

>2. **estimator.fit(): 訓練過程。**
  目前不同模型的訓練是分散的func，希望有一個類似interface的class能更飯用的輸入不同的模型、訓練、存模型。
>- 利用cross-validation做訓練: n_splits要能設定
>- 有test method ".socres()": 紀錄precision, recall
>- 有attribute ".pred_Kfold": 預想為pd.DataFrame([預估機率, 真實標籤], index=instance_index)

>3. **estimator.predict(): 預估過程。**
>- 要有predict_proba()
>- 要有fit_predict()
