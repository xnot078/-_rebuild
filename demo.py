import pandas as pd
import pickle
from lightgbm import LGBMClassifier # not necessary, just for example
from sklearn import model_selection

from car_corpBehavior.src import preprocess_phase
from car_corpBehavior.src import model
from car_corpBehavior.src import pipe_Xy
#
# with open('./car_corpBehavior/data/inp.pickle', 'wb') as f:
#     pickle.dump([X_train, y_train], f)
# X_train, y_train

if __name__ == '__main__':

    """
    update input_data
    """
    raw = preprocess_phase.update_data(return_data=True)

    """
    input data. (type of X, y must be pandas.core.DataFrame and pandas.core.Series.)
    """
    raw = pd.read_parquet('./car_corpBehavior/data/medium_pivot/input_data.parq')
    # raw.reset_index(inplace=True)
    X, y = raw.drop('fassured', axis=1), raw['fassured']

    """
    1. X preprocess:
        - filter used columns.
        - filter instances are not all NA.
        - (if y is given) anomaly detection.
    2. (if y is given) sampling balanced negative instances and positive instances.
       (if y is not given) do test_train_split() after X preprocess and y preprocess. Be carfully, positive and negative instances are usually unbalanced.
    """
    p = pipe_Xy.pipe_preprocess()
    p.fit(X)
    X_train = p.transform(X, y, use_clustering=True, balanced_label='fassured', balanced_base_on=1)
    """
    y preprocess:
        - After X preprocess and balanced-sampling, it's necessary to index y by X.index.
    """
    y_train = y.loc[X_train.index]
    y_train.value_counts()


    """
    if need use customized model, initialize it (See following example).
    ** .fit(), .predict_proba() methods are necessary for customized model. **
    """
    used_model = LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                max_depth=4, num_leaves=20,
                bagging_seed=2018,
                verbose=0, n_jobs=-1,
                random_state=2018
                )

    """
    Train model.
    pos_label: positive label, '2' means Crops and '1' means Natures in this case.
    """
    cpl = model.car_potential_legal(model=used_model, pos_label=1, neg_label=0) # init_model
    """
        Using default model:
        cpl = model.car_potential_legal(pos_label='2', neg_label='1')
    """
    cpl.fit(X_train, y_train, K=10) # fit
    cpl.model.feature_name_ #check feature_name_
    for k, v in cpl.scores.items(): #show scores
        print(k, end='')
        if isinstance(v, dict):
            print()
            for thresh, score in v.items():
                print(f'\t{thresh=}: {score:.1%}')
        else:
            print(f': {v:.1%}')

    """
    Save trained preprocess pipeline(feature_name_ particularly, i.e. used columns) & model with pickle.
    """
    with open('./car_corpBehavior/trained_model/feat_pipe.pickle', 'wb') as f:
        pickle.dump(p, f)

    with open('./car_corpBehavior/trained_model/trained_cpl.pickle', 'wb') as f:
        pickle.dump(cpl, f)

    """
    Use trained model. (Read saved model with pickle first).
    """
    raw = pd.read_parquet('./car_corpBehavior/data/medium_pivot/input_data.parq')
    X, y = raw.drop('fassured', axis=1), raw['fassured']
    with open('./car_corpBehavior/trained_model/trained_cpl.pickle', 'rb') as f:
        trained_cpl2 = pickle.load(f)
    pred_proba = trained_cpl2.predict_proba(X)

    res = pd.DataFrame([X["iassured"], pred_proba], index=["iassured", "positive_proba"]).T
    res.to_csv("./car_corpBehavior/trained_model/res.txt", sep='|', index=False)

    # 要先跑上面的36-44行才能跑以下:
    # X_test = X.loc[X.index.difference(X_train.index)]
    # y_test = y.loc[X_test.index]
    # y_test = y_test.apply(lambda x: 1 if x=='2' else 0)
    # y_pred_proba = trained_cpl2.predict_proba(X_test)









#
