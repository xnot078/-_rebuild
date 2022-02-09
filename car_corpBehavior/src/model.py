import pandas as pd
import numpy as np
import re, tqdm, pickle, math, copy
from collections import defaultdict


"""plot"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

"""report"""
from scipy import stats
from car_corpBehavior.preprocess import featureInsight

"""prepare"""
from sklearn import model_selection
from sklearn import base
from sklearn import exceptions as sk_excpt

"""prrprocess"""
from car_corpBehavior.src import pipe_Xy

"""metric"""
from sklearn import metrics

"""alg"""
from lightgbm import LGBMClassifier


class cross_validation():
    """
        args:
            X: pd.DataFrame; input feature data.
            y: pd.DataFrame; label of input feature data.
            model: the model would been trained.
            *
        attributes:
            model: trained model.
            pred_Kfold: a dataframe of true labels and proba{positive predtion}.
            scores: dict of scores (avg. precisionl, precision & recall under specific positive-proba-thresh.)
    """
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, model=None, n_splits=5, random_state=2018, pos_label='2', neg_label='1', **kwargs):
        pos_label = kwargs.get('pos_label', pos_label)
        neg_label = kwargs.get('neg_label', neg_label)
        pd.options.mode.chained_assignment = None
        self.model = cross_validation.init_model(model)
        self.model, self.pred_Kfold = self.cv_process(X, y, self.model, n_splits=n_splits, random_state=random_state)
        self.scores = self.scores_process(self.pred_Kfold['true'], self.pred_Kfold['pred'], pos_label=pos_label, neg_label=neg_label)

    @staticmethod
    def cv_process(X_train, y_train, model, n_splits=5, random_state=2018):
        """
        returns:
            -trained model.
            -a dataframe of true labels and proba{positive predtion}.
        """
        # cross-validation process
        pred_Kfold = pd.DataFrame([], index=X_train.index, columns=['fold', 'true', 'pred']) # 儲存各fold結果，可以用來畫precision_recall_curve
        pred_Kfold['true'] = y_train
        kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(pred_Kfold.index, pred_Kfold['true'])):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
            model_ = cross_validation.init_model(model)
            model_.fit(X_train_fold, y_train_fold)
            pred_proba = model_.predict_proba(X_test_fold)
            pred_Kfold['pred'].iloc[test_idx] = pred_proba[:, 1]
            pred_Kfold['fold'].iloc[test_idx] = fold

        model.fit(X_train, y_train)
        return model, pred_Kfold

    @staticmethod
    def scores_process(y_true, y_pred, thresh=[.5, .75, .9], pos_label='2', neg_label='1'):
        """
        args:
            y_true: array-like; true label of instances.
            y_pred: array-lilk; positive probabilities of instances.
            thresh: array-like; to see scores with specific proba-threshs of positive predictions.
        return:
            dict(
                avg_precision = float; average precision of whole y_true and y_pred.
                precision = dict( thresh=precision score of (y_true, y_pred>=th) )
                recall = dict( thresh=recall score of (y_true, y_pred>=th) )
            )
        """
        scores = {'avg_precision':None, 'precision':defaultdict(dict), 'recall':defaultdict(dict)}
        # scores = defaultdict(lambda: defaultdict(list)) # 各thresh的分數
        scores['avg_precision'] = metrics.average_precision_score(y_true, y_pred, pos_label=pos_label) # 整體平均的avg_precision
        for th in thresh: # 在各thresh下的scores
            y_pred_label = y_pred>=th
            y_pred_label[y_pred>=th] = pos_label
            y_pred_label[~(y_pred>=th)] = neg_label
            scores['precision'][th] = metrics.precision_score(y_true, y_pred_label, pos_label=pos_label)
            scores['recall'][th] = metrics.recall_score(y_true, y_pred_label, pos_label=pos_label)
        return scores

    @staticmethod
    def init_model(model):
        # 先copy再訓練
        # 如果model=None，使用預設的LGBMClassifier。特別注意為了保持結果一致性，預設的LGBM有把random_state訂死。
        if model is None:
            model_ = LGBMClassifier(
                        objective='binary',
                        metric='binary_logloss',
                        max_depth=8, num_leaves=15,
                        bagging_seed=2018,
                        verbose=0, n_jobs=-1,
                        random_state=2018
                        )
        else:
            model_ = copy.deepcopy(model)
        return model_

class car_potential_legal(base.ClassifierMixin, base.BaseEstimator, cross_validation):
    def __init__(self, model=None, **kwargs):
        """
        model: sklearn.estimator; 任一分類器，如果還沒訓練過warm_start必須設定為Fasle。
        warm_start: bool; 輸入的模型是否已經訓練過。(不論有無都可以在fit，但沒有fit過的model不能predict。)
        """
        self.model = model # if None, use default light GBM

    def fit(self, X, y, **kwargs):
        print('model: ', self.model)
        print('========== fitting and cross validation ==========')
        cross_validation.__init__(self, X, y, self.model, **kwargs)
        return self

    def predict_proba(self, X, y=None, model_feature_name=None, **kwargs):
        try:
            if model_feature_name is None and self.model is not None:
                # !!對齊欄位非常重要!! 因為欄位的順序可能不一樣
                # 但因為不同模型可能有不同的feature_name存放模式，所以提供一個接口
                X = X[self.model.feature_name_]
            pred = self.model.predict_proba(X)
        except (AttributeError, UnboundLocalError) as e:
            raise Exception('No model is initialized or fitted.\nUse "car_potential_legal.fit(X, y, model)" first.  (Note: If model=None when call .fit(), default model(LGBMClassifier) would be used.)') from e
        except sk_excpt.NotFittedError as e:
            raise repr(e)
        if isinstance(X, pd.DataFrame):
            pred = pd.Series(pred[:, 1], index=X.index)
        return pred

    def predict(self, X, y=None, thresh=0.5, **kwargs):
        # pred > thresh(default=0.5), it's positive.
        return (self.predict_proba(X) >= thresh).astype(int)

    def fit_predict(self, X, y, thresh=0.5, **kwargs):
        self.fit(X, y, **kwargs)
        return self.predict(X, thresh=thresh)

if __name__ == '__main__':

    raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq')
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
    X_train = p.transform(X, y, use_clustering=True, balanced_label='fassured', balanced_base_on='2')
    """
    y preprocess:
        - After X preprocess and balanced-sampling, it's necessary to index y by X.index.
        - To use sklearn scoring methods, binary labels should be 0 and 1.
    """
    # y[y=='2'] = 1
    # y[y=='1'] = 0
    y_train = y.loc[X_train.index]

    model = LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                max_depth=4, num_leaves=20,
                bagging_seed=2018,
                verbose=0, n_jobs=-1,
                random_state=2018
                )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=2018)

    cpl = car_potential_legal() # init_model
    cpl.fit(X_train, y_train, K=10) # fit
    cpl.model.feature_name_
    for k, v in cpl.scores.items():
        print(k, end='')
        if isinstance(v, dict):
            print()
            for thresh, score in v.items():
                print(f'\t{thresh=}: {score:.1%}')
        else:
            print(f': {v:.1%}')


    cpl2 = car_potential_legal(cpl.model)
    # pred
    test = pd.concat([X_test, y_test], axis=1)
    X_test, y_test = pipe_Xy.pipe_prepare(test, is_train=False)
    y_test.value_counts()
    pred_proba = cpl.predict_proba(X_test)
    pred = pd.concat([pred_proba, y_test], axis=1)
    pred.columns = ['pred_proba', 'true']
    y_test.value_counts()
    print('avg. precision of holdout validation data: ', metrics.average_precision_score(pred['true'], pred['pred_proba']))


"""
下個目標是用.8050開一個hover連動的圖
左邊是precision & recall，右邊是該thresh以上的subset的統計量
"""


#
# def compare_report(list_predictionBasedOnKFolds, list_names, list_model, name):
#     ## precision-recall curve
#     figHtml = make_subplots(rows=1, cols=2)
#
#     fig, axs = plt.subplots(1, 2, figsize=(13,5))
#     fig.suptitle(f'關於閥值: {name}', fontsize=20)
#     for n, predictionBasedOnKFolds in zip(list_names, list_predictionBasedOnKFolds):
#         preds = pd.concat([y_train, predictionBasedOnKFolds.loc[:, 1]], axis=1)
#         preds.columns = ['true', 'pred']
#         precision, recall, thresh = metrics.precision_recall_curve(preds['true'], preds['pred'])
#         avg_precision = metrics.precision_score(preds['true'], preds['pred'])
#
#         axs[0].step(recall, precision, alpha=.7, where='post', label=f'{n}:{avg_precision:.2%}')
#         axs[0].fill_between(recall, precision, alpha=.3, step='post')
#
#         fpr, tpr, thresh = metrics.roc_curve(preds['true'], preds['pred'])
#         areaUnderROC = metrics.auc(fpr, tpr)
#         axs[1].plot(fpr, tpr, lw=2, label=f'{n}:{areaUnderROC:.2f}')
#
#         figHtml.add_trace(
#                     go.Scatter(x=recall, y=precision),
#                     row=1, col=1
#                 )
#         figHtml.add_trace(
#                     go.Scatter(x=fpr, y=tpr),
#                     row=1, col=2
#                 )
#
#     axs[0].set_xlabel('recall')
#     axs[0].set_ylabel('precision')
#     axs[0].set_ylim([0, 1.05])
#     axs[0].set_xlim([0, 1])
#     axs[0].set_title(f'Precision(猜正且對/猜是正) v.s. Recall(猜正且對/所有正)', fontsize=12)
#     axs[0].grid()
#     axs[0].legend()
#
#     axs[1].plot([0,1], [0,1], color='k', linestyle='--')
#     axs[1].set_xlabel('False positive rate')
#     axs[1].set_ylabel('True positive rate')
#     axs[1].set_title(f'TruePositive(真的猜對) v.s. NegativePositive(猜對但其實是錯)', fontsize=12)
#     axs[1].grid()
#     axs[1].legend(loc='lower right')
#
#     plt.tight_layout()
#     fig.savefig(f'./trained_model/{name}_compare.png')
#
#     pyo.plot(figHtml, filename=f'./trained_model/{name}_compare.html')
#
#     pred_status = []
#     for m in list_model:
#         predBasedOnTest = m.predict(X_test)
#         acc = metrics.accuracy_score(y_test, predBasedOnTest)
#         p = metrics.precision_score(y_test, predBasedOnTest)
#         r = metrics.recall_score(y_test, predBasedOnTest)
#         pred_status.append((acc, p, r))
#     return pred_status
#
# def report_plotly(predictionBasedOnKFolds, model, filename):
#     ## precision-recall curve
#     preds = pd.concat([y_train, predictionBasedOnKFolds.loc[:, 1]], axis=1)
#     preds.columns = ['true', 'pred']
#     precision, recall, thresh = metrics.precision_recall_curve(preds['true'], preds['pred'])
#     avg_precision = metrics.precision_score(preds['true'], preds['pred'])
#
#     dfNature = raw[raw['fassured']=='1']
#     X, y = prepareXy(dfNature, sample_rate='all')
#     feat = X.columns
#     iassuard = X.index
#     X = pipe.transform(X)
#     X = pd.DataFrame(X, index=iassuard, columns=feat)
#
#     subplot_titles = ['Precision(猜+且對/猜+) v.s. Recall(猜+且對/所有+)',
#                         'TruePositive: pr(猜+|真的是+)v.s.NegativePositive: pr(猜+|其實是-)',
#                         '關於閥值選擇: 精準還是漏抓的權衡<br>(<span style="color:purple;font-size:0.8rem">紫線:precision</span>, <span style="color:orange;font-size:0.8rem">橘線:recall</span>, <span style="color:green;font-size:0.8rem">綠線:accuracy</span>)',
#                         f'閥值 v.s. 實際預測為法人人數<br>(pool=自然人{len(iassuard):,.0f}位)']
#     spec = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]]
#     fig = make_subplots(cols=1, rows=4, print_grid=True, subplot_titles=subplot_titles, specs=spec)
#     ###
#     fig.add_trace(
#         go.Scatter(
#             x=recall, y=precision, fill='tozeroy', hovertemplate='Precision:\t%{y:.1%}<br>Recall:\t%{x:.1%}'
#         )
#         ,col=1, row=1)
#     fig['layout']['xaxis']['title'] = 'Recall'
#     fig['layout']['yaxis']['title'] = 'Precision'
#     ###
#     fpr, tpr, thresh = metrics.roc_curve(preds['true'], preds['pred'])
#     areaUnderROC = metrics.auc(fpr, tpr)
#     fig.add_trace(
#         go.Scatter(
#             x=fpr, y=tpr, hovertemplate='True Positive Rate:\t%{y:.1%}<br>False Positive Rate:\t%{x:.1%}'
#         )
#         ,col=1, row=2)
#     fig['layout']['xaxis2']['title'] = 'NegativePositive'
#     fig['layout']['yaxis2']['title'] = 'TruePositive'
#     ###
#     accuracy_hold, precision_hold, recall_hold, positiveNum_holder = [], [], [], []
#     for th in np.arange(0, 1, 0.02):
#         pTh = preds['pred']>=th
#         precision = metrics.precision_score(preds['true'], pTh.astype(int))
#         recall = metrics.recall_score(preds['true'], pTh.astype(int))
#         acc = metrics.accuracy_score(preds['true'], pTh.astype(int))
#         accuracy_hold.append(acc)
#         precision_hold.append(precision)
#         recall_hold.append(recall)
#         positiveNum_holder.append(pTh.sum()/len(pTh))
#     for hold, name in zip([accuracy_hold, precision_hold, recall_hold], ['Accuracy', 'Precision', 'Recall']):
#         fig.add_trace(
#             go.Scatter(
#                 x=np.arange(0, 1, 0.02), y=hold, hovertemplate=name+':\t%{y:.1%}', name=name
#             )
#             ,col=1, row=3)
#
#     pred_proba = model.predict_proba(X[model.feature_name_])
#     predPopulation = pd.DataFrame([y.values, pred_proba[:, 1]], index=['true', 'pred']).T
#     positiveNumPopulation_holder = []
#     for th in np.arange(0, 1, 0.02):
#         pTh = predPopulation['pred']>=th
#         positiveNumPopulation_holder.append(pTh.sum())
#
#     meta = [f'<b>人數:{num:,.0f}</b><br>----------------<br>acc:{a:.1%}<br>precision:{p:.1%}<br>recall:{r:.1%}' for num, a, p, r in zip(positiveNumPopulation_holder, accuracy_hold, precision_hold, recall_hold)]
#
#     fig.add_trace(
#         go.Scatter(
#             x=np.arange(0, 1, 0.02), y=positiveNumPopulation_holder, meta=meta,
#             hovertemplate='閥值:%{x:.1%}<br>%{meta}', name='預測法人人數'
#         )
#         ,col=1, row=4)#, secondary_y=True)
#
#
#     fig['layout']['xaxis3']['title'] = '閥值'
#     fig['layout']['yaxis3']['title'] = '分數'
#
#     fig.update_layout(
#         width=800, height=1500,
#         hovermode='x'
#         )
#     fig.update_xaxes(showspikes = True)
#     fig.update_yaxes(showspikes = True)
#     # filename='lightbgm_險別車種分開_v3'
#     pyo.plot(fig, filename=f'{filename}_指標一覽.html')
#
#
# def shap_bar_oneInput_bar(df, idx, tops=1000):
#     """
#     看單一個input各特徵的影響力 & 原始值(以及原始值在X_train的PR,包括含0與不含0)
#     """
#     dfFts = df[model.feature_name_]
#     shap_values = pd.DataFrame(
#                         model.predict_proba(dfFts.iloc[[idx]].head(tops).values, pred_contrib=True),
#                         columns=feature_name.to_list()+['base']
#                         )
#
#     ser = shap_values.iloc[idx]
#     # print(ser)
#     topEffects = ser.abs().sort_values().tail(10).index
#     serTops = ser.loc[topEffects]
#     serTops.loc['others'] = ser.loc[ser.index.difference(topEffects)].sum()
#     serTops.name = 'shapVal'
#
#     info = pd.DataFrame(columns=['inputVal', 'pr', 'prNonZero'])
#     for ft in topEffects:
#         if ft in X_train.columns:
#             ft
#             inputVal = df[ft].iloc[idx]
#             populationSer =  X_train[ft]
#             pr = stats.percentileofscore(populationSer, inputVal)
#             prNonZero = stats.percentileofscore(populationSer[populationSer>0], inputVal)
#             info.loc[ft] = [inputVal, pr, prNonZero]
#
#     dfPlot = pd.concat([serTops, info], axis=1)
#     dfPlot.sort_values('shapVal', inplace=True)
#
#     code, pred_proba, SHAP_val = df.iloc[idx].name, df.iloc[idx]['pred'], serTops.sum()
#     data = [
#             go.Bar(
#                     y=dfPlot.index,
#                     x=dfPlot['shapVal'],
#                     text= dfPlot[['pr', 'prNonZero']].apply(lambda row: \
#                                                                 f'{row["pr"]:.0f} | {row["prNonZero"]:.0f} '\
#                                                                 if not row.isna().all() else '', axis=1\
#                                                                     ),
#                     textposition='auto',
#                     meta= dfPlot[['inputVal','pr','prNonZero']].apply(lambda row: f"值: {row['inputVal']:,.0f}<br>PR (含0): {row['pr']:.1f}<br>PR (不含0): {row['prNonZero']:.1f}" if row['pr']>=0 else '', axis=1),
#                     marker_color=['red' if _<0 else 'blue' for _ in dfPlot['shapVal']],
#                     hovertemplate='<b>%{y}</b><br>--------------------------<br>影響分數(SHAP): %{x:.1f}<br>--------------------------<br>%{meta}', #<br>原始值: %{meta}<br>在母體(訓練40萬人)中PR:<br>%{text}<br>(包含0|不含0)
#                     orientation='h'
#                 )
#             ]
#     layout = go.Layout(title=f'{code}<br>Prob: {pred_proba:.1%} (SHAP: {SHAP_val:.1f})')
#     fig = go.Figure(data, layout)
#     return fig
#
# # df = dfP[mask]
# def shap_bar_oneIntput(df, idx, X_train, y_train, tops=10):
#     """
#     看單一個input各特徵的影響力 & 原始值(以及原始值在X_train的PR,包括含0與不含0)
#     args:
#         df: 第一col為pred，接著是所有特徵的值
#         idx: 這次是要畫df中的哪一個row
#         X_train: 用來比較原始值在資料集當中位置的資料，最好用X_train
#         y_train: 用來區分法人 & 自然人
#     """
#     dfFts = df[model.feature_name_]
#     shapSer = pd.Series(
#                         model.predict_proba(
#                             dfFts.iloc[[idx]].values, pred_contrib=True
#                             )[0],
#                         index=model.feature_name_+['base']
#                         )
#     title = f'{df.index[idx]}\nProba={df["pred"].iloc[idx]:.1%}, SHAP={shapSer.sum():.1f}'
#
#     """圖框"""
#     fig = plt.figure(constrained_layout=True, figsize=(12, 5))
#     fig.suptitle(title)
#     gs = fig.add_gridspec(3, 6)
#     """SHAP貢獻圖"""
#     axsLeft = fig.add_subplot(gs[:, :3])
#
#     # 排除base後影響最大的前10:
#     topEffectIdx = shapSer.drop('base').abs().sort_values(ascending=False).head(tops).index
#     topEffect = shapSer[topEffectIdx]
#     topEffect.sort_values(ascending=False, inplace=True)
#     topEffect['others'] = shapSer.drop('base')[shapSer.drop('base').index.difference(topEffect.index)].sum() # top10之外的總和
#     topEffect['base'] = shapSer['base']
#
#     axsLeft.barh(
#         topEffect.index.str.replace('_mean|-1', '')[::-1],
#         topEffect[::-1],
#         color=topEffect[::-1].apply(lambda x: '#e8203b' if x<0 else '#547ce3')
#         ) # 貢獻度水平長條圖
#     axsLeft.plot(
#         topEffect[::-1].cumsum(),
#         topEffect.index.str.replace('_mean|-1', '')[::-1],
#         color='black', linestyle='--', linewidth=1, marker='.',
#         ) # 貢獻度累計曲線
#     axsLeft.grid()
#
#     """分布圖"""
#     X_train_law = X_train[y_train==1]
#     X_train_nature = X_train[y_train==0]
#
#     c = 0
#     for i in range(3):
#         for j in range(3):
#             ax = fig.add_subplot(gs[i, 3+j]) # 建立框架 & 指定圖片位置
#             ftName = topEffect.index[c] # 確定目標欄位
#             sns.histplot(X_train_law[ftName], stat='density', color='#fca547', label='法人', ax=ax)
#             sns.histplot(X_train_nature[ftName], stat='density', color='#4774fc', label='自然人', ax=ax)
#             realValue = df.iloc[idx][topEffect.index[c]]
#             vl = ax.axvline(realValue, ymin=0, ymax=1, color='red')
#
#             vals = X_train[ftName]
#             prWithZero = stats.percentileofscore(vals, realValue)
#             prWithoutZero= stats.percentileofscore(vals[vals>0], realValue)
#             ax.annotate(
#                 f'PR:\n all:{prWithZero:.1f}\n >0: {prWithoutZero:.1f}',
#                 xy=[realValue, ax.get_ylim()[-1]],
#                 verticalalignment='top',
#                 color='#9c0500')
#             ax.set_xlabel(re.sub('_mean|-1', '', ftName))
#             c += 1
#     plt.tight_layout()
#     return fig
#
#
# def sigmoid(x):
#     sig = 1 / (1 + math.exp(-x))
#     return sig
#
# if __name__ == '__main__':
#     """prepare"""
#     raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq')
#     raw['fassured'].value_counts(normalize=True)
#     X, y = pipe_Xy.pipe_prepare(raw, is_train=True, sample_rate=3,
#                                 use_anomaly_detect=True, anomaly_drop_ratio=0.05,
#                                 anomaly_method='cluster', K=10, method_in_cluster='pca',
#                                 fill_value=0, random_state=2018)
#
#     feature_name = X.columns
#     iassuard = X.index
#     # pd.DataFrame(feature_name).sort_values(0)
#     print(f'# of feat.s: {len(feature_name)}')
#
#
#
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=2018)
#     # y_test.value_counts()
#     X_train = pd.DataFrame(X_train, columns=feature_name, index=y_train.index)
#     y_train.value_counts()
#
#
#     # 比較3個不同的模型
#     models = [
#         LogisticRegression(penalty='l2', class_weight='balanced', C=1, n_jobs=4),
#         # KNeighborsClassifier(n_neighbors=100, weights='distance', n_jobs=4),
#         RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=4),
#         LGBMClassifier(objective='binary', metric='binary_logloss', max_depth=8, bagging_seed=2018, verbose=0, num_threads=11, num_leaves=15, feature_name=[str(i) for i in range(len(feature_name))])
#     ]
#     names = [
#          'LR',
#          # 'KNN',
#          'RF',
#          'lightGBM'
#     ]
#
#     # Cross Validation for 5 folds
#     kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
#
#     list_predictionBasedOnKFolds = []
#     list_trainedModel = []
#     for m, n in zip(models, names):
#         trained_model, predictionBasedOnKFolds = cv_training(m, n)
#         list_trainedModel.append(trained_model)
#         list_predictionBasedOnKFolds.append(predictionBasedOnKFolds)
#
#     # 繪製報告
#     name = 'LR_RF_LGBM_Ver.iroute'
#     pred_status = compare_report(list_predictionBasedOnKFolds, names, list_trainedModel, name)
#     pd.DataFrame(pred_status, index=names, columns=['accuracy', 'precision', 'recall'])
#
#     pd.DataFrame(pred_status, index=names, columns=['accuracy', 'precision', 'recall']).to_csv(f'./trained_model/{name}_summary.csv')
#
#     name = 'lightbgm_險別車種分開_v5'
#     # 統整報告 (包含符合人數)
#     report_plotly(list_predictionBasedOnKFolds[-1], models[-1], name)
#
#     model = models[-1]
#     with open(f'./trained_model/model_lightGBM.pickle', 'wb') as f:
#         pickle.dump(model, f)
#
#
#
#     """拿去打自然人"""
#     dfNature = raw[raw['fassured']=='1']
#     X_nature, y_nature = prepareXy(dfNature, sample_rate='all', use_anomaly_detect=False)
#     feat = X_nature.columns
#     iassuard = X_nature.index
#     X_nature = pipe.transform(X_nature)
#     X_nature = pd.DataFrame(X_nature, index=iassuard, columns=feat)
#     X_nature = X_nature[model.feature_name_]
#
#     pred_proba = model.predict_proba(X_nature)
#     pred_proba = pd.Series(pred_proba[:, 1], index=iassuard)
#     pred_proba.name = 'pred'
#     SHAP =  pd.DataFrame(
#                     model.predict_proba(X_nature[model.feature_name_], pred_contrib=True),
#                     columns=list(model.feature_name_)+['base'],
#                     index=X_nature.index
#                         )
#
#     dfP = pd.concat([pred_proba, X_nature[model.feature_name_]], axis=1)
#     mask = pred_proba>0.9
#     sum(mask)
#
#     dfTops = dfP[mask].sort_values('pred', ascending=False)
#     ipolicy = pd.read_csv('./data/tem/byPersonIpolicy.csv')
#
#     dfTops = dfTops.merge(ipolicy, left_index=True, right_on='iassured', how='left')
#     dfTops.set_index(['iassured', 'ipolicy'], inplace=True)
#     """th > 90"""
#     dfFts = dfTops[model.feature_name_]
#     dfSHAP = pd.DataFrame(
#                         model.predict_proba(
#                             dfFts.values, pred_contrib=True
#                             ),
#                         columns=model.feature_name_+['base']
#                         )
#     dfSHAP.index = dfTops.index
#
#     # 每個人只取top10的SHAP (EXCEL格式化用)
#     holder = []
#     for i, row in dfSHAP.iterrows():
#         ser = pd.Series(0, index=row.index)
#         topIdx = row.drop('base').abs().sort_values(ascending=False).head(10).index
#         ser.loc[topIdx] = row[topIdx]
#         holder.append(ser)
#     dfSHAPtop10 = pd.concat(holder, axis=1).T
#     dfSHAPtop10.index = dfSHAP.index
#
#
#     """每個樣本的top10貢獻fts，組成的維度，在統計上看得到法人和自然人分開嗎?"""
#     holder = []
#     for i, row in dfSHAPtop10.iterrows():
#         usedFts = row[row!=0].index
#         arrLaw = X_train[y_train==1][usedFts]
#         arrNature = X_train[y_train==0][usedFts]
#         overlapLaw, overlapNature = featureInsight.nonoverlap_area_multiDimension(arrLaw, arrNature, return_overlapNat=True)
#         holder.append({'overlapLaw':overlapLaw, 'overlapNature':overlapNature})
#     dfNonOverlap = pd.DataFrame(holder, index=dfSHAPtop10.index)
#
#     """存檔"""
#     writer = pd.ExcelWriter('./圖表/個案/dfTh90_anomaly.xlsx')
#     dfTops = pd.concat([dfTops, dfNonOverlap], axis=1)
#     dfTops = dfTops[['pred', 'overlapLaw', 'overlapNature']+model.feature_name_]
#     dfTops.columns = ['pred', '法人不重疊自然人', '自然人不重疊法人']+model.feature_name_
#     dfTops.reset_index().to_excel(writer, sheet_name='個案', index=False)
#     dfSHAP.reset_index().to_excel(writer, sheet_name='SHAP', index=False)
#     dfSHAPtop10.reset_index().to_excel(writer, sheet_name='SHAP_Tops', index=False)
#
#     writer.save()
#     writer.close()
#     #
#     #
#     # dfList = pd.read_excel(r'D:\新安\正式\車險\圖表\個案\dfTh90_2021.xlsx')
#     # dfList = dfList.set_index(['iassured', 'ipolicy'])[dfTops.columns]
#     #
#     # dfTops
#     # import os
#     # listTop_noCago_old = os.listdir('D:\新安\正式\車險\圖表\個案\個案圖\新增資料夾')
#     # listTop_noCago_old = [i.split('_')[0] for i in listTop_noCago_old if i.endswith('.jpg')]
#     # tem = dfTops.loc[listTop_noCago_old]
#     # dfTops_noCago.index.get_level_values()
#     # tem = X_test.loc[set(X_test.index).intersection(dfTops_noCago.index.get_level_values(0))]
#     # tem.filter(regex='貨')
#     # tem[(tem.filter(regex='貨') == 0).all(axis=1)]
#     #
#
#     # # for i in tqdm.trange(len(dfTops)):
#     # dfList[(dfList.reset_index()['iassured'] == 'e28e24a62df5f78d9bc05b1808e8208b6229548').values]
#     dfTops_noCago = dfTops[(dfTops.filter(regex='貨') == 0).all(axis=1)]
#     for i in tqdm.trange(len(dfTops_noCago)):
#         # row = dfTops_noCago.iloc[[i]]
#         # model.predict_proba(row.iloc[:, 3:][model.feature_name_])
#         fig = shap_bar_oneIntput(dfTops_noCago, i, X_train, y_train)
#         dst = '_'.join(dfTops_noCago.iloc[i].name) + '.jpg'
#         fig.savefig('./圖表/個案/個案圖/新增資料夾/'+dst)
#
#     dfTops_noCago.to_excel('./圖表/個案/個案圖/新增資料夾/dfTH90_不含貨車相關保單.xlsx')
