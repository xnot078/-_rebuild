"""main"""
import pandas as pd
import numpy as np
import os, re, pickle, tqdm
from itertools import combinations

"""plot"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo

"""prepare."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

"""alg"""
from  sklearn.decomposition import PCA

def nonoverlap_area_1D(arr_nature, arr_law):
    """
    輸入兩個arr，轉成distplot後，看重疊面積的百分比(arr_law不重複於arr_law的比例)
    """
    fig, axs = plt.subplots()
    sns.distplot(arr_nature, ax=axs)
    sns.distplot(arr_law, ax=axs)
    if len(axs.lines) <= 1:
        return None
    x1 = axs.lines[0].get_xdata()
    x2 = axs.lines[1].get_xdata()

    if x1.max()>x2.min() or x2.max()>x1.min():
        left = max(x1.min(), x2.min())
        right = min(x1.max(), x2.max())
        clip = {'clip': (left, right)}
        fig, axs = plt.subplots()
        sns.distplot(arr_nature, ax=axs, kde_kws=clip)
        sns.distplot(arr_law, ax=axs, kde_kws=clip)
        ymin = np.minimum(axs.lines[0].get_ydata(), axs.lines[1].get_ydata())
        overlap = np.trapz(ymin, axs.lines[0].get_xdata())
        return 1-overlap
    else:
        return 1

def nonoverlap_area_multiDimension(arr_law, arr_nature, prX=0.8, return_overlapNat=False, visualize=False, alpha=0.05, figType='2D'):
    """
    輸入兩個arr(第一個為法人、第二個為自然人)
    劃出不在彼此涵蓋80%資料範圍的比例 (e.g. 法人資料不在涵蓋自然人80%資料的範圍內的比例)
    visualize=True則回傳圖表
    如果輸入的arr是2維、figType='2D'時，將繪製2D散佈圖
    如果輸入的arr是3維、figType='3D'時，將繪製3D散佈圖
    """
    # 重心
    centerLaw = arr_law.mean(axis=0)
    centerNature = arr_nature.mean(axis=0)
    # pr80的半徑
    dist_law2CenterLaw = pairwise_distances([centerLaw], arr_law)
    prRadiusLaw = np.quantile(dist_law2CenterLaw, prX)
    dist_nat2CenterNat = pairwise_distances([centerNature], arr_nature)
    prRadiusNat = np.quantile(dist_nat2CenterNat, prX)
    # 落入對方prRadius範圍內的比例
    dist_law2CenterNat = pairwise_distances([centerNature], arr_law) # 所有法人資料 到 自然人重心的距離
    overlapLaw = np.mean(dist_law2CenterNat <= prRadiusNat) # 上述距離，超過涵蓋自然人prX範圍的比例 (表示離自然人重心夠遠)

    dist_nat2CenterLaw = pairwise_distances([centerLaw], arr_nature)
    overlapNat = np.mean(dist_nat2CenterLaw <= prRadiusLaw)

    overlap = {'nature': overlapNat, 'law': overlapLaw}

    if not visualize:
        if not return_overlapNat:
            return 1-overlap['law']
        else:
            return 1-overlap['law'], 1-overlap['nature']

    elif figType=='2D':
        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        axs[0].set_title('法人+自然人')
        axs[1].set_title('自然人')
        axs[2].set_title('法人')
        # 自然人
        axs[0].scatter(arr_nature[:, 0], arr_nature[:,1], linewidth=0.5, edgecolor='k', alpha=0.2, c='#6d87fc')
        axs[1].scatter(arr_nature[:, 0], arr_nature[:,1], linewidth=0.5, edgecolor='k', alpha=0.2, c='#6d87fc')
        # 法人
        axs[0].scatter(arr_law[:, 0], arr_law[:,1], linewidth=0.5, edgecolor='k', alpha=0.2, c='#fcc06d')
        axs[2].scatter(arr_law[:, 0], arr_law[:,1], linewidth=0.5, edgecolor='k', alpha=0.2, c='#fcc06d')
        # 框出範圍
        axs[0].scatter(centerNature[0], centerNature[1], linewidth=0.5, c='blue', alpha=1, label='自然人')
        circle = plt.Circle(centerNature, prRadiusNat, fill=False, color='blue')
        axs[0].add_patch(circle)
        axs[0].scatter(centerLaw[0], centerLaw[1], linewidth=0.5, c='r', alpha=1, label='法人')
        circle = plt.Circle(centerLaw, overlapLaw, fill=False, color='r')
        axs[0].add_patch(circle)
        plt.legend()
        if not return_overlapNat:
            return 1-overlap['law'], fig
        else:
            return 1-overlap['law'], 1-overlap['nature'], fig
    else:
        import plotly.express as px

        data = [
            go.Scatter3d(x=arr_law.iloc[:, 0].sample(frac=0.05).values,
                       y=arr_law.iloc[:, 1].sample(frac=0.05).values,
                       z=arr_law.iloc[:, 2].sample(frac=0.05).values,
                       name='法人', legendgroup='法人', mode='markers', marker=dict(color='red', size=3), opacity=0.5, showlegend=False),
            go.Scatter3d(x=arr_nature.iloc[:, 0].sample(frac=0.05).values,
                       y=arr_nature.iloc[:, 1].sample(frac=0.05).values,
                       z=arr_nature.iloc[:, 2].sample(frac=0.05).values,
                       name='自然人', legendgroup='自然人', mode='markers', marker=dict(color='blue', size=3, symbol='circle-open'), line=dict(color='blue', width=2), opacity=0.5, showlegend=False)
        ]
        fig = go.Figure(data)
        name = ' & '.join([c.replace('_sum_mean', '').replace('-1', '') for c in arr_law.columns[:3]])
        fig['layout']['title'] = f"{name}: {1-overlap['law']:.1%}"
        fig['layout']['scene']['xaxis_title']=re.search('-(\w{1,})-', arr_law.columns[0]).group(1)
        fig['layout']['scene']['yaxis_title']=re.search('-(\w{1,})-', arr_law.columns[1]).group(1)
        fig['layout']['scene']['zaxis_title']=re.search('-(\w{1,})-', arr_law.columns[2]).group(1)

        if not return_overlapNat:
            return 1-overlap['law'], fig
        else:
            return 1-overlap['law'], 1-overlap['nature'], fig

def nonoverlap_scores_multiDimension(data_law, data_nature, featsList, comb=2, prX=0.8, drop_all_zero=True):
    """
    輸入特徵(feats: array-like str)，並輸入要怎麼從feats取得組合(e.g. comb=2表示Cn取2)
    2D以上時: 回傳不重疊比例
    """

    holder = []
    for feats in tqdm.tqdm(combinations(featsList, comb), total=len(list(combinations(featsList, comb)))):
        # feats = ['iins_sum_mean-責任險-1', 'iins_count_mean-責任險-1']
        arr_law = data_law[list(feats)].values
        arr_nature = data_nature[list(feats)].values
        if drop_all_zero:
            arr_law = arr_law[(arr_law > 0).any(axis=1)]
            arr_nature = arr_nature[(arr_nature > 0).any(axis=1)]
        if np.isnan(arr_law).any() or np.isnan(arr_nature).any():
            continue
        if len(arr_nature)!=0:
            nonoverlap = nonoverlap_area_multiDimension(arr_law, arr_nature, return_overlapNat=True)
            nonoverlap = {'law':nonoverlap[0], 'nature':nonoverlap[1], 'ft':feats}
            holder.append(nonoverlap)
        elif len(arr_nature)==0:
            holder.append({'law':0, 'nature':1, 'ft':feats})

    df = pd.DataFrame(holder)
    df.sort_values('law', ascending=False, inplace=True)
    return df
    # df3D[df3D['ft'].apply(lambda x: x[0]).str.match('.*責任險.*')].sort_values('law', ascending=False).head(20).values
    # df3D_nonZero.to_excel('./圖表/統計分析/3D重心距離_皆不為0_pr80.xlsx')

"""畫組圖 matplotlib (只有2D)"""
# tarFtSet = 'cartype'
# tarFt = '大貨車'
def scatter_matplot_set(data, nonOverlapScores, tarFtSet='iins', sample_frac=0.05, drop_zero=True, prX=0.8, title=''):

    feats = data.filter(like=f'{tarFtSet}_sum').columns
    feats = [re.search(r'-(\w{1,})-', f).group(1) for f in feats]
    fig, axs = plt.subplots(len(feats), 5, figsize=(15, 15))#, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=20)
    for r, tarFt in enumerate(feats):
        matchMask = pd.concat([nonOverlapScores['ft'].apply(lambda x: x[i]).str.match(f'.*sum.*{tarFt}') for i in range(2)], axis=1)
        match = matchMask.any(axis=1)
        tarTops_2D = nonOverlapScores[match].sort_values('law', ascending=False).head(5).copy()
        matchIdx = np.where(matchMask.loc[tarTops_2D.index])
        tarTops_2D.reset_index(inplace=True)
        # 把tarFt移到組合的第一個  (作圖時就一定會在x軸)
        for rr, i, ft in zip(matchIdx[0], matchIdx[1], tarTops_2D['ft']):
            ft = list(ft)
            tarTops_2D['ft'].iloc[rr] = [ft[i]] + ft[:i] + ft[i+1:]

        for c, row in tarTops_2D.iterrows():
            arr = data[list(row['ft'])]
            arrLaw, arrNature = arr[y_train==1], arr[y_train==0]
            # 畫圖用
            if drop_zero:
                arrN = arrNature[(arrNature>0).any(axis=1)]
                arrL = arrLaw[(arrLaw>0).any(axis=1)]
            else:
                arrN = arrNature
                arrN = arrLaw
            xN, yN = arrN.iloc[:, 0], arrN.iloc[:, 1]
            xL, yL = arrL.iloc[:, 0], arrL.iloc[:, 1]

            # 重心
            centerarrLaw = arrL.mean(axis=0)
            centerNature = arrN.mean(axis=0)
            # pr80的半徑
            if not arrL.empty:
                dist_to_centerarrLaw = pairwise_distances([centerarrLaw], arrL)
                prRadiusarrLaw = np.quantile(dist_to_centerarrLaw, prX)
            else:
                prRadiusarrLaw = 0
                overlap = 0
            if not arrN.empty:
                dist_to_centerNature = pairwise_distances([centerNature], arrN)
                prRadiusNature = np.quantile(dist_to_centerNature, prX)
                dist_lawToCenterNature = pairwise_distances([centerNature], arrL)
                overlap = np.mean(dist_lawToCenterNature >= prRadiusNature)
            else:

                prRadiusNature = 0
                overlap = 1

            ax = axs[r][c]
            if c == 0:
                ax.set_title(f'{tarFt}:{overlap:.1%}', fontsize=16)
            else:
                ax.set_title(f'{overlap:.1%}', fontsize=16)
            print(r, c, row['ft'], len(arrN), len(arrL))
            # ax.scatter(arrN.sample(frac=sample_frac).iloc[:, 0], arrN.sample(frac=sample_frac).iloc[:, 1], linewidth=0.8, edgecolor='#6d87fc', alpha=0.5, facecolors='none') # 自然人
            # ax.scatter(arrL.sample(frac=sample_frac).iloc[:, 0], arrL.sample(frac=sample_frac).iloc[:, 1], linewidth=0.8, edgecolor='#fcc06d', alpha=0.1, c='#fcc06d') # 法人

            # ax.scatter(xN.sample(frac=0.05), yN.sample(frac=0.05), linewidth=0.8, edgecolor='#6d87fc', alpha=0.1, facecolors='none')
            # ax.scatter(xL.sample(frac=0.05), yL.sample(frac=0.05), linewidth=0.8, edgecolor='#fcc06d', alpha=0.1, facecolors='none')
            if not arrN.empty:
                sns.kdeplot(data=arrN.sample(frac=sample_frac), x=arrN.columns[0], y=arrN.columns[1], color='#6d87fc', ax=ax, alpha=0.8, zorder=2)
            if not arrL.empty:
                sns.kdeplot(data=arrL.sample(frac=sample_frac), x=arrL.columns[0], y=arrL.columns[1], color='#fcc06d', ax=ax, alpha=0.5, zorder=1)
            # 框出範圍
            if not arrN.empty:
                ax.scatter(centerNature[0], centerNature[1], linewidth=0.5, c='blue', alpha=1, label='自然人', zorder=3)
                circle = plt.Circle(centerNature, prRadiusNature, fill=False, color='blue', zorder=2)
                ax.add_patch(circle)
            if not arrL.empty:
                ax.scatter(centerarrLaw[0], centerarrLaw[1], linewidth=0.5, c='r', alpha=1, label='法人', zorder=3)
                circle = plt.Circle(centerarrLaw, prRadiusarrLaw, fill=False, color='r', zorder=2)
                ax.add_patch(circle)
            ax.set_xlabel(re.search('-(\w{1,})-', arrL.columns[0]).group(1))
            ax.set_ylabel(re.search('-(\w{1,})-', arrL.columns[1]).group(1))

    fig.tight_layout()
    return fig


def anomaly_detection(df, model, thresh=0.5, ratio=None, by='fassured'):
    """
    排除標準化重構誤差>0.5的資料 (by label分開評估重購誤差)
    Args:
        df: pd.DataFrame，要含label
        model: 要使用的decomposition model
        thresh: 排除的閥值
        by: label的欄位名稱
    Return:
        mask: bp.array(bool); True表示要排除
    """
    mask = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    k, g = 0, df[df['fassured']==0]
    for k, g in df.groupby(by):
        g = g.drop(by, axis=1)
        X_tem = pipe.fit_transform(g)

        X_tem_reduced = model.fit_transform(X_tem)
        X_tem_reduced_inverse = model.inverse_transform(X_tem_reduced)
        loss = np.sum( (X_tem - X_tem_reduced_inverse)**2, axis=1 )
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        if ratio == None:
            mask.loc[g.index] = loss > thresh
            print(f'異常數量-{k}: {sum(loss > thresh)}')
        else:
            num = int(len(loss)*ratio)
            idx = np.argsort(loss)[-num:]
            idx = g.index[idx]
            mask.loc[idx] = True
            print(f'異常數量-{k}: {len(idx)}')

    return mask

def prepareXy(df, random_state=2018, sample_rate=3, anomaly_ratio=0.05):
    df = df.filter(regex='^(?!.*others|.*其他|.*max|.*carQt|.*outlier).*')
    df.columns
    # df = df.filter(regex='^(?!.*others|.*其他|.*outlier).*')
    df['fassured'] = df['fassured'].apply(lambda x: 1 if x=='2' else 0) # 雖然是在處理df，但是為了用抽樣來平行標籤比例，先保留標籤值，抽樣玩再丟掉
    columnsOrder = (~df.drop('fassured', axis=1).isna()).sum(axis=0).sort_values(ascending=False).index
    df = df[['fassured'] + list(columnsOrder)]
    df_recent = df.filter(regex='.*-1|fassured').copy() # 只取近五年
    # df_recent.filter(like='outlier').any(axis=1).sum()
    mask = ~df_recent.drop('fassured', axis=1).isna().all(axis=1) # 丟掉近五年完全沒有紀錄的人
    df_recent = df_recent[mask]
    # 排除極端值 (重購誤差最大的5%)
    anomaly_model = PCA(n_components=17, whiten=False, random_state=2018)
    anomaly_mask = ~anomaly_detection(df_recent, anomaly_model, ratio=anomaly_ratio)
    df_recent = df_recent[anomaly_mask]

    if sample_rate == 'all':
        return df_recent.drop('fassured', axis=1), df_recent['fassured']
    else:
        numPositive = sum(df_recent['fassured']==1) # 平衡label>>> 1:3
        df_recentSample = pd.concat([df_recent[df_recent['fassured']==1], df_recent[df_recent['fassured']==0].sample(numPositive*sample_rate, random_state=random_state)])
        df_recentSample = df_recentSample.sample(frac=1, random_state=random_state) # 打亂順序
        X_ans, y_ans = df_recentSample.drop('fassured', axis=1), df_recentSample['fassured']
        return X_ans, y_ans

pipe = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    # ('standard', StandardScaler()),
                    # ('decomposition', SparsePCA(n_components=50, alpha=1, random_state=2018))
                    # ('decomposition', PCA(n_components=30, random_state=2018))
                    # ('decomposition', Isomap(n_components=50, n_jobs=-1))
                ])


if __name__ == '__main__':
    """prepare"""
    raw = pd.read_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq')
    raw['fassured'].value_counts(normalize=True)
    X, y = prepareXy(raw, anomaly_ratio=0.05)
    feature_name = X.columns
    # pd.DataFrame(feature_name).sort_values(0)
    print(f'# of feat.s: {len(feature_name)}')
    pd.DataFrame(feature_name)

    X = pipe.fit_transform(X)

    with open('./trained_model/pipeX.pickle', 'wb') as f:
        pickle.dump(pipe, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2018)
    X_train = pd.DataFrame(X_train, columns=feature_name, index=y_train.index)

    """1D"""
    # 為了排序，先全部跑一遍
    holderOverlap = []
    X_train = X_train[[c for c in X_train.columns if c != 'label']]
    for ft in tqdm.tqdm(X_train.columns):
        mask = X_train[ft] > 0
        t  =  pd.concat(
                    [
                    X_train[ft][mask],
                    y_train[mask]
                    ], axis=1
                )

        arr1 = t[ft][t['fassured']==0]
        arr2 = t[ft][t['fassured']==1]
        holderOverlap.append( nonoverlap_area_1D(arr1, arr2) )

    # 作圖 1D
    featScores = pd.Series(holderOverlap, index=X_train.columns).sort_values(ascending=False)
    feats = featScores.index.values.reshape(5, 8)
    fig, axs = plt.subplots(5, 8, figsize=(30, 15))
    for r, row in enumerate(feats):
        for c, ft in enumerate(row):
            mask = X_train[ft] > 0

            t  =  pd.concat(
                        [
                        X_train[ft][mask],
                        y_train[mask]
                        ], axis=1
                    )
            # sns.histplot(data=t, x=ft, hue='fassured', ax=axs[r][c])
            # sns.distplot(data=t, x=ft, hue='fassured', ax=axs[r][c])
            sns.distplot(t[ft][t['fassured']==0], ax=axs[r][c])
            sns.distplot(t[ft][t['fassured']==1], ax=axs[r][c])

            axs[r][c].set_title(f'{ft}\noverArea: {featScores[ft]:.1%}')

    fig.tight_layout(pad=1)
    fig.savefig('./圖表/統計分析/單維度差異.jpg')

    """2D"""
    # 2維的scale必須相同，否則作圖和算涵蓋比例時會被scale大的那邊帶走
    X_train_scaled = (X_train - X_train.min(axis=0).values) / (X_train.max(axis=0) - X_train.min(axis=0))
    usedFeats = [c for c in X_train_scaled.columns if re.match('.*sum.*', c)]
    data_law, data_nature = X_train_scaled[y_train==1], X_train_scaled[y_train==0]
    nonOverlapScores_2D = nonoverlap_scores_multiDimension(data_law, data_nature, usedFeats, comb=2, drop_all_zero=False)
    nonOverlapScores_2D.to_excel('./圖表/統計分析/2D重心距離_包含皆為0_pr80.xlsx')
    nonOverlapScores_2D = nonoverlap_scores_multiDimension(data_law, data_nature, usedFeats, comb=2, drop_all_zero=True)
    nonOverlapScores_2D.to_excel('./圖表/統計分析/2D重心距離_排除皆為0_pr80.xlsx')

    for tar, dst, title in zip(['iins', 'cartype', 'iroute'],
                        ['./圖表/統計分析/iins_2d.jpg', './圖表/統計分析/cartype_2d.jpg', './圖表/統計分析/iroute_2d.jpg'],
                        ['維度: 各險種類別', '維度: 各車種類別', '維度: 各通路來源']
                        ):
        fig = scatter_matplot_set(X_train_scaled, nonOverlapScores_2D, tarFtSet=tar, sample_frac=0.05, drop_zero=True, prX=0.8, title=title)
        fig.savefig(dst)

    """3D"""
    nonOverlapScores_3D = nonoverlap_scores_multiDimension(data_law, data_nature, usedFeats, comb=3, drop_all_zero=False)
    nonOverlapScores_3D.to_excel('./圖表/統計分析/3D重心距離_包含皆為0_pr80.xlsx')
    nonOverlapScores_3D = nonoverlap_scores_multiDimension(data_law, data_nature, usedFeats, comb=3, drop_all_zero=True)
    nonOverlapScores_3D.to_excel('./圖表/統計分析/3D重心距離_排除皆為0_pr80.xlsx')

    for ft in nonOverlapScores_3D['ft'].head(100):
        ft
        arr_law, arr_nature = data_law[list(ft)], data_nature[list(ft)]
        _, fig = nonoverlap_area_multiDimension(arr_law, arr_nature, prX=0.8, return_overlapNat=False, visualize=True, alpha=0.05, figType='3D')
        name = '_'.join([c.replace('_sum_mean', '').replace('-1', '') for c in ft])
        pyo.plot(fig, filename=f'./圖表/統計分析/3D/{name}.html', auto_open=False)


    #
    # """畫組圖 plotly"""
    # # tarFt = '大客車'
    # def scatter_plotly_set(df2D, tarFt='責任險', sample_frac=0.05, return_div=True, drop_zero=None):
    #     """
    #     輸入一個當作基底的特徵 > 拿其他特徵和基底特徵組合 > 找出2D & 3D 的top5 >  畫2D & 3D的散布圖 > subplot的title表示法人不在自然人Pr80範圍內的比例
    #     """
    #     matchMask = pd.concat([df2D['ft'].apply(lambda x: x[i]).str.match(f'.*sum.*{tarFt}') for i in range(2)], axis=1)
    #     match = matchMask.any(axis=1)
    #     tarTops_2D = df2D[match].sort_values('law', ascending=False).head(5).copy()
    #     matchIdx = np.where(matchMask.loc[tarTops_2D.index])
    #     tarTops_2D.reset_index(inplace=True)
    #     # 把tarFt移到組合的第一個  (作圖時就一定會在x軸)
    #     for r, i, ft in zip(matchIdx[0], matchIdx[1], tarTops_2D['ft']):
    #         ft = list(ft)
    #         tarTops_2D['ft'].iloc[r] = [ft[i]] + ft[:i] + ft[i+1:]
    #
    #     matchMask = pd.concat([df3D['ft'].apply(lambda x: x[i]).str.match(f'.*sum.*{tarFt}') for i in range(3)], axis=1)
    #     match = matchMask.any(axis=1)
    #     tarTops_3D = df3D[match].sort_values('law', ascending=False).head(5).copy()
    #     matchIdx = np.where(matchMask.loc[tarTops_3D.index])
    #     tarTops_3D.reset_index(inplace=True)
    #     # 把tarFt移到組合的第一個  (作圖時就一定會在x軸)
    #     for r, i, ft in zip(matchIdx[0], matchIdx[1], tarTops_3D['ft']):
    #         ft = list(ft)
    #         tarTops_3D['ft'].iloc[r] = [ft[i]] + ft[:i] + ft[i+1:]
    #
    #     fig = make_subplots(rows=2, cols=5,
    #                         specs=[[{'type': 'xy'} for _ in range(5)],
    #                                [{'type': 'scene'} for _ in range(5)]],
    #                         subplot_titles=[f'{i:.1%}' for i in tarTops_2D['law']]+[f'{i:.1%}' for i in tarTops_3D['law']],
    #                         shared_yaxes=True
    #                         )
    #     # 2D
    #     for c, row in tarTops_2D.iterrows():
    #         arr = X_train_scaled[list(row['ft'])]
    #         arrLaw, arrNature = arr[y_train==1], arr[y_train==0]
    #         if drop_zero == 'any':
    #             arrLaw = arrLaw[(arrLaw > 0).any(axis=1)]
    #             arrNature = arrNature[(arrNature > 0).any(axis=1)]
    #         elif drop_zero == 'all':
    #             arrLaw = arrLaw[(arrLaw > 0).all(axis=1)]
    #             arrNature = arrNature[(arrNature > 0).all(axis=1)]
    #
    #         showlegend=False if c != 0 else True
    #         fig.add_trace(
    #             go.Scatter(x=arrLaw.iloc[:, 0].sample(frac=sample_frac).values, y=arrLaw.iloc[:, 1].sample(frac=sample_frac).values, name='法人', legendgroup='法人', mode='markers', marker=dict(color='red'), opacity=0.5, showlegend=showlegend),
    #             # go.Histogram2dContour(x=arrLaw.iloc[:, 0], y=arrLaw.iloc[:, 1], colorscale='Reds', name='法人', legendgroup='法人', opacity=0.5),
    #             row=1, col=c+1
    #         )
    #         fig.add_trace(
    #             go.Scatter(x=arrNature.iloc[:, 0].sample(frac=sample_frac).values, y=arrNature.iloc[:, 1].sample(frac=sample_frac).values, name='自然人', legendgroup='自然人', mode='markers', marker=dict(color='blue', symbol='circle-open'), line=dict(color='blue', width=2), opacity=0.5, showlegend=showlegend),
    #             # go.Histogram2dContour(x=arrNature.iloc[:, 0], y=arrNature.iloc[:, 1], colorscale='Blues', name='自然人', legendgroup='自然人', opacity=0.5),
    #             row=1, col=c+1
    #         )
    #         # axis labels
    #         fig['layout'][f'xaxis{c+1}']['title']=re.search('-(\w{1,})-', arrLaw.columns[0]).group(1)
    #         fig['layout'][f'yaxis{c+1}']['title']=re.search('-(\w{1,})-', arrLaw.columns[1]).group(1)
    #         # 包含80%資料的範圍
    #         if not arrNature.empty:
    #             center = arrNature.mean().values
    #             d = pairwise_distances([center], arrNature)[0]
    #             prX = np.quantile(d, .8)
    #             fig.add_shape(type="circle",xref="x", yref="y",
    #                 x0=center[0]-prX, y0=center[1]-prX,
    #                 x1=center[0]+prX, y1=center[1]+prX,
    #                 opacity=0.8, line_color="#3a238d",
    #                 row=1, col=c+1
    #             )
    #         if not arrLaw.empty:
    #             center = arrLaw.mean().values
    #             d = pairwise_distances([center], arrLaw)[0]
    #             prX = np.quantile(d, .8)
    #             fig.add_shape(type="circle", xref="x", yref="y",
    #                 x0=center[0]-prX, y0=center[1]-prX,
    #                 x1=center[0]+prX, y1=center[1]+prX,
    #                 opacity=0.8, line_color="#cc0e0a",
    #                 row=1, col=c+1
    #             )
    #     # 3D
    #     for c, row in tarTops_3D.iterrows():
    #         arr = X_train_scaled[list(row['ft'])]
    #         arrLaw, arrNature = arr[y_train==1], arr[y_train==0]
    #         if drop_zero == 'any':
    #             arrLaw = arrLaw[(arrLaw > 0).any(axis=1)]
    #             arrNature = arrNature[(arrNature > 0).any(axis=1)]
    #         elif drop_zero == 'all':
    #             arrLaw = arrLaw[(arrLaw > 0).all(axis=1)]
    #             arrNature = arrNature[(arrNature > 0).all(axis=1)]
    #
    #         fig.add_trace(
    #             go.Scatter3d(x=arrLaw.iloc[:, 0].sample(frac=sample_frac).values,
    #                        y=arrLaw.iloc[:, 1].sample(frac=sample_frac).values,
    #                        z=arrLaw.iloc[:, 2].sample(frac=sample_frac).values,
    #                        name='法人', legendgroup='法人', mode='markers', marker=dict(color='red', size=3), opacity=0.5, showlegend=False),
    #             # go.Histogram2dContour(x=arrLaw.iloc[:, 0], y=arrLaw.iloc[:, 1], colorscale='Reds', name='法人', legendgroup='法人', opacity=0.5),
    #             row=2, col=c+1
    #         )
    #         fig.add_trace(
    #             go.Scatter3d(x=arrNature.iloc[:, 0].sample(frac=sample_frac).values,
    #                        y=arrNature.iloc[:, 1].sample(frac=sample_frac).values,
    #                        z=arrNature.iloc[:, 2].sample(frac=sample_frac).values,
    #                        name='自然人', legendgroup='自然人', mode='markers', marker=dict(color='blue', size=3, symbol='circle-open'), line=dict(color='blue', width=2), opacity=0.5, showlegend=False),
    #             # go.Histogram2dContour(x=arrNature.iloc[:, 0], y=arrNature.iloc[:, 1], colorscale='Blues', name='自然人', legendgroup='自然人', opacity=0.5),
    #             row=2, col=c+1
    #         )
    #         # axis labels
    #         fig['layout'][f'scene{c+1}']['xaxis_title']=re.search('-(\w{1,})-', arrLaw.columns[0]).group(1)
    #         fig['layout'][f'scene{c+1}']['yaxis_title']=re.search('-(\w{1,})-', arrLaw.columns[1]).group(1)
    #         fig['layout'][f'scene{c+1}']['zaxis_title']=re.search('-(\w{1,})-', arrLaw.columns[2]).group(1)
    #     pyo.plot(fig, auto_open=False)
    #
    #     if return_div:
    #         return pyo.plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)
    #     else:
    #         return pyo.plot(fig, auto_open=False)
    #
    # def html_plot(div, name):
    #     return f'''
    #     <h1 id=Loss class=hd1>
    #         <hr>{name}
    #         <hr>
    #     </h1>
    #     <div style="height:90%; background-color:#F5B0A1;">
    #         '''+div+'''
    #     </div>
    #     '''
    #
    #
    #
    #
    #
    # htmlString ='''
    #     <html>
    #         <head>
    #             <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    #             <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
    #
    #             <style>
    #             .hd1 {
    #                 text-align: center;
    #                 background-color:#ADE1F2;
    #             }
    #             </style>
    #         </head>
    #         <body>
    #     '''
    # feats = X_train.filter(like='iins_sum').columns
    # for ft in feats:
    #     ft = re.search('-(\w{1,})-', ft).group(1)
    #     fig = scatter_plotly_set(df2D, df3D, tarFt=ft)
    #     div = html_plot(fig, ft)
    #     htmlString += div
    #
    #
    #     htmlString +='''
    #         </body>
    #     </html>
    #     '''
    # with open('test_iins.html', 'w', encoding='utf-8') as f:
    #     f.write(htmlString)
