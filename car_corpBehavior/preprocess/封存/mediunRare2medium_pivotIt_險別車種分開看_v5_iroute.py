import pandas as pd
from myfunc import preprocessing
import re, os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

print(
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    This program is aim at producing pivot_table from raw data. (.parq format)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
)

"""
目標B組合: 整理出需要的特徵

group by:
1. (商用/自用) or (車種) 車 "plyRelation.車種" -> 人 "policy.iassured"
--
    車: by: "policy.itag"
        - 年, 附加險 ply_ins_type.iins_type   "plyInsType.iins_type & plyInsType.iins_type_name"
        val:
        - 保費、保單 ply_ins_type.mins_premium "plyInsType.mins_premium"
        不by:
        附加險種類 ply_ins_type.iins_type.nunique
==>
- 每個人，車子的數值
    1~5年 平均 & 最大 "plyRelation.dply_begin"
    6~10年 平均 & 最大
- 每年，總保單、總保費、不重複車牌
    1~5年 平均 & 最大
    6~10年 平均 & 最大

每台車 每年 有多少保單 & 保費
-> 每台"車" 依時間區間("2011-2016 & "近五年") 平均/最多有幾張保單 & 保費
-> 每個"人" 依時間區間 & 依車種 平均/最多 有幾張保單 & 保費

* 險種有124種 --(如果換成險別)--> 8種
* 車種9種

--------------------------------------------------------
note:
plyInsType: 如果一張保單有多個附加險，就會有多筆紀錄。mins_premiun也在這裡。 iins_type	mins_premium	mpure_prem
plyRelation: fcard_class強制險任意別、車種、保單日期、業務來源; ipolicy, ply_begin, iroute, 車種
policy: ipolicy, fassured, iassured, itag車牌
"""

#
print('loading ./data/medium_rare/policy.parq', end='   ')
policy = pd.read_parquet('./data/medium_rare/policy.parq', columns=['ipolicy','fassured','iassured', 'itag', 'iroute']) #,'fpt','fapplicant','iapplicant'
print('ok')
iroute = pd.read_csv('./data/code/policy.iroute.csv', dtype=str)
iroute = iroute.set_index('code')['name'].to_dict()
iroute_target = ['經銷商', '保經代公司', '網路', '員工自行招攬', '車商業務員', '車行', '貨運行', '客運公司', '計程車行', '']
iroute = {k:v if v in iroute_target else 'others' for k, v in iroute.items()} # 只保留"體系內車商保代" (和法人高度相關)
policy['iroute_name'] = policy['iroute'].apply(lambda x: iroute.get(x[:2], 'others') if isinstance(x, str) else 'others')
le_iroute = LabelEncoder()
policy['irouteEncoded'] = le_iroute.fit_transform(policy['iroute_name'])

# policy[policy['iassured'] == '60f636703ef7f170a89a1bab651172f25c32c358']
policy['ipolicy'].value_counts()
#
print('loading ./data/medium_rare/ply_relation.parq', end='   ')
plyRelation = pd.read_parquet('./data/medium_rare/ply_relation.parq', columns=['ipolicy', 'fcard_class', 'ibrand', 'dply_begin', 'dply_end'])#, 'ilast_ply', 'inext_ply', 'icar_type', 'icar_type_class'

car_type = pd.read_csv('./data/code/ply_relation.ibrand_h4.csv', dtype=str)
car_type.columns = ['ibrand_h1-2', 'ibrand_h3-4', '車種', '車系']
car_type = car_type.dropna(subset=['ibrand_h1-2', 'ibrand_h3-4'])

plyRelation['ibrand_h1-2'] = plyRelation['ibrand'].str.slice(0, 2)
plyRelation['country'] = plyRelation['ibrand_h1-2'].apply(lambda x: re.search('\((\w+)\)$', str(x)).group(1) if re.search('\(\w+\)', str(x)) else None)
plyRelation['ibrand_h3-4'] = plyRelation['ibrand'].str.slice(2, 4)
plyRelation = plyRelation.merge(car_type, on=['ibrand_h1-2', 'ibrand_h3-4'], how='left')
plyRelation['車種'] = plyRelation['車種'].fillna('others').str.replace('曳引車|小型特種車|大型特種車', 'others')
le_carType = LabelEncoder()
plyRelation['carTypeEncoded'] = le_carType.fit_transform(plyRelation['車種'])


# 整理需要的特徵
# top5Resource = plyRelation['iroute'].value_counts(normalize=True)[:5].index
# iroute = {k:v if k=='1' else 'others' for k, v in iroute.items()} # 只保留"體系內車商保代" (和法人高度相關)
# plyRelation['iroute_name'] = plyRelation['iroute'].apply(lambda x: iroute.get(x, 'others'))
print('ok')

#
print('loading ./data/medium_rare/ply_ins_type.parq', end='   ')
plyInsType = pd.read_parquet('./data/medium_rare/ply_ins_type.parq')
iins_type = pd.read_csv('./data/code/ply_ins_type_from小豆.csv', dtype=str)
iins_type = iins_type[~iins_type['險種代號'].isna()].set_index('險種代號')

iins_type['險別'] = iins_type['險別'].str.replace('金融機構從業人員汽車綜合保險', '其他')
iins_type['險別'] = iins_type['險別'].str.replace('.*機車.*', '機車相關')
iins_type = iins_type['險別'].to_dict()
plyInsType['iins_type_name'] = plyInsType['iins_type'].apply(lambda x: iins_type.get(x, None))
plyInsType['iins_type_name'].fillna('others', inplace=True)
le_insType = LabelEncoder()
plyInsType['iinsTypeEncoded'] = le_insType.fit_transform(plyInsType['iins_type_name'])

# plyInsType['iins_type_name'].fillna('其他').value_counts(normalize=True)
print('ok')
#
policy.head(1)
plyRelation.head(1)
plyInsType.head(1)

print('merge...', end='    ')
partA_1 = policy[['ipolicy', 'fassured', 'iassured', 'itag', 'iroute_name']]#, 'iroute', iroute_name, irouteEncoded
partA_2 = plyRelation[['ipolicy', '車種']] # 'dply_begin', '車種', 'carTypeEncoded'
partA = partA_1.merge(partA_2, on='ipolicy', how='inner')
partB = plyInsType[['ipolicy', 'mins_premium', 'year', 'iins_type_name']] #'iins_type_name', 'iinsTypeEncoded'

# plyInsType.columns.values
dfM = partB.merge(partA, on='ipolicy', how='left')
# dfM.to_csv('./data/medium_pivot/dfM.csv')
for i in range(5):
    tem = dfM.iloc[i*((len(dfM) // 5)):(i+1)*(len(dfM) // 5), :]
    tem.to_parquet(f'./data/medium_pivot/dfM_part{i}.parq', compression='brotli')
print('ok')

################################################
fileRoot = './data/medium_pivot'
files = [fileRoot+'/'+f for f in os.listdir(fileRoot) if f.startswith('dfM')]
holder = []
for i in files:
    holder.append(pd.read_parquet(i))
dfM = pd.concat(holder, axis=0)#pd.read_parquet('./data/medium_pivot/dfM.parq')
	# ipolicy	iins_type	mins_premium	mpure_prem	year	keep	iins_type_name	fassured	iassured	itag	dply_begin	iroute	車種
del holder

dfM['iroute_name'].value_counts(normalize=True)
dfM['車種'].value_counts(normalize=True)
t = dfM.groupby('iassured')['ipolicy'].apply(lambda x: x.max())
t.to_csv('./data/tem/byPersonIpolicy.csv')

"""
每台車 每年 有多少保單 & 保費
-> 每台"車" 依時間區間("2011-2016 & "近五年") & 依險種    [每年 平均/最多 有幾張保單 & 保費]
-> 每個"人" 依時間區間 & 依險種 & 依車種                  [每年 平均/最多 有幾張保單 & 保費]

#太厲害了，每年7,8千台不重複車輛投保
dfM[(dfM['iassured'] == 'bGYOMb3v3Dt6rQ') & (dfM['year_tag'] == 0) & (dfM['iins_type']=='21')].groupby('year')['itag'].count()

"""
print('by car...', end='    ')
dfM['year_tag'] = dfM['year'].apply(lambda x: 0 if x <= '2016' else 1)

"""
- 每台車的"年度"統計
欄位說明:
車輛t(包含資種資訊) [險別i, 年度y] 的保費總和(sum) & 保單數(count)
"""
gpByCar_iins = dfM.groupby(['itag', 'iins_type_name', 'fassured', 'iassured', 'year', 'year_tag'])['mins_premium'].agg(['sum', 'count'])
gpByCar_iins.columns = [f'iins_{c}' for c in gpByCar_iins.columns]

gpByCar_cartype = dfM.groupby(['itag', '車種', 'fassured', 'iassured', 'year', 'year_tag'])['mins_premium'].agg(['sum', 'count'])
gpByCar_cartype.columns = [f'cartype_{c}' for c in gpByCar_cartype.columns]

gpByCar_iroute = dfM.groupby(['itag', 'iroute_name', 'fassured', 'iassured', 'year', 'year_tag'])['mins_premium'].agg(['sum', 'count'])
gpByCar_iroute.columns = [f'iroute_{c}' for c in gpByCar_iroute.columns]

"""
- 每台車的"時間區間"統計
欄位說明:
車輛t(包含資種資訊) [險別i, "時間區間t"] 的 (目前時間區間為'2011-2016' & '2016-2021', 五年一劃分note:2021只到2021.04)
sum_mean: 該區間(五年)的年均保費
sum_max: 該區間(五年)最高的年度保費

count_mean: 該區間(五年)的年均保單數
count_max: 該區間(五年)最高的年度保單數
"""
gpByCar_iins_YTag = gpByCar_iins.groupby(['itag', 'iins_type_name', 'fassured', 'iassured', 'year_tag'])['iins_sum', 'iins_count'].agg(['mean', 'max'])
gpByCar_iins_YTag.columns = ['_'.join(i) for i in gpByCar_iins_YTag.columns]
gpByCar_iins_YTag.reset_index(inplace=True)


gpByCar_cartype_YTag = gpByCar_cartype.groupby(['itag', '車種', 'fassured', 'iassured', 'year_tag'])['cartype_sum', 'cartype_count'].agg(['mean', 'max'])
gpByCar_cartype_YTag.columns = ['_'.join(i) for i in gpByCar_cartype_YTag.columns]
gpByCar_cartype_YTag.reset_index(inplace=True)

gpByCar_iroute_YTag = gpByCar_iroute.groupby(['itag', 'iroute_name', 'fassured', 'iassured', 'year_tag'])['iroute_sum', 'iroute_count'].agg(['mean', 'max'])
gpByCar_iroute_YTag.columns = ['_'.join(i) for i in gpByCar_iroute_YTag.columns]
gpByCar_iroute_YTag.reset_index(inplace=True)
print('ok')

# 排除離群值
def iforest(df):
    iso = IsolationForest(n_jobs=-1, contamination=0.01)
    tt = df.filter(regex='.*sum|count.*')
    yhat = iso.fit_predict(tt)
    return yhat
gpByCar_iins_YTag['iins_outlier'] = iforest(gpByCar_iins_YTag)
gpByCar_cartype_YTag['cartype_outlier'] = iforest(gpByCar_cartype_YTag)
gpByCar_iroute_YTag['iroute_outlier'] = iforest(gpByCar_iroute_YTag)

gpByCar_iins_YTag['iins_outlier'].value_counts(normalize=True)
gpByCar_cartype_YTag['cartype_outlier'].value_counts(normalize=True)
gpByCar_iroute_YTag['iroute_outlier'].value_counts(normalize=True)

gpByCar_iins_YTag = gpByCar_iins_YTag[gpByCar_iins_YTag['iins_outlier']!=-1]
gpByCar_cartype_YTag =  gpByCar_cartype_YTag[gpByCar_cartype_YTag['cartype_outlier']!=-1]
gpByCar_iroute_YTag = gpByCar_iroute_YTag[gpByCar_iroute_YTag['iroute_outlier']!=-1]

print('by person...', end='    ')
"""
- 每個客戶的"時間區間"統計
欄位說明:
客戶c [險別i, 車種k, 時間區間t] 的統計

sum_mean-險別-車種-時間區間: 客戶c在該險別、車種、時間區間的 平均(每台車的年均保費) >>> 在時間區間內，平均每年願意在險別、車種投入的保費
sum_max-險別-車種-時間區間: 客戶c在該險別、車種、時間區間的 平均(每台車的最高年度保費) >>> 在時間區間內，平均每年願意在險別、車種投入的最高保費

count_mean-險別-車種-時間區間: 客戶c在該險別、車種、時間區間的 平均(每台車的年均保單數) >>> 在時間區間內，平均每年願意在險別、車種投入的保單數
count_max-險別-車種-時間區間: 客戶c在該險別、車種、時間區間的 平均(每台車的最高年度保單數) >>> 在時間區間內，平均每年願意在險別、車種投入的最高保單數
"""
gpByCar_iins_YTag[(gpByCar_iins_YTag['iassured'] == '083GeOQ058U406r9y') & (gpByCar_iins_YTag['year_tag'] == 0)]
gpByPerson_iins = gpByCar_iins_YTag.groupby(['fassured','iassured','year_tag', 'iins_type_name'])[['iins_sum_mean', 'iins_sum_max', 'iins_count_mean', 'iins_count_max']].mean()
gpByPerson_iins_carNum = gpByCar_iins_YTag.groupby(['fassured','iassured','year_tag', 'iins_type_name'])['itag'].count()
gpByPerson_iins_carNum.name = 'iins_carQt'
gpByPerson_iins = gpByPerson_iins.merge(gpByPerson_iins_carNum, on=['fassured','iassured','year_tag', 'iins_type_name'])

gpByPerson_cartype = gpByCar_cartype_YTag.groupby(['fassured','iassured','year_tag', '車種'])[['cartype_sum_mean', 'cartype_sum_max', 'cartype_count_mean', 'cartype_count_max']].mean()
gpByPerson_cartype_carNum = gpByCar_cartype_YTag.groupby(['fassured','iassured','year_tag', '車種'])['itag'].count()
gpByPerson_cartype_carNum.name = 'cartype_carQt'
gpByPerson_cartype = gpByCar_cartype_YTag.merge(gpByPerson_cartype_carNum, on=['fassured','iassured','year_tag', '車種'])

gpByPerson_iroute = gpByCar_iroute_YTag.groupby(['fassured','iassured','year_tag', 'iroute_name'])[['iroute_sum_mean', 'iroute_sum_max', 'iroute_count_mean', 'iroute_count_max']].mean()
gpByPerson_iroute_carNum = gpByCar_iroute_YTag.groupby(['fassured','iassured','year_tag', 'iroute_name'])['itag'].count()
gpByPerson_iroute_carNum.name = 'iroute_carQt'
gpByPerson_iroute = gpByCar_iroute_YTag.merge(gpByPerson_iroute_carNum, on=['fassured','iassured','year_tag', 'iroute_name'])
print('save...', end='    ')
print('ok')

print('pivot_table...', end='    ')
ptByPerson_iins = gpByPerson_iins.pivot_table(index=['iassured', 'fassured'], columns=['iins_type_name', 'year_tag'], aggfunc='mean')
ptByPerson_iins.columns = ['-'.join([str(i) for i in c]) for c in ptByPerson_iins.columns]

ptByPerson_cartype = gpByPerson_cartype.pivot_table(index=['iassured', 'fassured'], columns=['車種', 'year_tag'], aggfunc='mean')
ptByPerson_cartype.columns = ['-'.join([str(i) for i in c]) for c in ptByPerson_cartype.columns]

ptByPerson_iroute = gpByPerson_iroute.pivot_table(index=['iassured', 'fassured'], columns=['iroute_name', 'year_tag'], aggfunc='mean')
ptByPerson_iroute.columns = ['-'.join([str(i) for i in c]) for c in ptByPerson_iroute.columns]

ptByPerson_iins.shape
ptByPerson_cartype.shape
ptByPerson_iroute.shape
ptByPerson = ptByPerson_iins.merge(ptByPerson_cartype, how='left', on=['iassured', 'fassured']).merge(ptByPerson_iroute, how='left', on=['iassured', 'fassured'])

ptByPerson.reset_index('fassured', inplace=True)
ptByPerson['fassured'].value_counts(normalize=True)
ptByPerson.to_parquet('./data/medium_pivot/ptByPerson_險別車種分開_v5.parq', compression='brotli')
print('save...', end='    ')
print('ok')


"""範例"""
dfM[dfM['iassured'] == '083GeOQ05BF400k5a']

t = gpByCar_iins.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year']>='2016')]

t = gpByCar_cartype.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year']>='2016')]

t = gpByCar_iroute.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year']>='2016')]

t = gpByCar_iins_YTag.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year_tag']==1)]

t = gpByCar_cartype_YTag.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year_tag']==1)]

t = gpByCar_iroute_YTag.reset_index()
t[(t['iassured'] == '083GeOQ05BF400k5a') & (t['year_tag']==1)]

t = ptByPerson_iins.reset_index().filter(regex='.*-1|iassured')
t = t.filter(regex='^(?!.*others|.*其他|.*max|.*carQt|.*outlier).*')
t[(t['iassured'] == '083GeOQ05BF400k5a')]
.shape


t = ptByPerson_cartype.reset_index().filter(regex='.*-1|iassured')
t = t.filter(regex='^(?!.*others|.*其他|.*max|.*carQt|.*outlier).*')
t[(t['iassured'] == '083GeOQ05BF400k5a')]
.shape


t = ptByPerson_iroute.reset_index().filter(regex='.*-1|iassured')
t = t.filter(regex='^(?!.*others|.*其他|.*max|.*carQt|.*outlier).*')
t[(t['iassured'] == '083GeOQ05BF400k5a')]


ptByPerson.columns
ptByPerson.filter(like='count')




ptByPerson['iins_count_max-責任險-1'] - ptByPerson['iins_count_mean-責任險-1']
