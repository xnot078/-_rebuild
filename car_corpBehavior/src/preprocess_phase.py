import pandas as pd
import os, glob, tqdm, datetime

from typing import List

from sklearn.preprocessing import LabelEncoder

"""
讀取raw DATA:
'ply_relation', 'policy', 'ply_ins_type'
"""
# 要用的欄位: 分成特徵欄位 & 數值欄位 (之所以指定欄位，是要在讀取的時候減少記憶體的使用)
feats, vals = pd.read_excel('./car_corpBehavior/data/code/車險欄位.xlsx', sheet_name=None).values()
featsCols = {k: g['column'].to_list() for k, g in feats.groupby('table')}
valsCols = {k: g['column'].to_list() for k, g in vals.groupby('table')}
# usecols = {'feats':featsCols, 'vals':valsCols}

def ibrand_fix( df):
    df['ibrand_h02'] = df['ibrand'].str.slice(0, 2)
    df['ibrand_h24'] = df['ibrand'].str.slice(2, 4)
    return df

# table_name = "ply_relation"
def load_data(table_name:str="policy", raw_data_dir="./car_corpBehavior/data/raw/2022/*/*"):
    # basic setting
    used_cols = {
                    "policy": ['ipolicy', 'fassured', 'iassured', 'itag', 'iroute'],
                    "ply_relation": ['ipolicy', 'fcard_class', 'ibrand', 'dpolicy'], #, 'dply_begin', 'dply_end': 對齊2022定義，年度用簽單日
                    "ply_ins_type": ['ipolicy', 'mins_premium', 'iins_type']
                }

    if table_name == "policy":
        iroute_code = pd.read_csv('./car_corpBehavior/data/code/policy.iroute.csv', dtype=str)
        iroute_code = iroute_code.set_index('code')['name'].to_dict()
        iroute_code = {
                        ir: name
                        for ir, name in iroute_code.items()
                        if name in ['經銷商', '保經代公司', '網路', '員工自行招攬',
                                      '車商業務員', '車行', '貨運行', '客運公司', '計程車行', '吉時保']
                        }
    if table_name == "ply_relation":
        car_type = pd.read_csv('./car_corpBehavior/data/code/ply_relation.ibrand_h4.csv', dtype=str)
        car_type.columns = ["ibrand_h02", "ibrand_h24", "車種", "車系"]
        car_type.dropna(subset=["ibrand_h02", "ibrand_h24"], inplace=True)
        car_type.drop("車系", axis=1, inplace=True)

    if table_name == "ply_ins_type":
        iins_type = pd.read_csv('./car_corpBehavior/data/code/ply_ins_type_from小豆.csv', dtype=str)
        iins_type.drop("商品名稱", axis=1, inplace=True)
        iins_type["險別"] = iins_type["險別"].str.replace("金融機構從業人員汽車綜合保險", "others")
        iins_type["險別"] = iins_type["險別"].str.replace('.*機車.*', "機車相關", regex=True)
        iins_type.columns = ["險別", "iins_type"]

    # load data paths
    # table_name = "ply_ins_type"
    paths = glob.glob(f'{raw_data_dir}/{table_name}.txt')

    # load data
    data = []
    for p in tqdm.tqdm(paths):
        tem = pd.read_csv(p, sep='|', usecols=used_cols.get(table_name), dtype=str)

        if "fassured" in tem.columns:
            tem["fassured"][tem["fassured"]!="2"] = 0
            tem["fassured"][tem["fassured"]=="2"] = 1

        if "iroute" in tem.columns:
            tem["iroute_name"] = tem["iroute"].fillna('')\
                                    .str.slice(0, 2)\
                                    .apply(lambda x: iroute_code.get(x, "others"))
            tem.drop("iroute", axis=1, inplace=True)
            # le_iroute = LabelEncoder()
            # policy['irouteEncoded'] = le_iroute.fit_transform(tem['iroute_name'])
        if "dpolicy" in tem.columns:
            tem["dpolicy"] = pd.to_datetime(tem["dpolicy"]).dt.year
            this_year = datetime.datetime.today().year
            tem["dply_year_window"] = (this_year - tem["dpolicy"]) // 5 # 第幾個近五年?

        if "ibrand" in tem.columns:
            tem = ibrand_fix(tem)
            tem = tem.merge(
                            car_type, on=["ibrand_h02", "ibrand_h24"], how="left"
                            )
            tem["icar_type"] = tem["車種"].fillna('others').str.replace('曳引車|小型特種車|大型特種車', 'others', regex=True)
            tem.drop(["ibrand", "ibrand_h02", "ibrand_h24", "車種"], axis=1, inplace=True)

        # if "icar_type" in tem.columns:
        #     tem["icar_type_class"] = tem["icar_type"]

        if "iins_type" in tem.columns:
            tem["mins_premium"] = tem["mins_premium"].astype(int)
            tem = tem.merge(
                    iins_type, on="iins_type"
                    )
            tem.columns = ["iins_type_name" if c=="險別" else c for c in tem.columns]

        data.append(
            tem
        )

    data = pd.concat(data).reset_index(drop=True)
    if table_name != "ply_ins_type":
        # 202207的資料部分與202104重疊，所以要去重複
        data.drop_duplicates(subset="ipolicy", inplace=True)
    else:
        # ply_ins_type，ipolicy會有重複的，所以要加上險種 & 保費一起篩
        data.drop_duplicates(subset=["ipolicy", "iins_type", "mins_premium"], inplace=True)

    return data

def process_load():
    ply = load_data("policy")
    rel = load_data("ply_relation") # ipolicy: uq
    iins = load_data("ply_ins_type") # ipolicy+iins: uq
    # 其實有點怪怪的，因為iins的ipolicy+iins_type才不會重複。應該要by險種的和另外兩個分開groupby
    # 但為了與過去對齊，先照舊
    df_m = iins.merge(
            ply, on="ipolicy", how="left"
        ).merge(
            rel, on="ipolicy", how="left"
        )

    # 篩選近五年的資料
    return df_m.query(" dply_year_window == 0 ")

# by車統計
def by_car(data, val_tag="iins_type_name", idx="dpolicy"):
    """
    以:val_tag(例如"險種")為agg條件，並計算:idx(例如"年")的保單數 & 保費平均
    """
    g = data.groupby(["iassured", "fassured", "itag", val_tag, idx]).agg({
            "ipolicy": "nunique",
            "mins_premium": "sum"
            })
    g_yearAvg = g.pivot_table(index=["iassured", "fassured", "itag"], columns=val_tag, aggfunc="mean")
    return g_yearAvg

def process_agg(data):
    # by Car
    byCar_iins = by_car(data, "iins_type_name")
    byCar_carType = by_car(data, "icar_type")
    byCar_iroute = by_car(data, "iroute_name")

    # by Person
    byP_iins = byCar_iins.groupby(["iassured", "fassured"]).mean()
    byP_carType = byCar_carType.groupby(["iassured", "fassured"]).mean()
    byP_iroute = byCar_iroute.groupby(["iassured", "fassured"]).mean()

    # 整理column names & 合併
    ply_data = [byP_iins["ipolicy"], byP_carType["ipolicy"], byP_iroute["ipolicy"]]
    for i, val_tag in enumerate(["ply_iins_mean_", "ply_carType_mean_", "ply_iroute_mean_"]):
        ply_data[i].columns = [val_tag+c for c in ply_data[i].columns]

    mprem_data = [byP_iins["mins_premium"], byP_carType["mins_premium"], byP_iroute["mins_premium"]]
    for i, val_tag in enumerate(["mprem_iins_mean_", "mprem_carType_mean_", "mprem_iroute_mean_"]):
        mprem_data[i].columns = [val_tag+c for c in mprem_data[i].columns]
    data = pd.concat(ply_data+mprem_data, axis=1)
    data.reset_index(drop=False, inplace=True)

    return data

def full_process():
    data = process_agg(process_load())
    return data

def update_data(path="./car_corpBehavior/data/medium_pivot/input_data.parq", return_data=False):
    data = full_process()
    data.to_parquet(path, compression="brotli")
    if return_data:
        return data

if __name__ == "__main__":
    data = update_data(return_data=True)


"""
    -依險別: 近五年  Avg.每車年均投入保費(&保單數)
        iins_sum(count)_mean-name-year: name=險種類別、year=第幾個近五年
    -依車種: 近五年  Avg.每車年均投入保費(&保單數)
        cartype_sum(count)_mean-name-year: name=車種、year=第幾個近五年
    -依業務來源: 近五年  Avg.每車年均投入保費(&保單數)
        iroute_sum(count)_mean-name-year: name=車種、year=第幾個近五年
"""
