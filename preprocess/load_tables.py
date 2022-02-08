import os, re
import pandas as pd
from collections import defaultdict


def load_tables_ibrand_fix(self, df):
    df['ibrand_h2'] = df['ibrand'].str.slice(0, 2)
    df['ibrand_h4'] = df['ibrand'].str.slice(0, 4)
    df['ibrand_h8'] = df['ibrand'].str.slice(0, 8)
    return df


class from_google_sheet():
    def __init__(self):
        self.labels, self.usecols = None, None

        feats, vals = pd.read_excel('./data/code/車險欄位.xlsx', sheet_name=None).values()
        featsCols = {k: g['column'].to_list() for k, g in feats.groupby('table')}
        valsCols = {k: g['column'].to_list() for k, g in vals.groupby('table')}
        self.usecols = {'feats':featsCols, 'vals':valsCols}

        table_label = pd.read_excel('./data/code/車險欄位.xlsx', sheet_name=None)
        for name, sh in table_label.items():
            sh.columns = range(len(sh.columns))
            self.labels = {k: g[1].to_list() for k, g in sh.groupby(0)}

    def load_tables_ibrand_fix(self, df):
        df['ibrand_h2'] = df['ibrand'].str.slice(0, 2)
        df['ibrand_h4'] = df['ibrand'].str.slice(0, 4)
        df['ibrand_h8'] = df['ibrand'].str.slice(0, 8)
        return df

    def load_tables(self, dataDir, yearlist=None, tables=['ply_relation', 'policy', 'ply_ins_type']):
        self.tables = defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(dataDir):
            print(filenames)
            year, keep = None, None
            if mat := re.search(r'[0-9]+', dirpath.split('\\')[-1]):
                year = mat.group(0)
                ######
                if isinstance(yearlist, list) and int(year) not in yearlist:
                    continue #因為資料太大，指定年分讀取(yearlist)，不在yearlist裡就跳過迴圈，這樣寫其實滿慘的
                #####
            if mat := re.search(r'[a-zA-Z]+', dirpath.split('\\')[-1]):
                keep = mat.group(0)

            for file in filenames:
                # print(os.path.splitext(file)[0])
                if os.path.splitext(file)[0] not in tables:
                    continue
                else:
                    print(file)

                name, ext = os.path.splitext(file)
                if (ext == '.txt') & ((name in self.usecols['feats'].keys()) | (name in self.usecols['vals'].keys())):
                    print('.', end='')
                    # dtype = {
                    #     **{c:'str' for c in usecols['feats'].get(name, [])},
                    #     **{c:'float' for c in usecols['vals'].get(name, [])}
                    # }
                    # dtype 部分row資料有誤要跳過，直接用dtype會出錯
                    # t = pd.read_csv(f'{dirpath}/{file}', sep='|', dtype=dtype)
                    usecolsTem = self.usecols['feats'].get(name, []) + self.usecols['vals'].get(name, [])
                    # print(usecolsTem)
                    t = pd.read_csv(f'{dirpath}/{file}', sep='|', dtype=str)
                    print(f'缺失欄位 {name}: {set(usecolsTem) - set(t.columns)}')
                    t = t[t.columns.intersection(usecolsTem)]
                    for c in self.usecols['vals'].get(name, []):
                        if c in t.columns:
                            mask = t[c].str.match('^[\d.]+$').fillna(False)
                            t = t[mask]
                            t[c] = t[c].astype(float)
                    if 'ibrand' in t.columns:
                        t = self.load_tables_ibrand_fix(t)
                    if 'icar_type' in t.columns:
                        t['icar_type_class'] = t['icar_type']
                    # 替換代碼類col
                    if interCols := t.columns.intersection(self.labels.keys()).to_list():
                        for ic in interCols:
                            t[ic] = t[ic].apply(lambda x: self.labels[ic].get(x, x))
                    t['year'] = year
                    t['keep'] = keep
                    self.tables[name].append(t)
        print('done')
        self.tables = {k: pd.concat(v) for k, v in self.tables.items()}
        return self.tables

if __name__ == '__main__':

    used_tables = ['ply_ins_type', 'ply_relation', 'policy']

    df_feats = pd.read_excel('./data/code/車險欄位.xlsx', sheet_name=None)
    des_cols, val_cols = df_feats['feats'], df_feats['vals'] # 要使用的描述型特徵 & 數值型特徵
    # 把藥使用的cols依table名稱整理
    used_des_cols = {table:g['column'].to_list() for table, g in des_cols[['column', 'table']].groupby('table')}
    used_val_cols = {table:g['column'].to_list() for table, g in val_cols[['column', 'table']].groupby('table')}
    des_tables = set(used_des_cols.keys()).intersection(used_val_cols.keys()) # 出現在df_feats中的tables

    print('set(used_table)-set(車險欄位中的table):', des_tables.difference(used_tables))
    for table in used_tables:
        des_cols_t, val_cols_t = used_des_cols[table], used_val_cols[table]
        
        val_cols_t

    a = {'a':[1,2,3], 'aa':[1,2,3]}
    b = {'a':[4,5,6], 'bb':[4,5,6]}

    {**a, **b}

    usecols = {}

    df_feats

    featsCols =
    {k: g['column'].to_list() for k, g in feats.groupby('table')}
    valsCols = {k: g['column'].to_list() for k, g in vals.groupby('table')}

    usecols = {'feats':featsCols, 'vals':valsCols}

    table_label = pd.read_excel('./data/code/車險欄位.xlsx', sheet_name=None)
    for name, sh in table_label.items():
        sh.columns = range(len(sh.columns))
        labels = {k: g[1].to_list() for k, g in sh.groupby(0)}
