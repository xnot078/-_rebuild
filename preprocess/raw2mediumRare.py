import pandas as pd
from preprocess.load_tables import from_google_sheet
import re
#
# import os
# os.path.splitext('ca_ply_info.txt')[0]
# os.listdir(r'./data/raw')

print('start loading data.')
print()
t = from_google_sheet()
car = t.load_tables(r'./data/raw')
print()
print('convert to .parq and save.')
print()
for k, data in car.items():
    data.to_parquet(f'./data/medium_rare/{k}.parq', compression='brotli')
print()
print('completed.')
# ['ply_relation', 'policy', 'ply_ins_type']
#
#
# data.keys()
# car['policy'].head(2)['iroute']
# car['policy']['ipolicy'].value_counts()
