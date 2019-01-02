import glob
import pandas as pd
import datetime as dt
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# since Q4 2012, a new column was added
# for f in glob.glob('ERCOT_load_profiles/*Profiles*.csv'):
#     print(f)
#     df = pd.read_csv(f)
#     print(df.columns)

# for i in range(len(lst)):
#     print({i: x.shape for x in lst})

cols = ['Type', 'Date'] + ['seg_' + n for n in list(map(str, range(4, 96+4)))]  # start with 4 so that integer div returns 1
lst = []
for f in glob.glob('ERCOT_load_profiles/*Profiles*.csv'):
    print(f)
    df = pd.read_csv(f)
    df = df.loc[df.iloc[:, 0].isin(['BUSMEDLF_COAST', 'RESLOWR_COAST'])]
    if df.shape[1] > 102:  # since Q4 2012, a new column was added
        n = 5  # remove last 5 columns: 1 record time (new), 4 DST columns
    else:
        n = 4
    df = df.iloc[:, :-n]
    df.columns = cols
    lst.append(df)
profile_load = pd.concat(lst)
profile_load.to_csv('test_profiles.csv', index=False)

profile_busmed = profile_load.loc[profile_load['Type'] == 'BUSMEDLF_COAST'].drop(columns=['Type'])
profile_busmed = pd.melt(profile_busmed, id_vars=['Date'], var_name='Segment', value_name='COAST')
profile_busmed['Segment'] = [int(x.split('_')[1])//4 - 1 for x in profile_busmed['Segment']]
profile_busmed.index = pd.to_datetime(profile_busmed['Date'].map(str) + ' ' + profile_busmed['Segment'].map(str), format='%m/%d/%Y %H')
profile_busmed = profile_busmed.groupby(profile_busmed.index).mean()

profile_reslo = profile_load.loc[profile_load['Type'] == 'RESLOWR_COAST'].drop(columns=['Type'])
profile_reslo = pd.melt(profile_reslo, id_vars=['Date'], var_name='Segment', value_name='COAST')
profile_reslo['Segment'] = [int(x.split('_')[1])//4 - 1 for x in profile_reslo['Segment']]
profile_reslo.index = pd.to_datetime(profile_reslo['Date'].map(str) + ' ' + profile_reslo['Segment'].map(str), format='%m/%d/%Y %H')
profile_reslo = profile_reslo.groupby(profile_reslo.index).mean()

# converting string to datetime needs hour string to start with zero!
# temp = profile_busmed['Date'].map(str) + ' ' + profile_busmed['Segment'].map(str)
# temp2 = []
# for x in temp:
#     try:
#         converted = dt.datetime.strptime(x, '%m/%d/%Y %H')
#     except ValueError:
#         print(x, 'conversion error')
#     else:
#         temp2.append(converted)

# profile_busmed[profile_busmed.isnull().any(axis=1)]

# renaming to aggregate load to agree with input_clean.py
# switch to the desired load profile here
aggregate_load = profile_busmed['COAST'].dropna().to_frame()

# continue to execute sections 'joining weather and load data' and beyond in input_clean.py
