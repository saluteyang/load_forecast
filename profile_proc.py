import glob
import pandas as pd
import datetime as dt
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# processing weather data  ####################################################################################

# below will fail initially as there is 24 in the hour string in the 2017 file (see commented out code)
# there is also a thousands separator inside numbers in that file that causes parsing problems

# aggregate_load['Date'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[0]
# aggregate_load['Hour'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[1]
# aggregate_load['Hour'] = aggregate_load['Hour'].str.split(':', expand=True)[0]
# set(aggregate_load['Hour'])

aggregate_load = pd.DataFrame()
for f in glob.glob('ERCOT_load_profiles/*native*.csv'):
    print(f)
    df = pd.read_csv(f)
    # print(df.columns)
    aggregate_load = aggregate_load.append(df, ignore_index=True)

aggregate_load['Hour_End'] = pd.to_datetime(aggregate_load['Hour_End'])
aggregate_load.iloc[:, 1:10].astype('float')
aggregate_load.to_csv('test.csv', index=False)


# since Q4 2012, a new column was added
# for f in glob.glob('ERCOT_load_profiles/*Profiles*.csv'):
#     print(f)
#     df = pd.read_csv(f)
#     print(df.columns)

# for i in range(len(lst)):
#     print({i: x.shape for x in lst})


# processing load data  ######################################################################################
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

# run the following seciton if using profiles  ##################################################################
with open('test_profiles.csv', 'r') as f:
    profile_load = pd.read_csv(f)

# section to run for busmed profile
# profile_busmed = profile_load.loc[profile_load['Type'] == 'BUSMEDLF_COAST'].drop(columns=['Type'])
# profile_busmed = pd.melt(profile_busmed, id_vars=['Date'], var_name='Segment', value_name='COAST')
# profile_busmed['Segment'] = [int(x.split('_')[1])//4 - 1 for x in profile_busmed['Segment']]
# profile_busmed.index = pd.to_datetime(profile_busmed['Date'].map(str) + ' ' + profile_busmed['Segment'].map(str), format='%m/%d/%Y %H')
# profile_busmed = profile_busmed.groupby(profile_busmed.index).mean()

# section to run for reslo profile
profile_reslo = profile_load.loc[profile_load['Type'] == 'RESLOWR_COAST'].drop(columns=['Type'])
profile_reslo = pd.melt(profile_reslo, id_vars=['Date'], var_name='Segment', value_name='COAST')
profile_reslo['Segment'] = [int(x.split('_')[1])//4 - 1 for x in profile_reslo['Segment']]
profile_reslo.index = pd.to_datetime(profile_reslo['Date'].map(str) + ' ' + profile_reslo['Segment'].map(str), format='%m/%d/%Y %H')
profile_reslo = profile_reslo.groupby(profile_reslo.index).mean()

# renaming to aggregate load to agree with input_clean.py
# switch to the desired load profile here
aggregate_load = profile_reslo['COAST'].dropna().to_frame()

# continue to execute sections 'joining weather and load data' and beyond in input_clean.py
