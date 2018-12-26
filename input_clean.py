import csv
import datetime as dt
from matplotlib import pyplot as plt
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

filename = 'Houston_tx_hobby_2010-2017.csv'

# choose only REPORTTPYE (misspelt) FM-15
# exclude time periods when hourly drybulb or humidity data are missing
# missing data can be blanks or wildcard character *

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    # print header with position
    for index, column_header in enumerate(header_row):
        print(index, column_header)

    dates, drybulb, humidity = [], [], []
    for row in reader:
        if row[6] == 'FM-15':
            try:
                date_obs = dt.datetime.strptime(row[5], "%Y-%m-%d %H:%M")
                drybulb_temp = int(row[10])
                relative_humidity = int(row[16])

            except ValueError:
                print(date_obs, 'missing data')

            else:
                dates.append(date_obs)
                drybulb.append(drybulb_temp)
                humidity.append(relative_humidity)

weather = pd.DataFrame({'Hour_End': dates,
                        'Drybulb': drybulb,
                        'Humidity': humidity})

weather = weather.set_index('Hour_End')
weather_hr = weather.groupby([lambda x: x.date, lambda x: x.hour])['Drybulb', 'Humidity'].mean()
# plot using index needs to turn index into a column first
# note that multi-index will be turned into columns by level
weather_hr['Hour_End'] = pd.to_datetime(weather_hr.index.map(lambda x: '-'.join((str(x[0]), str(x[1])))), format='%Y-%m-%d-%H')
weather_hr.plot(x='Hour_End', y='Drybulb')
plt.show()

# weather['Date'] = weather['Hour_End'].dt.date
# weather['Hour'] = weather['Hour_End'].dt.hour
# weather_grouped = weather.groupby(['Date', 'Hour'])
# weather_hr = weather_grouped.agg({'Drybulb': np.average, 'Humidity': np.average})
# weather_hr = weather_hr.reset_index()
# weather_hr.plot(x='Date', y='Drybulb')
# plt.show()

# import actual load including profiles from two business and residential profiles

aggregate_load = pd.DataFrame()

# run the below code if need to reprocess from source spreadsheets
# or skip and import processed data
for f in glob.glob('ERCOT_load_profiles/*native*.csv'):
    print(f)
    df = pd.read_csv(f)
    # print(df.columns)
    aggregate_load = aggregate_load.append(df, ignore_index=True)

aggregate_load['Hour_End'] = pd.to_datetime(aggregate_load['Hour_End'])
# aggregate_load.iloc[:, 1:10].astype('float')

# above will fail initially as there is 24 in the hour string in the 2017 file
# there is also a thousands separator inside numbers in that file that causes parsing problems

# aggregate_load['Date'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[0]
# aggregate_load['Hour'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[1]
# aggregate_load['Hour'] = aggregate_load['Hour'].str.split(':', expand=True)[0]
# set(aggregate_load['Hour'])
# aggregate_load[aggregate_load['Hour'] == '24']

# start here to import processed data
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)


# plot time series with matplotlib

aggregate_load.plot(x='Hour_End', y='ERCOT')
plt.show()

# join weather and load information

weather_hr = weather_hr.set_index('Hour_End')
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)
joined = aggregate_load.join(weather_hr, how='inner')

joined[['COAST', 'Drybulb', 'Humidity']].corr()

lm = sm.ols(formula='COAST ~ Drybulb + Humidity', data=joined).fit()
lm.params
lm.summary()

# the relationship is not linear
joined.plot(x='Drybulb', y='COAST', kind='scatter')
plt.show()