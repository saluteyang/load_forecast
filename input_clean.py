import csv
# from datetime import datetime, timedelta
import datetime as dt
from matplotlib import pyplot as plt
import glob
import pandas as pd
import numpy as np

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

# groupby keys have to be hashable
weather['Date'] = weather['Hour_End'].dt.date
weather['Hour'] = weather['Hour_End'].dt.hour
weather_grouped = weather.groupby(['Date', 'Hour'])
weather_hr = weather_grouped.agg({'Drybulb': np.average, 'Humidity': np.average})
weather_hr = weather_hr.reset_index()
weather_hr.plot(x='Date', y='Drybulb')
plt.show()

# import actual load including profiles from two business and residential profiles

aggregate_load = pd.DataFrame()
for f in glob.glob('ERCOT_load_profiles/*native*.csv'):
    print(f)
    df = pd.read_csv(f)
    # print(df.columns)
    aggregate_load = aggregate_load.append(df, ignore_index=True)

# aggregate_load.shape
# aggregate_load.dtypes

aggregate_load['Hour_End'] = pd.to_datetime(aggregate_load['Hour_End'])
# aggregate_load.iloc[:, 1:10].astype('float')

# above will fail initially as there is 24 in the hour string in the 2017 file
# there is also a thousands separator inside numbers in that file that causes parsing problems

# aggregate_load['Date'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[0]
# aggregate_load['Hour'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[1]
# aggregate_load['Hour'] = aggregate_load['Hour'].str.split(':', expand=True)[0]
# set(aggregate_load['Hour'])
# aggregate_load[aggregate_load['Hour'] == '24']


# plot time series with matplotlib

aggregate_load.plot(x='Hour_End', y='ERCOT')
plt.show()



