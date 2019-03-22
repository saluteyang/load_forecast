
import seaborn as sns
import matplotlib.animation as animation
import matplotlib.pylab as plt
import numpy as np
from numpy import fft
import pandas as pd
import holidays
import pickle
from fbprophet import Prophet
from helpers import *

sns.set()

with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000  # source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# if running without weather data
# duplicate index due to additional hour in Nov due to DST
joined = aggregate_load.groupby(aggregate_load.index).first()
joined = joined.dropna().copy()
joined = joined['COAST']

train_data = joined[joined.index.year != 2017]
test_data = joined[joined.index.year == 2017]

# fourier transform ###########################################
def fourierExtrapolation(x, n_predict, detrend=True, n_harm=3):
    n = x.size
    if detrend == True:
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        x_notrend = x - p[0] * t  # detrended x
    else:
        x_notrend = x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

x = test_data[:168*3,0]
n_predict = 72
extrapolation = fourierExtrapolation(x, n_predict, n_harm=5)
plt.plot(np.arange(0, x.size), x, 'b', label='x', linewidth=3)
plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation')
plt.legend()
plt.show()

# DTW clustering ####################################
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000  # source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# if running without weather data
# duplicate index due to additional hour in Nov due to DST
joined = aggregate_load.groupby(aggregate_load.index).first()
joined = joined.dropna().copy()

joined['Week_Year'] = joined.index.weekofyear
joined['Wknd_Flag'] = (joined.index.weekday > 4) * 1
joined['Date'] = joined.index.date
us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined['Date']]
joined['Off_Flag'] = joined[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
joined['Year'] = joined.index.year

joined = joined[['COAST', 'Year', 'Week_Year', 'Off_Flag']]
# filter out weeks where there are holidays in addition to weekends
filter = joined.groupby(['Year', 'Week_Year'])['Off_Flag'].sum().reset_index()
filter.rename(columns={'Off_Flag': 'Off_Hours'}, inplace=True)
joined = joined.merge(filter, on=['Year', 'Week_Year'], how='left')
cluster_data = joined[joined['Off_Hours']==48][['COAST', 'Year', 'Week_Year']]
# create a within group hour counter to use for column labels after pivot
cluster_data['idx'] = cluster_data.groupby(['Year', 'Week_Year']).cumcount()
cluster_data = cluster_data.pivot_table(index=['Year', 'Week_Year'], columns='idx', values='COAST')
cluster_data = cluster_data[~np.isnan(cluster_data).any(axis=1)]
week_counter = cluster_data.index  # save for later use
cluster_data = cluster_data.as_matrix()


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return np.sqrt(LB_sum)

import random


def k_means_clust(data, num_clust, num_iter, w=4):
    centroids = random.sample(list(data), num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1
        print(counter)
        assignments = {}
        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    return (centroids, assignments)

centroids = k_means_clust(cluster_data, 4, 100, 4)

# with open(f'centroids.pickle', 'wb') as pfile:
#     pickle.dump(centroids, pfile)

with open(f"centroids.pickle", "rb") as pfile:
    exec(f"centroids = pickle.load(pfile)")

# plotting centroids
for i in centroids[0]:
    plt.plot(i)
plt.savefig('centroids.png', dpi=600, bbox_inches="tight")
plt.show()

# find assignment to centroids for the weeks
# import operator
# sorted_centroids = sorted(centroids[1].items(), key=operator.itemgetter(1))
assignment = []
week_num = []
for cluster in centroids[1]:
    print(cluster)
    for i in centroids[1][cluster]:
        print(i)
        week_num.append(i)
        assignment.append(cluster)

pd_centroids = pd.DataFrame({'weeknum': week_num,
                             'cluster': assignment})
# pd_centroids.sort_values(by='weeknum', inplace=True)
pd_week_counter = pd.DataFrame({'weeknum': range(len(week_counter)),
                                'year': week_counter.to_frame().iloc[:, 0],
                                'week_year': week_counter.to_frame().iloc[:, 1]})

pd_centroids = pd_centroids.merge(pd_week_counter, on='weeknum')
pd_centroids.groupby(['week_year', 'cluster'])['weeknum'].count()

# why are the following week_num missing from the final assignments
# [i for i in range(cluster_data.shape[0]) if i not in week_num]  # [0, 1, 16, 19]

# animating plot ####################################

# series to animate
series_ani = joined

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(10, 6))
plt.ylim(np.min(series_ani), np.max(series_ani))
plt.xlabel('Hour', fontsize=15)
plt.ylabel('GW', fontsize=15)
plt.title('Energy Consumption of the Texas Coast\n over 7 week period', fontsize=20)

def animate(i):
    # fig.clear()
    data = series_ani[24*(int(i)-1):96+24*(int(i)-1)]  # select data range
    p = sns.lineplot(x=data.index, y=data, color='b')
    p.tick_params(labelsize=10, rotation=45)
    plt.setp(p.lines, linewidth=2)

ani = animation.FuncAnimation(fig, animate, frames=45, repeat=True)
ani.save('load.mp4', writer=writer)

# alternative animation
series_ani = joined
series_ani_onemon = series_ani['2010-01']
series_ani_onemon = pd.concat([pd.Series(series_ani_onemon.values),
                               pd.Series(range(series_ani_onemon.shape[0]))],
                              axis=1)

fig = plt.figure(figsize=(16, 10))
ax = plt.axes(xlim=(0, 47), ylim=(6.5, 14.5))
ax.set_xlabel('Hour', fontsize=20)
ax.set_ylabel('GW', fontsize=20)
ax.tick_params(labelsize=15)
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

# x_data, y_data = [], []

def animate(i):  # called sequentially, staring with 0
    idx_start = int(i) + 48
    x = list(range(0, 48))
    y = list(series_ani_onemon.iloc[:, 0][(idx_start-48):idx_start].values)
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=20, blit=True)
anim.save('load2.mp4', fps=30)

# times series decomposition #############################

with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000  # source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

us_holidays = holidays.UnitedStates(years=range(2010, 2017))
us_holidays = pd.DataFrame.from_dict(us_holidays, orient='index').reset_index()
us_holidays.columns = ['ds', 'holiday']

m = Prophet(holidays=us_holidays, changepoint_prior_scale=0.001)
# m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
# m.add_country_holidays(country_name='US')

series = aggregate_load['2010':'2016']['COAST']
series = series.reset_index()
series.columns = ['ds', 'y']
m.fit(series)

# do not include forecast for decomp plots
future = m.make_future_dataframe(periods=0)  # periods by default means days
forecast = m.predict(future)
fig_decomp = m.plot_components(forecast)

plt.savefig('ts_decomp.png', dpi=1000, bbox_inches="tight")
plt.show()

# forecast next 168 hours
future = m.make_future_dataframe(periods=1680, freq='H')
forecast = m.predict(future)
fig_forecast = m.plot(forecast)

plt.show()

# forecast_save = forecast[forecast['ds'].dt.year==2017]
# with open(f'prophet_forecast.pickle', 'wb') as pfile:
#     pickle.dump(forecast_save, pfile)

forecast_comp = forecast_save[['ds', 'yhat']]
mape(aggregate_load['2017']['COAST'][:168], forecast_comp['yhat'][:168])

# with open(f'prophet_model.pickle', 'wb') as pfile:
#     pickle.dump(m, pfile)
# with open(f'prophet_forecast.pickle', 'wb') as pfile:
#     pickle.dump(forecast, pfile)

with open(f"prophet_model.pickle", "rb") as pfile:
    exec(f"m = pickle.load(pfile)")
with open(f"prophet_forecast.pickle", "rb") as pfile:
    exec(f"forecast = pickle.load(pfile)")

forecast_wk = forecast[pd.to_datetime(forecast['ds']).isin \
    (pd.date_range(start='2017-01-01', end='2017-01-08', freq='H'))][['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
test = aggregate_load['2017']['COAST'][:168]

plt.plot(forecast_wk['yhat'].values)
plt.plot(test.reset_index()['COAST'].values)
plt.show()