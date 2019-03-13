import matplotlib.pylab as plt
import numpy as np
from numpy import fft
import pandas as pd
import holidays

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

# fourier harmonics ???
# hourly harmonic (pattern in a day)
# daily harmonic (pattern in a week)
# monthly harmonic (pattern in a year)


import plotly.plotly as py

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq, abs(Y), 'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
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
# create a within group counter to use for column labels after pivot
cluster_data['idx'] = cluster_data.groupby(['Year', 'Week_Year']).cumcount()
cluster_data = cluster_data.pivot_table(index=['Year', 'Week_Year'], columns='idx', values='COAST')
cluster_data = cluster_data.as_matrix()
cluster_data = cluster_data[~np.isnan(cluster_data).any(axis=1)]


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

    return centroids

centroids = k_means_clust(cluster_data, 4, 100, 4)

for i in centroids:
    plt.plot(i)

plt.show()

# animating plot ####################################
import seaborn as sns
import matplotlib.animation as animation
sns.set()
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