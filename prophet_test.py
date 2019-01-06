from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import pandas as pd

aggregate_load = pd.DataFrame()
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000
aggregate_load['Hour_End'] = pd.to_datetime(aggregate_load['Hour_End'])

df = aggregate_load[['Hour_End', 'COAST']]
df = df.rename(columns={'Hour_End': 'ds', 'COAST': 'y'})

# prophet has a built-in cross validation function
# initial is the training data length
# horizon is the forecast horizon (from the cutoff date)
# period is the spacing between cutoff dates, a forecast is made for points between every cutoff and cutoff + horizon

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=24, freq='H')
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

fig1.show()
fig2.show()

df_cv = cross_validation(m, initial='365 days', period='100 days', horizon='24 hours')  # froze computer
df_cv.head()
df_cv.to_csv('prophet_crossval.csv')

df_p = performance_metrics(df_cv)
df_p.head()
df_p.to_csv('prophet_performance.csv')