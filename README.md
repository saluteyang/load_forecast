# load_forecast
load forecast using ML techniques

I used a number of different machine learning algorithms to forecast ERCOT (Electricity Reliability Council of Texas) electricity load.
The simpler methods that work well include:
* Decision tree regression (DT) 

Doing such analysis on the mean as well as certain interested quantiles can provide error bands that can potentially account for outlier outcomes.

For DT, I created indicator varaibles such as
1. time of day
2. day of week
3. day of year, etc.

More advanced methods I tested in various configurations (different architectures that finally included time saving convolutional layers) include Recurrent Neural Network (RNN), which can discover cyclicality on its own without specifying in the model setup stage. For decision tree models, such specification is needed in the form of explicit lagging and time dummies.

In the test project, we only care about very near-term load forecast (on the scale of 24 hours to a week). Even then, the availability of frequently updated data played a big role in the model I chose and the performance of the models.

**RNN small-setup model assuming weekly updated inputs (including 95% interval bounds)**
![Alt text](/quantile_reg_168.png?raw=true)

**DT model assuming daily updated inputs (including 95% interval bounds)**
![Alt text](/quantile_rnn_168.png?raw=true)
Future improvements and modelling notes:
1. While an alternative approach of forecasting many different load profiles at more disaggregated level (by customer and usage types) was explored using dilated causal convolutional layers. The performance on test data leaves more to be desired. The "irregular" patterns in smaller-scale usage remain challenging without introducing hard-to-obtain exogenous variables such as production time schedules of commercial clients.

2. It's well-known that weather and load are highly correlated. We can also obtain usable location-specific weather forecast for a few days ahead. However, since my sample dataset include aggregated load from various locations, the weather varialbes will need to roll up as well.
