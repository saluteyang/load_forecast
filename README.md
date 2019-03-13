# load_forecast
load forecast using ML techniques

I use a number of different machine learning algorithms to forecast ERCOT (Electricity Reliability Council of Texas) load.
The simpler methods that work well include:
* Decision tree regressor (DT)
* Fully connected neural net (NN)

For DT, I create indicator varaibles that supply information as to
1. time of day
2. day of week
3. day of year, etc.

More advanced methods include Recurrent Neural Network (RNN). While RNN can discover cyclicality of short time scales, I find that it works better if I supply indicator variables, especially those on large time scales (ex. day of year).

Moreover, RNN improvement over feed-forward NN is small enough that I would recommend the latter.

Finally, the above assumed infrequent updating of historical data used for forecast (bi-weekly) and that we only care about very near-term load forecast (on the scale of 24 hours to a week). Obviously, if the update frequency were to increase, we can significantly improve the forecast accuracy (measured as the ratio of mean absolute error to average load) to upper 90's at hourly.

Future improvements and modelling notes:
1. While an alternative approach of forecasting many different load profiles at more disaggregated level (by customer and usage types) was explored using dilated causal convolutional layers. The performance on test data leaves more to be desired. The "irregular" patterns in smaller-scale usage remain challenging without introducing exogenous variables such as production time schedules or weather variables.
2. While RNNs would usually obviate the need to include explicit lagged features, I included them in the final version to help the model along. This also does not require much additional computation time but ensures domain knowledge of consumption pattern gets integrated.
3. It's well-known that weather and load are highly correlated. We can also obtain usable location-specific weather forecast for a few days ahead. However, since my sample dataset include aggregated load from various locations, the weather varialbes will need to roll up as well, requiring additional considerations.
