# WindPowerForecasting

We are given 2 years worth of observations of active power being generated by a windmill. The observations are recorded ever 10 minutes. We would like to give a forecast for the next 15 days of active power.

The time series given is tricky to handle because it is very noisy and 20% of the observations are null. The null observations come in large chunks, where there are often days worth of data missing.

The distribution appears to be stationary, although it does enjoy daily and yearly seasonality.

We choose to downsample the observations by taking the mean active power for each day. The decision to downsample is for a few reasons. Firstly, it makes the time series easier to interpolate. For example, if 3 days worth of data is missing in a row, then we only need to interpolate 3 observations instead of 3*24*6 = 432. Secondly, this is more aligned with the business question, as it asks for a forecast in units of days. In addition, a daily aggregation is natural way to abstract away the daily seasonality. We choose to downsample via the method of taking the daily mean since it will preserve our ability to estimate the amount of energy generated per day, if we choose to do so. 

Various autoregressive models are tested and compared using rolling-window validations. The performance of these models is underwhelming as they seem to be underfit, which is surprising. Further experimentation is needed to obtain a satisfactory model.
