from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
from sklearn.metrics import mean_squared_error

def adf_summary(X):
    '''Applies the Automated Dickey-Fuller test.
    Null-hypothesis: The time series is not stationary.'''
    adf = adfuller(X)
    print(f'ADF Statistic: {adf[0]}')
    print(f'p-value: {adf[1]}')
    print('Critical Values:')
    for key, value in adf[4].items():
        print(f'\t{key}: {value}')

    if adf[0] < adf[4]['5%']:
        print("Reject Null: Assume time series is stationary.")
    else:
        print("Fail to reject null: Time series may not be stationary.")

def eval_AR(lags, train, val):
    '''Creates and trains an AutoRegressive model with the 
    supplied significant lags. Returns the root mean 
    squared error of the model predictions against the 
    supplied validation set.''' 

    model = AutoReg(train, lags, old_names=False)
    model_fit = model.fit()
    val_hat = model_fit.forecast(len(val))
    rmse = sqrt(mean_squared_error(val, val_hat))
    return rmse 

def rolling_eval_AR(lags, X, h, window_h_ratio=5):
    '''Performs a rolling window analysis on an 
    AutoRegressive model with the suppllied significant lags. 
    X is the full time series, h is the size of the forecast horizon. 

    Returns the pre-live fitted model, the RMSE mean, and the
    RMSE variance over the windows.

    Note: If len(X) is not divisible by h, then the head of X
    is sliced off in the smallest way possible so that the
    result is divisible by h.'''

    m = h*window_h_ratio        # The size of the rolling window.
    idx = (len(X) - m) % h      # The index used to slice X.        
    T = len(X) - idx            # Sample size after slice.
    N = ((T-m)//h)+1            # Number of windows.
    assert m <= len(X), 'Window size must be smaller than time series.'

    # Walk forward validation
    rmse_vals = []
    for i in range(N):
        a = h*i     # Start index for the window
        train = X[idx+a:idx+a+m-h]
        val = X[idx+a+m-h:idx+a+m]
        rmse = eval_AR(lags, train, val)
        rmse_vals.append(rmse)
        rmse_avg = sum(rmse_vals) / N
        rmse_var = sum([(e - rmse_avg)**2 for e in rmse_vals]) / (N-1)

    # Pre-Live Optimization
    train = X[-m:]
    model = AutoReg(train, lags, old_names=False)
    model_fit = model.fit()

    return model_fit, rmse_avg, rmse_var

def compare_AR_models(lag_sets, X, h, window_h_ratio=5):
    '''Performs a Walk Forward Analysis for predictive performance
    for each set of significant lags given. 
    `lag_sets` is a list of lists of significant lags.
    `X` is the time series.
    `h` is the forecast horizon size.'''
    m = h*window_h_ratio        # The size of the rolling window.
    idx = (len(X) - m) % h      # The index used to slice X.        
    T = len(X) - idx            # Sample size after slice.
    N = ((T-m)//h)+1            # Number of windows.
    assert m <= len(X), 'Window size must be smaller than time series.'
    print(f"Window Size: {m}\tNum Windows: {N}\n")

    for i in range(len(lag_sets)):
        print(f"Test {i}: AR(lags={lag_sets[i]})\n========================================================")
        lags = lag_sets[i]
        _, rmse_avg, rmse_var = rolling_eval_AR(lags=lags, X=X, h=h, window_h_ratio=window_h_ratio)
        print(f"RMSE Mean: {rmse_avg:.2f}\tRMSE Variance: {rmse_var:.2f}\n")