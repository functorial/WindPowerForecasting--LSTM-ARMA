from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
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


def eval_ARMA(AR_lags, MA_lags, train, val):
    '''Creates and trains an ARMA model with the 
    supplied significant lags. Returns the root mean 
    squared error of the model predictions against the 
    supplied validation set.''' 

    val_hat = val.copy()
    train_updated = train.copy()
    for i in range(len(val_hat)):
        # Train on window
        model = ARIMA(train_updated, order=(AR_lags, 0, MA_lags), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        # Make prediction
        pred = model_fit.forecast(1)
        val_hat.iloc[i] = pred
        # Move window
        train_updated.append(val_hat.iloc[i:i+1])
        train_updated = train_updated.iloc[1:]
    rmse = sqrt(mean_squared_error(val, val_hat))
    return rmse 

def rolling_eval_ARMA(AR_lags, MA_lags, X, h:int, window_h_ratio:int=5):
    '''Performs a rolling window analysis on an 
    ARMA model with the suppllied significant lags. 
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
        rmse = eval_ARMA(AR_lags, MA_lags, train, val)
        rmse_vals.append(rmse)
        rmse_avg = sum(rmse_vals) / N
        rmse_var = sum([(e - rmse_avg)**2 for e in rmse_vals]) / (N-1)

    # Pre-Live Optimization
    train = X[-m:]
    model = ARIMA(train, order=(AR_lags, 0, MA_lags), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    return model_fit, rmse_avg, rmse_var

def compare_ARMA_models(AR_lag_sets:list, MA_lag_sets:list, X, h:int, window_h_ratio:int=5):
    '''Performs a Walk Forward Analysis for predictive performance
    for each set of significant lags given. 
    `AR_lag_sets` is a list of ints of significant lags for the AR part.
    `MA_lag_sets` is a list of ints of significant lags for the MA part.
    `X` is the time series.
    `h` is the forecast horizon size.'''
    m = h*window_h_ratio        # The size of the rolling window.
    idx = (len(X) - m) % h      # The index used to slice X.        
    T = len(X) - idx            # Sample size after slice.
    N = ((T-m)//h)+1            # Number of windows.
    assert m <= len(X), 'Window size must be smaller than time series.'
    num_tests = len(AR_lag_sets) * len(MA_lag_sets)
    rmse_avgs = list()
    print(f"Window Size: {m}+{h}\tNum Windows: {N}\n")
    i = 0
    for p in AR_lag_sets:
        for q in MA_lag_sets:
            i += 1
            print(f"Test {i}: ARMA(p={p}, q={q})")
            print("========================================================")
            try:
                _, rmse_avg, rmse_var = rolling_eval_ARMA(AR_lags=p, MA_lags=q, X=X, h=h, window_h_ratio=window_h_ratio)
                rmse_avgs.append(rmse_avg)
                print(f"RMSE Mean: {rmse_avg:.2f}\tRMSE Variance: {rmse_var:.2f}\n")
            except IndexError:
                print("ERR: CHECK LAG BOUNDS")
                num_tests -= 1
    try:
        g_mean = sum(rmse_avgs)/num_tests
        print(f"\nGrand Mean RMSE: {g_mean:.3f}")
    except ZeroDivisionError:
        print(f"\nGrand Mean RMSE: ERROR")

def eval_AR(lags:list, train, val):
    '''Creates and trains an AutoRegressive model with the 
    supplied significant lags. Returns the root mean 
    squared error of the model predictions against the 
    supplied validation set.''' 
    
    val_hat = val.copy()
    train_updated = train.copy()
    for i in range(len(val_hat)):
        # Train on window
        model = AutoReg(train_updated, lags, old_names=False)
        model_fit = model.fit()
        # Make prediction
        pred = model_fit.forecast(1)
        val_hat.iloc[i] = pred
        # Move window
        train_updated.append(val_hat.iloc[i:i+1])
        train_updated = train_updated.iloc[1:]
    rmse = sqrt(mean_squared_error(val, val_hat))
    return rmse 

def rolling_eval_AR(lags:list, X, h:int, window_h_ratio:int=5):
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

def compare_AR_models(lag_sets:list, X, h:int, window_h_ratio:int=5):
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
    num_tests = len(lag_sets)
    rmse_avgs = list()
    print(f"Window Size: {m}+{h}\tNum Windows: {N}\n")
    for i in range(len(lag_sets)):
        lags = lag_sets[i]
        print(f"Test {i}: AR(lags={lag_sets[i]})")
        print("========================================================")
        try:
            _, rmse_avg, rmse_var = rolling_eval_AR(lags=lags, X=X, h=h, window_h_ratio=window_h_ratio)
            print(f"RMSE Mean: {rmse_avg:.2f}\tRMSE Variance: {rmse_var:.2f}\n")
            rmse_avgs.append(rmse_avg)
        except IndexError:
            print("ERR: CHECK LAG BOUNDS")
            num_tests -= 1
    try:
        g_mean = sum(rmse_avgs)/num_tests
        print(f"\nGrand Mean RMSE: {g_mean:.3f}")
    except ZeroDivisionError:
        print(f"\nGrand Mean RMSE: ERROR")


class LastBaseline():

    def fit(self, X):
        self.last = X[-1:]
        return self
    
    def forecast(self, y:pd.Series) -> pd.Series:
        # Input y to easily grab datetime index
        y_hat = y.copy()
        h = len(y_hat)
        for i in range(h):
            y_hat.iloc[i] = self.last
        return y_hat

class RepeatBaseline():

    def fit(self, X:pd.Series):
        self.X = X
        return self

    def forecast(self, y:pd.Series) -> pd.Series:
        y_hat = y.copy()
        h = len(y_hat)
        for i in range(h):
            y_hat.iloc[i] = self.X.iloc[-h+i]
        return y_hat
