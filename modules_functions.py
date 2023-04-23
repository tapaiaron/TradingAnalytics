# Class for importing all of the modules
class ImportModules:
    # Import subprocess for the installation of missing modules.
    # Install function
    def install(package):
        import subprocess
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call(["pip", "install", package])
            
    # Installing the missing modules
    install("matplotlib")
    install("numpy")
    install("pandas")
    install("seaborn")
    install("statsmodels")
    install("yfinance")
    install("arch")
    install("scipy")
    install("tensorflow")
    install("keras")
    install("keras_tuner")
    
    # Importing the modules and methods.
    import datetime
    from dateutil.relativedelta import relativedelta
    import math
    from ntpath import join
    import numbers
    import os
    import random
    import sys
    from cgitb import scanvars
    from re import A
    from tkinter import font
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import cumsum, log, polyfit, sqrt, std, subtract
    import pandas as pd
    from pandas_datareader import test
    import seaborn as sns
    import statsmodels.api as sm
    import yfinance as yf
    from arch import arch_model
    from scipy import stats
    import statsmodels.tsa.stattools as tsa
    import statsmodels.stats as sm_stat
    from statsmodels.tsa.stattools import zivot_andrews
    from scipy.stats import kurtosis, skew, moment, norm, probplot, t
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf, adfuller, q_stat
    import tensorflow as tf
    import keras_tuner as kt
    from keras_tuner import HyperParameter as hp
    from tensorflow.keras.callbacks import History
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import SGD, Adam
    from sklearn.preprocessing import  StandardScaler

    def __init__(self):
        import warnings
        import os
        # Ignore errors
        warnings.simplefilter(action='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.system("cls")
# Importing all of the modules and calling this one variable.
modules = ImportModules()

def load_data_yahoo(tickerSymbol='META', start_date= (modules.datetime.date.today()-modules.relativedelta(years=20)).strftime('%Y-%m-%d'), end_date=(modules.datetime.date.today()-modules.datetime.timedelta(days=1)).strftime('%Y-%m-%d')):
    tickerData=modules.yf.Ticker(tickerSymbol)
    df=tickerData.history(period='1d', start=start_date, end=end_date)
    return df

def calc_log_return(data):
    data['log_return']=modules.np.log(data/data.shift(1))
    return data

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [modules.sqrt(modules.std(modules.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = modules.polyfit(modules.log(lags), modules.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def evaluate(observation, forecast):
    # Call sklearn function to calculate MAE
    mae = modules.mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MSE
    mse = modules.mean_squared_error(observation, forecast)
    print(f'Mean Squared Error (MSE): {round(mse,3)}')
    # Call sklearn function to calculate RMSE
    rmse= modules.mean_squared_error(observation, forecast, squared=False)
    print(f'Root Mean Squared Error (RMSE): {round(rmse,3)}')
    return mae, mse, rmse


