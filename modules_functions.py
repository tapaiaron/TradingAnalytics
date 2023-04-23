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

# Functions to load data and calculate basic metrics
def load_data_yahoo(tickerSymbol='META', start_date= (modules.datetime.date.today()-modules.relativedelta(years=20)).strftime('%Y-%m-%d'), end_date=(modules.datetime.date.today()-modules.datetime.timedelta(days=1)).strftime('%Y-%m-%d')):
    tickerData=modules.yf.Ticker(tickerSymbol)
    data=tickerData.history(period='1d', start=start_date, end=end_date)
    return data

def calc_log_return(data, mark_type='Close'):
    data['log_return']=modules.np.log(data[mark_type]/data[mark_type].shift(1))
    return data.dropna()

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

# Functions to transform data and backtes results
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

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a= dataset[i:(i+time_step),0]
		dataX.append(a)
		dataY.append(dataset[i+time_step,0])
	return modules.np.array(dataX), modules.np.array(dataY)

def descriptive_stats(data):
    return f"Annual Average: {data['log_return'].mean()*252}, Annual Stdev: {data['log_return'].std()*modules.sqrt(252) } \n " \
           f"{data['log_return'].describe()} \n" \

# Functions for plotting
def plot_logr_price(data, tickerSymbol='META'):
    fig = modules.plt.figure(figsize=(7,4))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(f'{tickerSymbol}', size=25, fontweight='bold')
    axs[0].plot(data.index, data['Close'], color='black', linewidth=2)
    axs[1].plot(data.index, data['log_return'], color='darkblue', linewidth=2)
    axs[1].yaxis.tick_right()
    axs[0].set_ylabel('Price', fontsize=12, labelpad=5)
    axs[1].set_xlabel('Date', fontsize=12)
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel('Log-return', fontsize=12, rotation=270, labelpad=15)
    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_yticks(modules.np.arange(min(data['Close']), max(data['Close']), (max(data['Close'])-min(data['Close']))*0.17))
    axs[1].set_yticks(modules.np.arange(min(data['log_return']), max(data['log_return']), (max(data['log_return'])-min(data['log_return']))*0.17))
    axs[1].tick_params(axis='y', labelsize=8.5)
    axs[0].tick_params(axis='y', labelsize=8.5)
    axs[0].yaxis.grid(True, linestyle='--')
    axs[0].xaxis.grid(True, linestyle='--')
    axs[1].yaxis.grid(True, linestyle='--')
    axs[1].xaxis.grid(True, linestyle='--')
    modules.plt.xticks(fontsize=12)
    modules.plt.show()

# Functions for modelling
def pred_interval(data,n=10):
    while (True):
        try:
            n=input("For how many days should the model predict for? (default=10)")
            if n == "":
                n=10
            n=int(n)
            break
        except ValueError:
            "The given value is not a number!"

    last_date_idx=data.index[-1]
    day_after_last_date_idx=last_date_idx+modules.datetime.timedelta(days=1)
    fcast_dates=[]
    for i in range(n):
        fcast_dates.append(day_after_last_date_idx+modules.datetime.timedelta(days=i))

    index_col=pd.DataFrame(index_col, columns=['Date'])
    index_col['date']=index_col['Date']
    index_col.set_index('Date', inplace=True)
    ### Need to continue from here
