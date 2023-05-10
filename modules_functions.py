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
def load_data_yahoo(tickerSymbol='META',period='1d', start_date= (modules.datetime.date.today()-modules.relativedelta(years=20)).strftime('%Y-%m-%d'), end_date=(modules.datetime.date.today()-modules.datetime.timedelta(days=1)).strftime('%Y-%m-%d')):
    tickerData=modules.yf.Ticker(tickerSymbol)
    data=tickerData.history(period, start=start_date, end=end_date)
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

# Functions for modelling
def pred_interval(data,n=10, day_type="1d"):
    while (True):
        try:
            n=input("For how many months/weeks/days should the model predict for? (default=10 days)")
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
    return n

def day_conv(day_type="1d"):
    dt=0
    if day_type=="1d":
         dt=1/252
    elif day_type=="5d":
        dt=1/52
    elif day_type=="1m":
        dt=1/12
    else:
        print(f"Given day_type is not valid i.e.({day_type})")
    return dt

def mc_model(data,n,sim=10000,day_type="1d",mark_type='Close'):
    r_mean_ann=data['log_return'].mean()*252
    r_std_ann=data['log_return'].std()*modules.sqrt(252)
    simulation_dt=day_conv(day_type)
    last_value=data[mark_type][-1]
    sim_values=modules.np.zeros(shape=(n, sim))
    log_return_mc_0=(r_mean_ann-0.5*r_std_ann**2)*simulation_dt+r_std_ann*modules.math.sqrt(simulation_dt)*modules.norm.ppf(modules.random.uniform(0, 1))
    sim_values[0,:]=last_value*modules.math.exp(log_return_mc_0)
    for i in range(1,sim):
        for j in range(1,n):
            sim_values[j,i]=sim_values[j-1,i]*modules.math.exp(((r_mean_ann-0.5*r_std_ann**2)*simulation_dt+r_std_ann*modules.math.sqrt(simulation_dt)*modules.norm.ppf(modules.random.uniform(0, 1))))

    # Lower and upper bound for the predicted trajectory
    min_values_mc = []
    max_values_mc = []
    for i in range(1, 11):
        min_value_mc = min(sim_values[i-1,1:sim])
        min_values_mc.append(min_value_mc)
        max_value_mc = max(sim_values[i-1,1:sim])
        max_values_mc.append(max_value_mc)
    mc_mid_pred=[]
    pred_mc_temp = zip(max_values_mc,min_values_mc)
    for list1_mc_i, list2_mc_i in pred_mc_temp:
        mc_mid_pred.append(list1_mc_i-list2_mc_i)
    for i in range(1,11):
        mc_mid_pred[i-1]=(mc_mid_pred[i-1]/2)+min_values_mc[i-1]





    ### Need to continue from here
    #Plan: predict the vol, calculate back the potention option prices and then trade based off of that.

# Functions for plotting
def plot_logr_price(data, tickerSymbol='META', mark_type='Close'):
    fig = modules.plt.figure(figsize=(7,4))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(f'{tickerSymbol}', size=25, fontweight='bold')
    axs[0].plot(data.index, data[mark_type], color='black', linewidth=2)
    axs[1].plot(data.index, data['log_return'], color='darkblue', linewidth=2)
    axs[1].yaxis.tick_right()
    axs[0].set_ylabel('Price', fontsize=12, labelpad=5)
    axs[1].set_xlabel('Date', fontsize=12)
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel('Log-return', fontsize=12, rotation=270, labelpad=15)
    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_yticks(modules.np.arange(min(data[mark_type]), max(data[mark_type]), (max(data[mark_type])-min(data[mark_type]))*0.17))
    axs[1].set_yticks(modules.np.arange(min(data['log_return']), max(data['log_return']), (max(data['log_return'])-min(data['log_return']))*0.17))
    axs[1].tick_params(axis='y', labelsize=8.5)
    axs[0].tick_params(axis='y', labelsize=8.5)
    axs[0].yaxis.grid(True, linestyle='--')
    axs[0].xaxis.grid(True, linestyle='--')
    axs[1].yaxis.grid(True, linestyle='--')
    axs[1].xaxis.grid(True, linestyle='--')
    modules.plt.xticks(fontsize=12)
    modules.plt.show()

def plot_predicted(model,interval, sim=10000):
    fig= modules.plt.figure(figsize=(6,4))
    for i in range(1,sim):
        modules.plt.plot(fcast_dates,sim_values[:,i], color="darkblue", alpha=0.2)
    modules.plt.plot(fcast_dates, mc_mid_pred, linewidth=3, color="blue")
    modules.plt.xlabel('Date', fontsize=12, labelpad=5)
    modules.plt.ylabel('Price', fontsize=12, labelpad=5)
    modules.plt.title(f'{tickerSymbol} Monte Carlo', fontsize=15, fontweight="bold", pad=10)
    modules.plt.legend(loc='lower left', prop={'size': 9})
    modules.plt.grid(True, linestyle='--')
    modules.plt.xticks(fontsize=10)
    modules.plt.yticks(fontsize=10)
    fig.autofmt_xdate()
    modules.plt.show()






