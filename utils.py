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
    return data, tickerSymbol

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
    return n, fcast_dates

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
    return mc_mid_pred, min_values_mc, max_values_mc




    ### Need to continue from here
    #Plan: predict the vol, calculate back the potention option prices and then trade based off of that.


def arma_garch_model(data,n,sim=10000,day_type='1d',mark_type='Close',arma_or_not="True",
                    optimizer="True", combined="True", vol_type='GARCH', dist_type='normal',
                    mean_type='constant', o_type=0):
                    if combined == "False":
                        if arma_or_not=="False":
                            def ARMA(optimizer):
                                if optimizer == True:
                                    order_aic_bic=[]
                                    for p in range(4):
                                        for q in range(4):
                                            try:
                                                model=modules.SARIMAX(np.log(data[mark_type]),trend='c', order=(p,1,q))
                                                results=model.fit()
                                                order_aic_bic.append((p,q,results.aic, results.bic))
                                            except:
                                                order_aic_bic.append((p,q, None, None))
                                    order_df=modules.pd.DataFrame(order_aic_bic, columns=['p', 'q', 'AIC', 'BIC'])
                                    print(order_df.sort_values('AIC'))
                                    print(order_df.sort_values('BIC'))
                                    while (True):
                                        try:
                                            p_input=int(float(input('Give the first parameter for ARIMA(p,x,x): ')))
                                            d_input=int(float(input(f'Give the second parameter for ARIMA({p_input},d,x): ')))
                                            q_input=int(float(input(f'Give the third parameter for ARIMA({p_input},{d_input},q): ')))
                                            break
                                        except ValueError:
                                            print("Give whole number, maximum parameter limit is 3.")
                                            continue
                                    model=modules.SARIMAX(np.log(data[mark_type]),trend='c', order=(p_input,d_input,q_input))
                                    results=model.fit()
                                    print(results.summary())
                                    forecast=results.get_forecast(steps=n)
                                    mean_forecast=modules.np.exp(forecast.predicted_mean)
                                    confidence_intervals=modules.np.exp(forecast.conf_int())
                                    lower_limits=confidence_intervals.loc[:,f"lower {mark_type}"]
                                    upper_limits=confidence_intervals.loc[:,f"upper {mark_type}"]
                                    
                                    params_arima=results.params.index.values.tolist()
                                    mu=results.params['intercept']
                                    if 'ar.L1' in params_arima:
                                        phi1=results.params['ar.L1']
                                    elif 'ar.L2' in params_arima:
                                        phi2=results.params['ar.L2']
                                    elif 'ar.L3' in params_arima['ar.L3']:
                                        phi3=results.params['ar.L3']
                                    else:
                                        phi1=0
                                    if 'ma.L1' in params_arima:
                                        theta1=results.params['ma.L1']
                                    elif 'ma.L2' in params_arima:
                                        theta1=results.params['ma.L2']
                                    elif 'ma.L3' in params_arima:
                                        theta1=results.params['ma.L3']
                                    else:
                                        theta1=0

                                else:
                                    p_input=1
                                    d_input=1
                                    q_input=1
                                    model=modules.SARIMAX(np.log(data[mark_type]),trend='c', order=(p_input,d_input,q_input))
                                    results=model.fit()
                                    print(results.summary())
                                    forecast=results.get_forecast(steps=n)
                                    mean_forecast=modules.np.exp(forecast.predicted_mean)
                                    confidence_intervals=modules.np.exp(forecast.conf_int())
                                    lower_limits=confidence_intervals.loc[:,f"lower {mark_type}"]
                                    upper_limits=confidence_intervals.loc[:,f"upper {mark_type}"]
                                    params_arima=results.params.index.values.tolist()
                                    mu=results.params['intercept']
                                    if 'ar.L1' in params_arima:
                                        phi1=results.params['ar.L1']
                                    elif 'ar.L2' in params_arima:
                                        phi2=results.params['ar.L2']
                                    elif 'ar.L3' in params_arima['ar.L3']:
                                        phi3=results.params['ar.L3']
                                    else:
                                        phi1=0
                                    if 'ma.L1' in params_arima:
                                        theta1=results.params['ma.L1']
                                    elif 'ma.L2' in params_arima:
                                        theta1=results.params['ma.L2']
                                    elif 'ma.L3' in params_arima:
                                        theta1=results.params['ma.L3']
                                    else:
                                        theta1=0

                            ARMA(optimizer)
                        else:
                            def GARCH(optimizer):
                                data['log_return_garch']=data['log_return'].mull(100)
                                if optimizer == True:
                                    order_aic_bic_garch=[]
                                    for p_garch in range(4):
                                        for q_garch in range(4):
                                            try:
                                                model_garch=modules.arch_model(data['log_return_garch'], p=p_garch, q=q_garch, o=o_type, vol=vol_type, dist=dist_type, mean=mean_type)
                                                results_garch_test=model_garch.fit(disp='off')
                                                order_aic_bic_garch.append((p_garch,q_garch,results_garch_test.aic, results_garch_test.bic))
                                            except:
                                                order_aic_bic_garch.append((p,q, None, None))
                                    order_garch_df=pd.DataFrame(order_aic_bic_garch, columns=['p', 'q', 'AIC', 'BIC'])
                                    print(order_garch_df.sort_values('AIC'))
                                    print(order_garch_df.sort_values('BIC'))

                                    while (True):
                                        try:
                                            p_input_garch=int(float(input('Give the first parameter for GARCH(p,x)-t: ')))
                                            q_input_garch=int(float(input(f"Give the second parameter for GARCH({p_input_garch},q)-t: ")))
                                            break
                                        except ValueError:
                                            print("Give whole number, maximum parameter limit is 3.")
                                            continue

                                    gm_rev=modules.arch(data['log_return_garch'], p=p_input_garch, q=q_input_garch, o=o_type, vol=vol_type, dist=dist_type)
                                    gm_rev_result=gm_rev.fit(disp='off')
                                    print(gm_rev_result.summary())

                                    ##GARCH-parameter 
                                    omega=gm_rev_result.params['omega']
                                    for i in range(1,p_input_garch):
                                        for i in range(1,q_input_garch):
                                            if p_input_garch == 0:
                                                local()[f'alpha_{0}']=0
                                            elif q_input_garch == 0:
                                                local()[f'beta_{0}']=0
                                            else:
                                                local()[f'alpha_{i}']=gm_rev_result.params[f'alpha{i}']
                                                local()[f'beta_{i}']=gm_rev_result.params[f'beta{i}']
                                    
                                    sigma_t=np.sqrt(omega + alpha * (gm_rev_result.resid**2).shift(1) + beta*(gm_rev_result.conditional_volatility**2).shift(1)) ##GARCH 1,1 sigma
                                    epsilon_t=sigma_t*np.random.standard_normal(len(sigma_t))
                                    df['forecast_garch']=mu+phi1*df['Log_return_garch'].shift(1)+epsilon_t+theta1*epsilon_t.shift(1)
                                    df['sigma_t']=sigma_t

                                    df['var_garch']=(mu+sigma_t*((norm.ppf(0.99))*-1))

                                    df=df.dropna()
                                    index_col=df.index.values.tolist()
                                    index_col=pd.to_datetime(index_col)
                                    index_col=pd.DataFrame(index_col, columns=['Date'])
                                    index_col['date']=index_col['Date']
                                    index_col.set_index('Date', inplace=True)

                                    abs_r=abs(df['Log_return']) #
                                    abs_r=abs_r*100


                                    while (True):
                                        log_or_eff_or_arima=input('Log vagy Effektív hozam alapján jelezzen előre (alap="Log")') or 'Log_return_garch'
                                        if log_or_eff_or_arima in opcio_garch:
                                            break
                                        else:
                                            continue	
                                    if log_or_eff_or_arima in opcio_log_eff:
                                        train=df[log_or_eff_or_arima]
                                    else:
                                        train=resid_arima_garch
                                    if mean_type == '':
                                        model_f=arch_model(train, p=p_input_garch, q=q_input_garch, o=o_type, vol = vol_type , dist = dist_type)
                                    else:
                                        model_f=arch_model(train, p=p_input_garch, q=q_input_garch, o=o_type, vol = vol_type , dist = dist_type, mean = mean_type)
                                    model_f_fit = model_f.fit(disp='off')

                                    pred = model_f_fit.forecast(horizon=n)
                                    pred_std = np.sqrt((pred.variance.values[-1,:])/10000) #variance 100*100
                                    pred_std=list(pred_std)
                                    pred_return=np.zeros(shape=(10))
                                    pred_return[0]= mu+phi1*df['Log_return'][-1]+pred_std[0]*np.random.standard_normal(1)+theta1*pred_std[0]*np.random.standard_normal(1)

                                    bound_n=1000
                                    bounds=np.zeros(shape=(10,bound_n))
                                    bounds[0,:]=df['Close'][-1]*math.exp(pred_return[0])

                                    for j in range(1,bound_n):
                                        for i in range(1,10):
                                            pred_return[i]=mu+phi1*pred_return[i-1]+pred_std[i]*np.random.standard_normal(1)+theta1*pred_std[i-1]*np.random.standard_normal(1)
                                            bounds[i,j]=bounds[i-1,j]*math.exp(pred_return[i])

                                    max_values_ar_garch = []
                                    min_values_ar_garch = []
                                    for i in range(1, 11):
                                        min_value_ar_garch = min(bounds[i-1,1:bound_n])
                                        min_values_ar_garch.append(min_value_ar_garch)
                                        max_value_ar_garch = max(bounds[i-1,1:bound_n])
                                        max_values_ar_garch.append(max_value_ar_garch)
                                    difference=[]
                                    pred_transform_middle = zip(max_values_ar_garch,min_values_ar_garch)
                                    for list1_i, list2_i in pred_transform_middle:
                                        difference.append(list1_i-list2_i)
                                    for i in range(1,11):
                                        difference[i-1]=(difference[i-1]/2)+min_values_ar_garch[i-1]
                                else:
                                    pass
                            GARCH(optimizer)
                    else:
                        ARMA(optimizer)
                        GARCH(optimizer)
                                    #                         elif log_or_eff_or_arima not in opcio_log_eff and log_or_eff_or_arima != "resid_arima_garch":
                                    #     if mean_type =='':
                                    #         gm_rev=arch_model(abs_r, p=1, q=1, o=o_type, vol=vol_type, dist=dist_type)
                                    #     else:
                                    #         gm_rev=arch_model(abs_r, p=1, q=1, o=o_type, vol=vol_type, dist=dist_type, mean=mean_type)
                                    # else:
                                    #     if mean_type =='':
                                    #         gm_rev=arch_model(resid_arima_garch, p=p_input_garch, q=q_input_garch, o=o_input_garch, vol=vol_type, dist=dist_type)
                                    #     else:
                                    #         gm_rev=arch_model(resid_arima_garch, p=p_input_garch, q=q_input_garch, o=o_input_garch, vol=vol_type, dist=dist_type, mean=mean_type)
                                    # params_arima=results.params.index.values.tolist()
                                    # mu=results.params['intercept']
                                    # if 'ar.L1' in params_arima:
                                    #     phi1=results.params['ar.L1']
                                    # else:
                                    #     phi1=0
                                    # if 'ma.L1' in params_arima:
                                    #     theta1=results.params['ma.L1']
                                    # else:
                                    #     theta1=0

# Functions for plotting

def plot_logr_price(data,tickerSymbol, mark_type='Close'):
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

def plot_predicted(data, mid, lower, upper, fcast_dates, tickerSymbol, mark_type='Close', model_name='MC'):
    try:
        fig = modules.plt.figure(figsize=(8, 4.5))
        modules.plt.plot(data.loc[modules.datetime.date.today()-modules.datetime.timedelta(days=31):, mark_type], label='Price', color='black', linewidth=2)
        modules.plt.plot(fcast_dates, mid, color='blue', label='Predicted', linewidth=3)
        modules.plt.fill_between(fcast_dates, lower, upper, color='grey', alpha=0.3)
        modules.plt.xlabel('Date', fontsize=15, labelpad=5)
        modules.plt.ylabel('Price', fontsize=15, labelpad=5)
        modules.plt.title(f'{tickerSymbol} {model_name}', fontsize=20, fontweight="bold", pad=10)
        modules.plt.legend(loc='lower left')
        modules.plt.grid(True, linestyle='--')
        modules.plt.xticks(fontsize=12)
        modules.plt.yticks(fontsize=12)
        fig.autofmt_xdate()
        modules.plt.show()
    except KeyError as e:
        print(f"KeyError: {e}. Please ensure that the data and column are correctly specified.")
    except Exception as e:
        print(f"An error occurred: {e}.")






