# System modules
import subprocess
import datetime
import math
from ntpath import join
import numbers
import os
import random
import sys
import warnings
from cgitb import scanvars
from re import A
from tkinter import font

# Install function for the modules
def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Installing the modules
install("matplotlib")
install("numpy")
install("pandas")
install("seaborn")
install("statsmodels")
install("yfinance")
install("arch")
install("scipy")
install("scikit-learn")
install("tensorflow")
install("keras")
install("keras_tuner")

# Importing the modules
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

# Ignore errors
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.system("cls")

