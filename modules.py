import datetime
import math
from ntpath import join
import numbers
import os
import random
### This file import all the necessary modules for the visualization and computing ###

import sys
import warnings
from cgitb import scanvars
from re import A
from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import pylab
import statsmodels.tsa.stattools as tsa
import statsmodels.stats as sm_stat
from statsmodels.tsa.stattools import zivot_andrews
import yfinance as yf
from arch import arch_model
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from pandas_datareader import test
from scipy import stats
import random
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import moment, norm, probplot, t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, q_stat
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.system("cls")

