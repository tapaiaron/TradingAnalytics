from arch import arch_model
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

def load_data_yahoo(tickerSymbol='META',period='1d', start_date= (datetime.date.today()-relativedelta(years=20)).strftime('%Y-%m-%d'), end_date=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')):
    tickerData=yf.Ticker(tickerSymbol)
    data=tickerData.history(period, start=start_date, end=end_date)
    return data, tickerSymbol

data, tickerSymbol=load_data_yahoo(tickerSymbol='META', period='1d')


data['log_return_garch']=np.log(data['Close']).diff().mul(100)
data=data.dropna()

print(type(data['log_return_garch']))
print(data['log_return_garch'])


test=arch_model(data['log_return_garch'], p=1, q=1, vol='GARCH', o=0, dist='normal', mean='constant')
test_fit=test.fit()

print(test_fit.summary())

