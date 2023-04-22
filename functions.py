import modules
from modules import datetime, yf

def load_financial_data(tickerSymbol='META', start_date='2014-01-01', end_date=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')):
    tickerData=yf.Ticker(tickerSymbol)
    df=tickerData.history(period='1d', start=start_date, end=end_date)
    print(df)

