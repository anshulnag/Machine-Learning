import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib as mp
# import data
def get_data(stocks, start, end):
    # stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = yf.download(stocks, start,end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
print(stocks)
print(startDate)
print(endDate)
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)