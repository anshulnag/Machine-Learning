import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
# import data
def get_data(stocks, start, end):
    # stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = yf.download(stocks, start,end)
    stockData = stockData['Close']
    print(stockData)
    returns = stockData.pct_change()
    print(returns)
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

weights=np.random.random(len(meanReturns))
weights /=np.sum(weights)
print(weights)

#monte carlo method
mc_sims=2000#number of simulations
T=100#timeframe in days

meanM=np.full(shape=(T,len(weights)), fill_value=meanReturns)
print(meanM)
meanM=meanM.T

portfolio_sims=np.full(shape=(T,mc_sims),fill_value=0.0)

initialPortfolio=10000
for m in range(0,mc_sims):
    #MCloop
    Z=np.random.normal(size=(T,len(weights)))
    L=np.linalg.cholesky(covMatrix)
    dailyReturns=meanM+np.inner(L,Z)
    portfolio_sims[:,m]=np.cumprod(np.inner(weights,dailyReturns.T)+1)*initialPortfolio
    
plt.plot(portfolio_sims)
plt.ylabel('Portfolio value($)')
plt.xlabel('Days')
plt.axhline(initialPortfolio, color='black')
plt.title('MC simulation of a stock portfolio')
plt.show()