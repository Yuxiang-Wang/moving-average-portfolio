# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:44:07 2019

@author: yuxiang
"""

#import pandas_datareader
#from pandas_datareader.famafrench import get_available_datasets
from pandas_datareader import data
import os
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

end = dt.date.today()
start = end - dt.timedelta(365)

########## market
nasdaq = data.DataReader('^IXIC','yahoo',start,end)
nasdaq['mkt'] = np.log(nasdaq['Close']/nasdaq['Open'])
mkt = nasdaq['mkt'].to_numpy() * 100

########## risk free
rfRaw = data.DataReader('^IRX','yahoo',start,end)
rf = 0.5*(rfRaw['Open']+rfRaw['Close']).to_numpy() / 250

mktpre = mkt - rf
len(rf),len(mkt)

#plt.plot(nasdaq.index,nasdaq['Close'])

########## stock list
# http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
stockList = pd.read_csv('nasdaqlisted.txt',sep='|')[:-1] 
ticker = stockList['Symbol']

########## get stocks data

stock_csv_folder = "C:/Users/yuxiang/Desktop/alpha/stocks/"

error = []
for i,sym in enumerate(ticker):  # sym = 'AAPL'
    try:
        price = data.DataReader(sym,'yahoo',start,end)
        path = ''.join([stock_csv_folder,sym,'.csv'])
        price.to_csv(path)
        print('%d %s done' % (i,sym))
    except:
        error.append(sym)
        pass

###################################

stock_csv_folder = "C:/Users/yuxiang/Desktop/alpha/stocks/"
stock_csv_names = sorted(os.listdir(stock_csv_folder))

 
def SMA(price,window):
    temp = price[-window:]
    #w=np.repeat(1.0,window)/window
    #pSMA = np.convolve(price,w,'valid')
    pSMA = np.mean(temp)
    return pSMA

hold = {}
for csv_name in stock_csv_names: # csv_name = stock_csv_names[0]
    df = pd.read_csv(os.path.join(stock_csv_folder,csv_name))
    df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
    price = df['Close']
    if len(df)<200:
        continue
    if df['Date'][len(df)-1].to_pydatetime().date() != dt.date.today():
        continue

    SMA1_5,SMA5_50,SMA50_200=0,0,0
    if SMA(price,1) > SMA(price,5):
        SMA1_5 = 1
    if SMA(price,5) > SMA(price,50):
        SMA5_50 = 1
    if SMA(price,50) > SMA(price,200):
        SMA50_200 = 1
    if SMA1_5 or SMA5_50 or SMA50_200:
        hold[csv_name.split('.')[0]]=[SMA1_5,SMA5_50,SMA50_200,]

cols = ['SMA1_5','SMA5_50','SMA50_200']
hold_df = pd.DataFrame.from_dict(hold,columns=cols,orient='index')

hold_df[(hold_df[cols[0]]==1) & (hold_df[cols[1]]==1) & hold_df[cols[2]]==1]

hold_stock = list(hold_df.index)
np.random.shuffle(hold_stock)
hold_stock[:15]
len(hold_stock)
print(hold_stock[:15])
print(sorted(hold_stock)[:15])
################################## 
#        ret = np.log(price['Close']/price['Open']).to_numpy() * 100
#        retpre = ret - rf
#        cov = np.cov(retpre,mktpre)
#        tbeta = cov[0][1]/cov[1][1]
#        talpha = np.mean(retpre) - tbeta*np.mean(mktpre)
#        result[sym]=(talpha,tbeta)
#        print('%d %s done' % (i,sym))
#
#alpha_beta = pd.DataFrame.from_dict(result,orient='index')
#alpha_beta.sort_values(by=0)
#alpha_beta.sort_values(by=1)
#alpha_beta.columns
#
#
#alpha_beta[alpha_beta.index=='AMZN']
#'AMZN' in list(ticker)
########################################