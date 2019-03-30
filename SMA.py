# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:33:07 2019

@author: yuxiang
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


stock_csv_folder = "C:/Users/yuxiang/Desktop/alpha/stocks/"
stock_csv_names = sorted(os.listdir(stock_csv_folder))
 
def SMA(price,window):
    temp = price[-window:]
    pSMA = np.mean(temp)
    return pSMA
def SMA2(price,window):
    w=np.repeat(1.0,window)/window
    pSMA = np.convolve(price,w,'valid')
    return pSMA

def getDf(stock_csv_folder,csv_name):
    df = pd.read_csv(os.path.join(stock_csv_folder,csv_name))
    df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
    return df

#def calRet(stock_csv_folder,stock_csv_names):
#    for csv_name in stock_csv_names: # csv_name = stock_csv_names[0]
#        df = pd.read_csv(os.path.join(stock_csv_folder,csv_name))
#        df['Date'] = pd.to_datetime(df['Date'],format = '%Y-%m-%d')
#        df['ret'] = np.log(df['Close']/df['Open'])
#        df.to_csv(os.path.join(stock_csv_folder,csv_name))

#calRet(stock_csv_folder,stock_csv_names)

tab = pd.DataFrame()
count=0
for csv_name in stock_csv_names: # csv_name = stock_csv_names[0]
    df = getDf(stock_csv_folder,csv_name)
    df['lagRet'] = df['ret'].shift(-1)
    price = df['Close']
    
    if len(price)<200:
        continue
    
    SMA5 = SMA2(price,5)
    SMA50 = SMA2(price,50)
    SMA200 = SMA2(price,200)
    
    posiSMA5 = (SMA5>price[4:]).astype(int)
    posiSMA50 = (SMA50>SMA5[45:]).astype(int)
    posiSMA200 = (SMA200>SMA50[150:]).astype(int)
    
    n = len(posiSMA200)

    posi = pd.DataFrame({'date':df['Date'][-n:],'price':df['Close'][-n:],
                         'SMA5':SMA5[-n:],'SMA50':SMA50[-n:],'SMA200':SMA200,
                         'posiSMA5':posiSMA5[-n:],'posiSMA50':posiSMA50[-n:],'posiSMA200':posiSMA200,
                         'ret':df['lagRet'][-n:],'volume':df['Volume'][-n:]})

    tab = pd.concat([tab,posi[(posi.posiSMA5==1) & (posi.posiSMA50==1) & (posi.posiSMA200==1)]])
    count+=1
    print(count)    


##### ret
(tab['ret']>0).sum()
len(tab)

plt.plot(tab['ret'])
plt.show()
np.mean(tab['ret'])

thresh = 0.1
plt.hist(tab[(tab['ret']< thresh) & (tab['ret']>-thresh)]['ret'],bins=10)
plt.show()

((tab['ret']< thresh) & (tab['ret']>0)).sum()
(tab['ret']>0).sum()
(tab['ret']<0).sum()

tab['ret'][tab['ret']>0].sum()
tab['ret'].sum()

plt.scatter(tab['date'],tab['ret'])
plt.show()

####### ret and diff
tab['diff5']=(tab['SMA5']-tab['price'])/tab['price']
tab['diff50'] = tab['SMA50']-tab['SMA5']/tab['SMA5']
tab['diff200'] = tab['SMA200']-tab['SMA50']/tab['SMA50']
tab['const'] = 1


X=tab[['const','diff5','diff50','diff200']].to_numpy()
y=tab['ret']
reg = sm.OLS(y, X).fit()
print(reg.summary())

np.corrcoef([tab['diff5'],tab['diff50'],tab['diff200']])

plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
plt.scatter(tab['diff5'],y)
plt.title('diff5')
plt.subplot(3,1,2)
plt.scatter(tab['diff50'],y)
plt.title('diff50')
plt.subplot(3,1,3)
plt.title('diff200')
plt.scatter(tab['diff200'],y)


