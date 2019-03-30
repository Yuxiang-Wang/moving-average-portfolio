# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:49:20 2019

@author: yuxiang
"""

import pandas_datareader.data as web
import os
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def get_df(path,file=None):
    if file:
        path=''.join([path,file])
    df=pd.read_csv(path)
    df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df = df.drop(columns=['Unnamed: 0'])
    return df    

def rebalance(path,name=None):
    df = get_df(path,name)
    
    start = df['Date'][len(df)-1].to_pydatetime().date()
    end = dt.date.today()
    ticker = name.split('.')[0]
    new = web.DataReader(ticker,'yahoo',start,end)

    new['ret'] = np.log(new['Close']/new['Close'].shift(1))
    new = new.dropna()
    new = new.reset_index()

    data = pd.concat([df,new],ignore_index=True)
    data.to_csv(''.join([path,name]))

def get_returns(stock_csv_folder,stock_csv_names):
    ret=np.zeros((1,200),'float')
    stocks=[]
    for csv_name in stock_csv_names:
        df = get_df(stock_csv_folder,csv_name)
        if len(df)<200:
            continue
        if df['Date'][len(df)-1].to_pydatetime().date() != dt.date.today():
            continue
        stocks.append(csv_name)
        r = np.array(df['ret'][-200:]).reshape(1,200)
        ret=np.vstack([ret,r])
    ret = ret[1:]
    return ret

def long_only_weights(N,num):
    weight = np.zeros((N,num))
    for j in range(N):
        for i in range(num-1):
            weight[j,i] = np.random.uniform(0,1-weight[j].sum())
    weight[:,-1] = 1-weight.sum(axis=-1)
    for i in range(N):
        np.random.shuffle(weight[i])
    return weight

def alloc(ret,cov,w):
    return ret.dot(w),np.sqrt(w.dot(cov.dot(w)))