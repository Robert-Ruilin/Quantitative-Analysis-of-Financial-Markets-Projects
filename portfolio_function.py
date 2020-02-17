import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import DataCleansing as getdata
import StockFilter as sf
import Regress
import portfolio_function as pf

def getRp(portfolio_return):
    weigh = 1/len(portfolio_return)
    Rp = portfolio_return.apply(np.mean,axis=1)
    return Rp

def getBalance(portfolio_price, capital=100):
    balance = pd.DataFrame(index=portfolio_price.index,columns=portfolio_price.columns)
    money = capital/len(portfolio_price.columns)
    for stock in portfolio_price.columns:
        volume=money/portfolio_price[stock].iloc[0]
        balance[stock] = portfolio_price[stock]*volume
    return balance, balance.apply(sum,axis=1)        

def getSharpe(Rp,Rf):
    excessRp = Rp.values - Rf.values
    sharpe = excessRp.mean()/Rp.values.std(ddof=1)
    return sharpe

def getR(Price):
    R = ((Price - Price.shift(1))/Price.shift(1)).dropna()
    return R

def getReturn(price):
    return ((price-price.shift(1))/price.shift(1)).dropna()

def getSet(df,ratio=3):
    testSize = len(df)//ratio
    sampleSize = len(df)-testSize
    sampleSet = df.iloc[:sampleSize]
    testSet = df.iloc[sampleSize:]
    return sampleSet, testSet

def optimize_P(pool,stock_return,Rf):
    maxRp = getRp(stock_return[pool])
    maxSharpe = getSharpe(maxRp,Rf)
    maxP = pool
    for stock in pool:
        new_pool = list(pool).copy()
        new_pool.remove(stock)
        Rp = getRp(stock_return[new_pool])
        Sharpe = getSharpe(Rp,Rf)
        if Sharpe > maxSharpe:
            maxSharpe = Sharpe
            maxRp = pool
            maxP = new_pool    
    return maxP,maxSharpe

def getPortfolio(pool,stock_return,Rf,min_num = 10):
    pool = pool
    Sharpe = getSharpe(getRp(stock_return[pool]),Rf)    
    for i in range(len(pool)-min_num):
        new_pool, new_Sharpe = optimize_P(pool,stock_return,Rf)
        if set(new_pool) == set(pool):
            break
        pool, Sharpe = new_pool, new_Sharpe
    return pool, Sharpe

#plot
def plot(ax,benchmark,stocks,portfolio,sp,title):
    ax.plot(benchmark)
    ax.plot(stocks)
    ax.plot(portfolio,'r-')
    ax.plot(sp,'k-')
    
    ax.grid(axis='y')
    #ytick
    ticklabels1 = set()
    ticks1 = []
    for i in portfolio.index:
        if i[:4] not in ticklabels1:
            ticks1.append(i)
        ticklabels1.add(i[:4])
    ax.set_xticks(ticks1)
    ax.set_xticklabels([i[:4] for i in ticks1])
    ax.legend(['benchmark','stocks','portfolio','S&P'],loc='upper left')
    ax.set_title(title)
    ax.set_ylabel('balance')

#plot long short
def plotLongShort(ax,portfolio,title):
    ax.plot(portfolio,'r-')    
    ax.grid(axis='y')
    #ytick
    ticklabels1 = set()
    ticks1 = []
    for i in portfolio.index:
        if i[:4] not in ticklabels1:
            ticks1.append(i)
        ticklabels1.add(i[:4])
    ax.set_xticks(ticks1)
    ax.set_xticklabels([i[:4] for i in ticks1])
    ax.legend(['LongShort'],loc='upper left')
    ax.set_title(title)
    ax.set_ylabel('balance')

#report portfolio result
def report(p,Rf,Rm):
    res = pd.DataFrame()
    for i in p:
        Rf = Rf.loc[getR(i).index]
        Rm = Rm.loc[getR(i).index]
        result = dict()
        result['Sharpe Ratio']=round(getSharpe(getR(i),Rf),3)
        result['Treynor Ratio']=round(sf.Treynor(getR(i),Rm),5)
        result['Alpha_CAPM']=round(sf.Factor_AB(pd.DataFrame(Rm-Rf),getR(i)-Rf,get='a'),5)      
        result['M2']=round(sf.M2(getR(i),Rm,Rf),3)
        result['Information Ratio']=round(sf.InfoR(getR(i),Rf),3)
        result['Sortino Ratio']=round(sf.Sortino(getR(i), Rf),3)
        result = pd.DataFrame((result),index=[0])
        res = pd.concat([res,result])
    res = res.T
    res.columns = ['Benchmark','Stocks','Portfolio','S&P']
    return res

def getLongShort(poolR, f5, time):
    corr = poolR.corr()
    corr_pool = []
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if i > j:
                if corr.iloc[i,j]> 0.7:
                    corr_pool.append([corr.columns[i],corr.columns[j]])
    result={}
    x = f5.loc[time,:]
    y = poolR
    x=x.copy()
    x.insert(0,'constant',np.ones((len(x.index),)),True)
    for i in range(len(y.columns)):
        dep=x.copy().iloc[:,0]
        Reg=sm.OLS(y.iloc[:,i].values,dep.values).fit()
        coef=Reg.params
        result[y.columns[i]]=coef[0]

    long = []
    short = []
    for i in corr_pool:
        if result[i[0]]>result[i[1]]:
            long.append(i[0])
            short.append(i[1])
        else:
            long.append(i[1])
            short.append(i[0])
    return long, short