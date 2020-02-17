# -*- coding: utf-8 -*-
"""
Cal stock performance measures

input:
    pass
output:
    all ratio results
    score

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import DataCleansing as dc
stockfileter_get=dc.GETTER()
import datetime


ref_ratios=['M2',
     'Treynor',
     'Alpha_CAPM',
     'Alpha_3f',
     'Alpha_5f',
     'Beta_mkt_CAPM',
     'Beta_mkt_3f',
     'Beta_mkt_5f',
     'Beta_all_CAPM',
     'BullBeta',
     'BearBeta',
     'MaximumDrawdown',
     'Calmar',
     'Bull_Bear_Beta',
     'Monthlyavg',
     'Information_ratio',
     'Sortino_ratio']

def timeit(method):
    def timed(*args, **kw):
        start = datetime.datetime.now()
        result = method(*args, **kw)
        end = datetime.datetime.now()
        print(method.__name__,'took ',end-start)
        return result
    return timed

# Sharpe ratio
def Sharp(rp, rf):
    '''return Sharp ratio
    portfolio:rp, not excess return
    '''
    assert type(rp) in [pd.core.series.Series,np.ndarray], 'rp should be Series'
    assert type(rf) in [pd.core.series.Series,np.ndarray], 'rf should be Series'
    mu = (rp-rf).mean()
    sv = rp.var(ddof=1)
    if np.sqrt(sv) == 0:
        return 'n/a'
    else:
        sr = mu/np.sqrt(sv)
        return sr

def M2(rp,rm,rf):
    # cal M Square 
    for count,i in enumerate([rp,rm,rf]):
        assert type(i) in [pd.core.series.Series,np.ndarray], '{}th parameter should be Series'.format(count)
    s_p = np.std(rp,ddof=1)
    s_m = np.std(rm,ddof=1)
    return (rp-rf).mean()*s_m/s_p-(rm-rf).mean()

def InfoR(rp,rb):
    # cal Information Ratio
    assert type(rp) in [pd.core.series.Series,np.ndarray], 'rp should be Series'
    assert type(rb) in [pd.core.series.Series,np.ndarray], 'rb should be Series'
    if (rp-rb).std(ddof=1) == 0:
        return 'n/a'
    else:
        return (rp-rb).mean()/(rp-rb).std(ddof=1)

#Sortino ratio (with risk-free rate as target)
def Sortino(rp, rb):
    '''
    rp = portfolio, 1darray
    rb = benchmark, 1darray
    if same, sortino ratio = 1
    '''
    assert type(rp) in [pd.core.series.Series,np.ndarray], 'rp should be Series'
    assert type(rb) in [pd.core.series.Series, int], 'Rt should be Series or '\
                        'int(like fixed rf rate)'
    mu = (rp-rb).mean()
    DR = (rp-rb)[(rp-rb)<0] # downside returns
    sv = sum(DR.values**2)/len(rp)
    if sv == 0:
        return 'n/a'
    sr = mu/np.sqrt(sv)
    return sr

#Three-factor alpha
def Factor_AB(factors,rp_ex,get=None):
    '''
    X: xs, DF
    rp: y, Series
    get: a = Alpha, b= Beta, None = all
    '''
    assert type(rp_ex) in [pd.core.series.Series,np.ndarray], 'rp_ex should be Series'
    assert type(factors) in [pd.core.frame.DataFrame,np.ndarray], 'factors should be DF'
    assert get in ['a','b',None],'get should be in a(Alpha) or b(Beta)'
    linreg = LinearRegression()
    model=linreg.fit(factors, rp_ex)
    alpha = model.intercept_ 
    beta = model.coef_
    if get=='a':
        return alpha
    elif get == 'b':
        return beta[0]
    else:
        return alpha, beta

def Treynor(rp,rm):
    # cal Treynor ratio
    assert type(rp) in [pd.core.series.Series,np.ndarray], 'rp should be Series'
    assert type(rm) in [pd.core.frame.Series,np.ndarray], 'rm should be Series'
    beta = Factor_AB(pd.DataFrame(rm),rp,get='b')
    if beta == 0:
        return 'n/a'
    ratio = (rp-rm).mean()/beta
    return ratio

def bull_beta(rp_ex,rm_ex):
    # bull market beta
    rm_ck = rm_ex
    times = rm_ck[rm_ck>0].index
    if len(times)<=1:
        return 'n/a'
    beta = Factor_AB(pd.DataFrame(rm_ex[times]),rp_ex[times],get='b')
    return beta

def bear_beta(rp_ex,rm_ex):
    # bear market beta
    rm_ck = rm_ex
    times = rm_ck[rm_ck<0].index
    if len(times)<=1:
        return 'n/a'
    beta = Factor_AB(pd.DataFrame(rm_ex[times]),rp_ex[times],get='b')
    return beta

def maxdrawdown(price):
    window = len(price)
    roll_max = price.rolling(window, min_periods=1).max()
    daily_drawdown = price/roll_max- 1.0
    return daily_drawdown.min()


def calmar(price):
    '''used exponential return as nominator
    if maximumdrawdown == 0, return 1
    '''
    mdd = maxdrawdown(price)
    if mdd == 0:
        return 'n/a'
    r = price.tolist()[-1]/price.tolist()[0]
    s=[1 if r > 1 else 0 for i in [r]][0]
    term = len(price)/12
    ar = s*abs(r)*(1/term)-1
    return ar/mdd

#@timeit
def risk_metrix(start_time = '1999-04',end_time = '2018-12',
                stock_list=[], getter=stockfileter_get):
    '''generate performance metrix for all stocks, 
    all stock report
    can specify start/end time, which are both string
    can specify stock lists
    '''
    pkg = getter.change_time(start_time,end_time,stock_list=stock_list)
    data = pkg['return']
    prices = pkg['prices']
    
    if stock_list==[]:
        stock_list=data.keys()
    out = {}
    for i in stock_list:
        out[i]=PortfPerfm(rp=data[i],price=prices[i])
    rst = pd.DataFrame(out).T
    return rst

def filt(criteria_dict,start_time = '1999-04',end_time = '2018-12',
         ratio_metrix=None,getter=stockfileter_get,
         stock_list=[],
         pct=False, top=False,bottom=False):
    '''filter with criterions. 
    criterions in form of: 
        1. str, specific rules, 
        2. top percentile, pct=True, 
        3. top n, top=True
    INPUT either getter or ratio_metrix
    args:
        criteria_dict = dict with keys same as keys in ratio_metrix
        ratio_metrix: DF, result of main function
        getter: GETTER object in DataCleansing.py
        start_time/end_time: specify when getter is inputed
        stock_list: specify stock list
    '''
    if type(ratio_metrix)==type(None):
        assert type(getter)==dc.GETTER, 'either input a getter type in parameter getter, or input a ratio_metrix'
        ratio_metrix = risk_metrix(start_time,end_time,stock_list)
    else:
        assert type(ratio_metrix)==pd.core.frame.DataFrame,'ratio_metrix should be DataFrame type'
    assert len(ratio_metrix)>0, 'Holdong No stocks'
#    print('Initing filtering stocks...')
    temp = ratio_metrix
    if pct == True:
        output=set(temp.index)
        for i in criteria_dict:
            c1 = criteria_dict[i]
            t2 = round(len(temp)*c1)+1
            temp_set = set(temp[i].sort_values(ascending=False).iloc[:t2].index)
            output = output&temp_set
        print('total gives {} samples left.'.format(len(output)))
        return list(output)
    elif top==True:
        output=set(temp.index)
        for i in criteria_dict:
            c1 = criteria_dict[i]
            temp_set = set(temp[i].sort_values(ascending=False).iloc[:c1].index)
            output = output&temp_set
        print('total gives {} samples left.'.format(len(output)))
        return list(output)     
    elif bottom==True:
        output=set(temp.index)
        for i in criteria_dict:
            c1 = criteria_dict[i]
            temp_set = set(temp[i].sort_values(ascending=True).iloc[:c1].index)
            output = output&temp_set
        print('total gives {} samples left.'.format(len(output)))
        return list(output)     
    else:
        for i in criteria_dict:
            c1 = criteria_dict[i]
            t2 =eval(' (temp[i]{})'.format(c1))
            temp = temp[t2]
            print('{} {} gives {} samples left.'.format(i,criteria_dict[i],len(temp)))
        return list(temp.index)
    
def PortfPerfm(price=None,rp=None,getter = stockfileter_get):
    '''generate performance dict for portfolio. use kw to specify input as retunr or price
    price = DF: return dict, keys = columns
    price = Series: return performance measure.
    '''
    assert type(price)!=type(None) or type(rp)!=type(None), 'price and rp cannot both be None'
    gtt = getter
    if type(price)!=type(None):
        if type(price)==pd.core.frame.DataFrame:
            rst = {}
            for lb in price.columns:
                rst[lb] = PortfPerfm(price=price[lb],getter = gtt)
            return rst
        else:
            assert type(price)==pd.core.series.Series,'price can only accept Series or DF as prices'
            if type(rp)==type(None):
                rp = (price/price.shift(1)).dropna()
    else:
        if type(rp)==pd.core.frame.DataFrame:
            rst = {}
            for lb in rp.columns:
                rst[lb] = PortfPerfm(rp=rp[lb],getter = gtt)
            return rst
        else:
            assert type(rp)==pd.core.series.Series,'rp can only accept Series or DF as prices'
            if type(price)==type(None):
                price = (1+rp).rolling(window=len(rp),min_periods=1).apply(np.prod,raw=True)
    rf = (gtt.rf()['RF']).loc[rp.index]
    rm = (gtt.sp500()['Return']).loc[rp.index]
    f3 = gtt.factor_3().loc[rp.index,:]
    f5 = gtt.factor_5().loc[rp.index,:]
    
    rm_ex = rm-rf
    rp_ex = rp-rf
    rb=rm
    
    sr = Sharp(rp, rf)
    m2 = M2(rp,rm,rf) 
    trey = Treynor(rp,rm)
    bu_beta = bull_beta(rp_ex,rm_ex)
    be_beta = bear_beta(rp_ex,rm_ex)
    alpha = Factor_AB(pd.DataFrame(rm_ex),rp_ex,get='a')
    beta = Factor_AB(pd.DataFrame(rm_ex),rp_ex,get='b')
    alpha_3f = Factor_AB(f3,rp_ex,get='a')
    beta_3f = Factor_AB(f3,rp_ex,get='b')
    beta_3f_all = Factor_AB(f3,rp_ex)[1]
    alpha_5f = Factor_AB(f5,rp_ex,get='a')
    beta_5f = Factor_AB(f5,rp_ex,get='b')
    beta_5f_all = Factor_AB(f5,rp_ex)[1]
    maxdd = maxdrawdown(price)
    cm = calmar(price)
    inf_r = InfoR(rp,rb)
    sort_r = Sortino(rp, rf)
    perform = {'SharpRatio':sr,
         'M2':m2,
         'Treynor':trey,
         'Alpha_CAPM':alpha,
         'Alpha_3f':alpha_3f,
         'Alpha_5f':alpha_5f,
         'Beta_mkt_CAPM':beta,
         'Beta_mkt_3f':beta_3f,
         'Beta_mkt_5f':beta_5f,
         'Beta_all_CAPM':beta,
         'Beta_all_3f':beta_3f_all,
         'Beta_all_5f':beta_5f_all,
         'BullBeta':bu_beta,
         'BearBeta':be_beta,
         'MaximumDrawdown':maxdd,
         'Calmar':cm,
         'Bull_Bear_Beta':bu_beta-be_beta,
         'Monthlyavg':rp.mean(),
         'Information_ratio':inf_r,
         'Sortino_ratio':sort_r}
    return perform

def Tingstep2():
    '''
    # Ting's requirement step2, find alpha return
    '''
    perform=risk_metrix()
    top10 = perform['Alpha_CAPM'].sort_values(ascending=False).iloc[:10].index
    bottom90 = perform['Alpha_CAPM'].sort_values(ascending=False).iloc[10:].index
    gap = perform['Monthlyavg'][top10].mean()-perform['Monthlyavg'][bottom90].mean()
    return gap

def GenPortfReturn(stock_list=[],start_time = '1999-04',end_time = '2018-12',
                   weight=None,getter=stockfileter_get):
    pkg = getter.change_time(start_time,end_time,stock_list=stock_list)
    data = pkg['return']
    if stock_list==[]:
        stock_list=list(data.keys())
    if weight == None:
        weight = {i:1/len(stock_list) for i in stock_list}
    else:
        assert type(weight)==dict, 'weight should be dict type.'
    assert set(weight.keys()) == set(stock_list), 'stock_list and weight length does not match.'
    rp = pd.concat([data[s]*weight[s] for s in stock_list],axis=1).apply(sum,axis=1)
    return rp

def Return_to_Price(rp):
    '''generate price from return
    '''
    window = len(rp)
    price = (rp+1).rolling(window, min_periods=1).apply(np.prod,raw=True)
    return price
    
if __name__ == '__main__':

    st = '2010-04'
    ed = '2013-04'
    portfolio_value = stockfileter_get.marketprice()
    stock_l = ['1762','3938']
    
    '''Tutorial'''
    
    '''T1. performance matrix '''
    # get financial ratios
    # support specifying stock_list, start/end time
    perform_timesort=risk_metrix(start_time=st,end_time=ed,
                            stock_list=stock_l)
    # by default get for all stock
    perform=risk_metrix()
    
    '''T2. select stocks (input risk metrix/stock_list)'''
    criteria_dict={
               'Bull_Bear_Beta':'>-0.3',
               'Treynor':'>0' # positive premium for market risk
               }
    # select1, by all time, input risk metrix
    selected_stocks1 = filt(criteria_dict,ratio_metrix=perform)
    # select2, by specifying time and stock_list
    selected_stocks2 = filt(criteria_dict,start_time=st,end_time=ed,
                            stock_list=stock_l)
    
    # filter by rank percentage
    criteria_dict_pct={
               'Bull_Bear_Beta':0.1,
               'Treynor':0.5 # positive premium for market risk
               }
    selected_stocks3 = filt(criteria_dict_pct,ratio_metrix=perform,
                            pct=True)
    
    '''T3. portfolio performance check '''
    performance = PortfPerfm(price=portfolio_value,rp=None)
    
    '''T4. Generate Portfolio Return with Stock_list'''
    new_port_r = GenPortfReturn(stock_list=[],weight=None) # weight should be dict if specified
    new_porformance = PortfPerfm(rp=new_port_r)
    
    '''Tutorial End'''
    TingsReturn = Tingstep2()
    
    ''' get performance results for long top 10 Treynor ratio '''
    st = '2010-04'
    ed = '2013-04'
    # in 3 lines
    stocks = filt({'Treynor':0.1},start_time=st,end_time=ed,
                            pct=True)
    port_return = GenPortfReturn(stock_list=stocks,start_time=st,end_time=ed,
                                 weight=None)
    performance=PortfPerfm(price=None,rp=port_return)
    performance
    
    # in 1 line
    performance=PortfPerfm(price=None,rp=GenPortfReturn(
            stock_list=filt(
                    {'Treynor':0.1},start_time=st,end_time=ed,pct=True
                    ),start_time=st,end_time=ed,weight=None)
    )
    performance
    
if False:
    '''plot ratio distribution'''
    st = '1994-04'
    ed = '2017-01'
    perform=risk_metrix(start_time=st,end_time=ed)
    # plot hist for ratios
    plot_table = ['SharpRatio','Treynor',
                  'Alpha_CAPM','Alpha_3f','Alpha_5f',
                  'MaximumDrawdown','Monthlyavg','Bull_Bear_Beta',
                  ]
    for i in plot_table:
        fig, ax = plt.subplots()
        perform[i].plot.hist(bins=20,ax=ax)
        ax.set_title(i)
    
    # plot bull-bear beta scatter
    fig, ax = plt.subplots()
    plt.ylim((-0.5,3.5))
    plt.xlim((-0.5,3.5))
    ax.set_title('Bull/Bear market beta')
    perform[['BullBeta','BearBeta']].astype(float).plot.scatter(
            ax=ax,x='BullBeta',y='BearBeta')



    
    
    
    
