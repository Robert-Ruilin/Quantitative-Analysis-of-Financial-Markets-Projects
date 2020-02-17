# -*- coding: utf-8 -*-
"""
Data Cleansing
"""

import numpy as np
import pandas as pd
import datetime

def timeit(method):
    def timed(*args, **kw):
        start = datetime.datetime.now()
        result = method(*args, **kw)
        end = datetime.datetime.now()
        print(method.__name__,'took ',end-start)
        return result
    return timed

class GETTER():
    data_folder1 = '2018'
    data_folder2 = 'Data'
    stock_return={}
    benchmark_return={}
    benchmark_price={}
    stock_close={}
    pkg_stock={}
    
    def __init__(self,start = '1999-04-01',end = '2018-12-31'):
        self.start_period = pd.Timestamp(start)
        self.end_period = pd.Timestamp(end)
        self.rf()
        self.sp500()
#        self.all_stock_return()
#        self.factor_3()
#        self.factor_5()
    
    def get_stock_names(self):
#        print('getting stock names...')
        if 'files' in self.benchmark_price.keys():
            return self.benchmark_price['files']
        df1=pd.read_excel(self.data_folder2+'\\Project_603.xlsx')
        files=df1[df1['Team']==3]['Stock'].tolist()
        self.benchmark_price['files']=files
        return files
    
    def filter_monthend(self,raw):
        # get month end data, index = month id
        # month_end = raw.reset_index().groupby(pd.DatetimeIndex(raw.index).to_period('M')).max()
        month_end=raw.resample('M').last()
        return month_end
    
    def filter_timerange(self,data, start_period=None, end_period=None):
        if start_period==None:
            start_period=self.start_period
        if end_period==None:
            end_period=self.end_period
        # select time range
        data = data[start_period<=data.index]
        data = data[end_period>=data.index]
        return data
    
    def get_1stock_return(self,name):
        if name in self.stock_return:
            return self.stock_return[name]
        file_path = self.data_folder1+'\\'+name
        raw = pd.read_csv(file_path,index_col=0,parse_dates =True)
        file_name = file_path[-8:-4]
        # find month end dates
        monthly_data = self.filter_monthend(raw)
        # get Close
        raw=self.filter_timerange(monthly_data['Close'])
        raw.index=raw.index.astype(str).map(lambda x:x[:7])
        self.stock_close[file_name] = raw
        # generate return
        r = monthly_data['Close']/monthly_data['Close'].shift(1)-1
        r = self.filter_timerange(r)
        r.index=r.index.astype(str).map(lambda x:x[:7])
        # excess return
        r-=self.rf()['RF']
        # save
        self.stock_return[file_name] = r
    
#    @timeit
    def all_stock_return(self):
        # get total file list
#        print('getting stock data')
        files = self.get_stock_names()
        names = [s[-8:-4] for s in files]
        update = list(set(names) - set(self.stock_return.keys()))
        for name in update:
            # get data
            if name not in self.stock_return.keys():
                self.get_1stock_return(name+'.csv')
        return self.stock_return
    
    def factor_3(self):
        # changed market to sp500
        name = 'factor3'
#        print('getting 3factor data')
        if name in self.benchmark_return:
            return self.benchmark_return[name]
        sp=self.sp500()
        three_factors = pd.read_csv(self.data_folder2+'\\3_Factors.csv',
                                    skiprows=3,index_col=0,nrows=1119)
        three_factors.index = pd.to_datetime(three_factors.index,format='%Y%m')
        rst = self.filter_timerange(three_factors)
        rst.index=rst.index.astype(str).map(lambda x:x[:7])
        rst = pd.merge((rst/100),sp,left_index=True,right_index=True)
        rst['Mkt-RF'] = rst['Return']-rst['RF']
        self.benchmark_return['factor3']=  rst[['Mkt-RF','SMB','HML']]
        return self.benchmark_return[name]
    
    def rf(self):
        name = 'rf'
        if name in self.benchmark_return:
            return self.benchmark_return[name]
#        print('getting rf data from factor3')
        three_factors = pd.read_csv(self.data_folder2+'\\3_Factors.csv',
                                    skiprows=3,index_col=0,nrows=1119)
        three_factors.index = pd.to_datetime(three_factors.index,format='%Y%m')
        rst = self.filter_timerange(three_factors)
        rst.index=rst.index.astype(str).map(lambda x:x[:7])
        self.benchmark_return[name]=  rst[['RF']]/100
        return self.benchmark_return[name]

    def factor_5(self):
        # changed market to sp500
        name = 'factor5'
#        print('getting 5factor data')
        if name in self.benchmark_return:
            return self.benchmark_return[name]
        sp=self.sp500()
        five_factors = pd.read_csv(self.data_folder2+'\\5_Factors.csv',
                                   skiprows=3,index_col=0,nrows=675)
        five_factors.index = pd.to_datetime(five_factors.index,format='%Y%m')
        # revise date range
        rst = self.filter_timerange(five_factors)
        # align index
        rst.index=rst.index.astype(str).map(lambda x:x[:7])
        # adj returns
        rst = pd.merge((rst/100),sp,left_index=True,right_index=True)
        rst['Mkt-RF'] = rst['Return']-rst['RF']
        self.benchmark_return[name]= rst[['Mkt-RF','SMB','HML','RMW','CMA']]
        return self.benchmark_return[name]
    
    def sp500(self):
        if 'sp500' in self.benchmark_return:
            return self.benchmark_return['sp500']
        else:
            raw = pd.read_csv('Data\\^GSPC.csv',index_col=0,parse_dates=True)
            data=self.filter_monthend(raw)
            # adj price
            price = data[['Close']]
            price = self.filter_timerange(price)
            price.index=price.index.astype(str).map(lambda x:x[:7])
            self.benchmark_price['sp500']= price
            # adj return
            data['Return'] = data['Close']/data['Close'].shift(1)-1
            rst = self.filter_timerange(data)
            rst.index=rst.index.astype(str).map(lambda x:x[:7])
            self.benchmark_return['sp500']= rst[['Return']]
            return self.benchmark_return['sp500']
#    @timeit
    def package(self):
        a={'return' : self.all_stock_return(),
            'prices' : self.stock_price() ,
            'f3' : self.factor_3(),
            'f5' : self.factor_5(),
            'rm' : self.marketreturn()['Return'],
            'rf' : self.riskfree()['RF'],
            }
        return a
    
    def change_time(self,start,end,stock_list=[]):
        if type(start) == pd._libs.tslibs.timestamps.Timestamp:
            start = str(start)[:7]
        if type(end) == pd._libs.tslibs.timestamps.Timestamp:
            end = str(end)[:7]
        assert len(start)==7, 'start should be str or timestamp type'
        assert len(end)==7, 'end should be str or timestamp type'
        pkg = self.package()
        if stock_list==[]:
            stock_list=pkg['return'].keys()
        for i in pkg.keys():
            if i in ['return','prices']:
                data = pkg[i]
                pkg[i]={s:data[s][start:end] for s in stock_list}
            else:
                pkg[i]=pkg[i][start:end]
        return pkg
        
        
    def stock(self):
        # dict of all stock excess return Series
        self.all_stock_return()
        return self.stock_return 
    
    def riskfree(self):
        return self.benchmark_return['rf'] 
    
    def marketreturn(self):
        return self.benchmark_return['sp500'] 
    
    def stock_price(self):
        self.all_stock_return()
        return self.stock_close
    
    def marketprice(self):
        return self.benchmark_price['sp500'] 

def stock():
    # dict of all stock excess return Series
    a.all_stock_return()
    return a.stock_return 

def riskfree():
    return a.benchmark_return['rf'] 

def marketreturn():
    return a.benchmark_return['sp500'] 

def stock_price():
    a.all_stock_return()
    return a.stock_close

def marketprice():
    return a.benchmark_price['sp500'] 

def factor_3():
    return a.factor_3()

def factor_5():
    return a.factor_5()
a = GETTER()

if __name__ == '__main__':
#    b = GETTER()
    # stock return
#    s_return = b.all_stock_return()
    s_return =stock()
    # stock price
#    s_price = b.stock_price()
    s_price = stock_price()
    # 3 factors, 5 factors
#    f3 = b.factor_3()
#    f5 = b.factor_5()
    f3 = factor_3()
    f5 = factor_5()
    # risk free rate
#    rf = b.riskfree()
    rf = riskfree()
    # S&P500 return 
#    rm = b.marketreturn()
    rm = marketreturn()
    # S&P500 price
#    pm = b.marketprice()
    pm = marketprice()
    
    newpkg=a.change_time('2010-04','2011-04')
    
    













