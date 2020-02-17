# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:57:03 2019

@author: khorw
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

factor_number=[1,3,5]

sig_lv=0.05

#Identify MutliProblem
correlation=0.5

def reset_ramsey(res, degree=5):
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    # vander without constant and x, and drop constant
    norm_values = np.asarray(res.fittedvalues)
    norm_values = norm_values / np.sqrt((norm_values ** 2).mean())
    y_fitted_vander = np.vander(norm_values, order)[:, :-2]
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    exog /= np.sqrt((exog ** 2).mean(0))
    endog = res.model.endog / (res.model.endog ** 2).mean()
    res_aux = sm.OLS(endog, exog).fit()
    # r_matrix = np.eye(degree, exog.shape[1], k_vars)
    r_matrix = np.eye(degree - 1, exog.shape[1], k_vars)
    # df1 = degree - 1
    # df2 = exog.shape[0] - degree - res.df_model  (without constant)
    return res_aux.f_test(r_matrix) # , r_matrix, res_aux



def VIF(x,tol=5):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    vif["features"] = x.columns
    return any(vif["VIF Factor"]>tol)

def Criter(dic,y,x):
    AIC=[]
    AIC.append([np.mean([dic[i][j]['AIC']['rst'] for j in y.columns]) for i in factor_number])
    BIC=[]
    BIC.append([np.mean([dic[i][j]['BIC']['rst'] for j in y.columns]) for i in factor_number])
    Aver=[AIC[0][i]+BIC[0][i] for i in range(len(AIC[0]))]
    j=Aver.index(min(Aver))
    check=VIF(x.iloc[:,0:j+1])
    if check==True:
        factor_number.pop(j)
        Criter(dic,y,x)
    return factor_number[Aver.index(min(Aver))]


def model_filter(dic,y,x,factornumber=factor_number,R2_percentitle=40):
    n=Criter(dic,y,x)
    List=[dic[n][i]['Rsquared']['rst'] for i in y.columns]
    List.sort()
    MinR2=List[int(len(List)//(len(List)/R2_percentitle))]
    Model_Sur=[]
    for i in y.columns:
        num=0
        if dic[n][i]['Rsquared']['rst']>MinR2:
            num += 1
        if dic[n][i]['Ftest']['rst']=='Reject_H0':
            num += 1
        if dic[n][i]['Reset']['rst']=='Not_Reject_H0':
            num += 1
        if num > 2:
            Model_Sur.append(i)
    return Model_Sur ,n #, MinR2


def diagnostic(x,y,sig_lv=sig_lv,cor=correlation):
    rslt={}
    p_x=[ss.jarque_bera(x.iloc[:,i]) for i in range(len(x.columns))]
    n_test_x=['Reject_H0' if p_x[i][1]<sig_lv else 'Not_Reject_H0' for i in range(len(p_x))]
    p_y=[ss.jarque_bera(y.iloc[:,i]) for i in range(len(y.columns))]
    n_test_y=['Reject_H0' if p_y[i][1]<sig_lv else 'Not_Reject_H0' for i in range(len(p_y))]
    corre=x.corr()
    rslt['mutli']=[{x.columns[i]+' '+x.columns[j]:str(np.absolute(corre.iloc[i,j])>0.5)} for i in range(len(x.columns)-1) for j in range(i+1,len(x.columns))]
    rslt['JB_Test']=[{x.columns[i]:{'p':p_x[i][1],'rst':n_test_x[i]}} for i in range(len(x.columns))]+[{y.columns[i]:{'p':p_y[i][1],'rst':n_test_y[i]}} for i in range(len(y.columns))]
    return rslt


def run_reg(x,y,sig_lv=0.05,fator=factor_number,R2_percentitle=40):
    sig_lv=0.05
    result={}
    x=x.copy()
    x.insert(0,'constant',np.ones((len(x.index),)),True)
    for j in fator:
        result_tem={}
        for i in range(len(y.columns)):
            dep=x.copy().iloc[:,0:j+1]
            Reg=sm.OLS(y.iloc[:,i].values,dep.values).fit()
            #Reg.summary2()
            tp=Reg.pvalues
            t_test=['Reject_H0' if i<sig_lv else 'Not_Reject_H0' for i in tp]
            fp=Reg.f_pvalue
            f_test=['Reject_H0' if fp<sig_lv else 'Not_Reject_H0' ]
            coef=Reg.params
            r2=Reg.rsquared
            aic=Reg.aic
            bic=Reg.bic
            dur=sm.stats.stattools.durbin_watson(Reg.resid)
            dl=[1.664,1.653,1.643,1.633,1.623]
            du=[1.684,1.693,1.704,1.715,1.725]
            dur_test=['Reject_H0' if dl[(len(x.columns)-2)]>dur>(4-dl[(len(x.columns)-2)]) else ('Not_Reject_H0' if du[(len(x.columns)-2)]<dur<(4-du[(len(x.columns)-2)]) else 'Inclusive')]
            _,p_w,_,_white=sm.stats.diagnostic.het_white(Reg.resid,x)
            white=['Reject_H0' if p_w < sig_lv else 'Not_Reject_H0']
            reset=reset_ramsey(Reg, len(t_test)).pvalue
            reset_test=['Reject_H0' if reset<sig_lv else 'Not_Reject_H0']
            result_tem[y.columns[i]]={
                    'alpha':{'Coef':coef[0],'p':tp[1],'rst':t_test[0]},
                    'beta':[{'Coef':coef[i],'p':tp[i],'rst':t_test[i]} for i in range(1,len(coef))],
                    'DW':{'TestStatistic':dur,'rst':dur_test[0]},
                    'Ftest':{'rst':f_test[0]},
                    'Rsquared':{'rst':r2},
                    'AIC':{'rst':aic},
                    'BIC':{'rst':bic},
                    'White':{'rst':white[0]},
                    'Reset':{'rst':reset_test[0]}}
        result[j]=result_tem
    StockList=model_filter(result,y,x,R2_percentitle=R2_percentitle)
    return StockList


''
if __name__=='__main__':
    
x=IndependentVariable

y=DependentVariable


#should be generated by Zhu's py 

#x
fator_5=pd.read_csv(r'Data\5_Factors.csv', index_col=0, parse_dates=True)
fator_5.index=pd.to_datetime(fator_5.index,format='%Y%m')
fator_5=fator_5.loc['1999-04':'2016-12',fator_5.columns[:-1]]

#y
gene_excess_return=pd.read_csv(r'Data\raw.csv', index_col=0, parse_dates=True)
gene_excess_return=gene_excess_return.loc['1999-04':'2016-12',:]

dic,fic_all=run_reg(fator_5,gene_excess_return,R2_percentitle=30)
    
#Then to diagnostic
dianos=diagnostic(fator_5,gene_excess_return)
'''