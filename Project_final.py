from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#1.3 Plot time series of two indexes
DJIA = pd.read_csv('^DJI.csv')
DJIA.index = pd.to_datetime(DJIA.Date)
plt.plot(DJIA.Close)
plt.show()

SP500 = pd.read_csv('^GSPC.csv')
SP500.index = pd.to_datetime(SP500.Date)
plt.plot(SP500.Close)
plt.show()

#1.4.1 Daily log return on DJIA index
Rt_DJIA = pd.DataFrame(DJIA.Close)
Rt_DJIA['log_Rt'] = np.log(Rt_DJIA.Close)-np.log(Rt_DJIA.Close.shift(1))
print(Rt_DJIA)

#1.4.2 Daily log return on S&P500 index
Rt_SP500 = pd.DataFrame(SP500.Close)
Rt_SP500['log_Rt'] = np.log(Rt_SP500.Close)-np.log(Rt_SP500.Close.shift(1))
print(Rt_SP500)

#1.5 Plot the time series of log return on DJIA and SP500 index
DJIA['log_Rt'] = np.log(Rt_DJIA.Close)-np.log(Rt_DJIA.Close.shift(1))
print(DJIA)
SP500['log_Rt'] = np.log(Rt_SP500.Close)-np.log(Rt_SP500.Close.shift(1))
print(SP500)
plt.plot(DJIA.log_Rt)
plt.show()
plt.plot(SP500.log_Rt)
plt.show()

#1.6 Sample mean and unbaised smaple variance of log returns
Mean_DJIA = DJIA.log_Rt.mean()
Var_DJIA = DJIA.log_Rt.var()
Std_DJIA = DJIA.log_Rt.std()
print('The sample mean of log return on DJIA is {:.4}'.format(Mean_DJIA) +
      ' ,' + 'the unbaised sample variance is {:.4}.'.format(Var_DJIA))

Mean_SP500 = SP500.log_Rt.mean()
Var_SP500 = SP500.log_Rt.var()
Std_SP500 = SP500.log_Rt.std()
print('The sample mean of log return on S&P500 is {:.4}'.format(Mean_SP500) +
      ' ,' + 'the unbaised sample variance is {:.4}.'.format(Var_SP500))

#1.7 Annualized average and volatility of log return in percent, each year has 252 trading days
Ann_mean_DJIA = 252 * Mean_DJIA
Ann_mean_SP500 = 252 * Mean_SP500
Ann_std_DJIA = np.sqrt(252 * Std_DJIA**2)
Ann_std_SP500 = np.sqrt(252 * Std_SP500**2)
print('The annualized average of log return on DJIA in percent is {:.2%}'.format(Ann_mean_DJIA) + ' ,' + 'the annualized volatility in percent is {:.2%}.'.format(Ann_std_DJIA))
print('The annualized average of log return on S&P500 in percent is {:.2%}'.format(Ann_mean_SP500) + ' ,' + 'the annualized volatility in percent is {:.2%}.'.format(Ann_std_SP500))

#1.8 Sample skewness and sample kurtosis
def skewkurt(m):
    s = 0
    for i in range(1,len(m)):
        s = s + m[i]
    mu = s/(len(m)-1)
    
    s = 0
    for i in range(1,len(m)):
        s = s + (m[i] - mu)**2
    var = s/(len(m)-1)
    
    std = np.sqrt(var)
    s = 0
    for i in range(1,len(m)):
        s = s + (m[i] - mu)**3
    skew = s/((len(m)-1)*std**3)
    sample_skew = round(skew,4)
    
    s = 0
    for i in range(1,len(m)):
        s = s + (m[i] - mu)**4
    kurt = s/((len(m)-1)*std**4)
    sample_kurt = round(kurt,4)
    
    return ('the sample skewness is ' + str(sample_skew) + ' ,' + 'the sample kurtosis is ' + str(sample_kurt) + '.')

print('For DJIA index, ' + skewkurt(DJIA.log_Rt))
print('For S&P500 index, ' + skewkurt(SP500.log_Rt))

#1.9 Jarque-Bera test statistic
JBtest_DJIA = stats.jarque_bera(DJIA.log_Rt[1:])
JBtest_SP500 = stats.jarque_bera(SP500.log_Rt[1:])

def hypothesis(x):
    if x[1]>0.05:
        print('We cannot reject the null hypothesis H0: JB=0')
    else:
        print('We reject the null hypothesis H0: JB=0, ' + 'the Jarque-Bera test statistic is ' + str(round(x[0],4)))

hypothesis(JBtest_DJIA)
hypothesis(JBtest_SP500)

#2.1 Correlation between log returns of DJIA and S&P500 indexs
Corr_DJSP = np.corrcoef(DJIA.log_Rt[1:],SP500.log_Rt[1:])
print('The correlation between the log returns of DJIA and S&P500 indexes is ' + str(round(Corr_DJSP[0][1],4)) + '.')

#2.2 Examine whether 2 samples have equal mean at the alpha = 5% significance level
Mu1 = Mean_DJIA
Mu2 = Mean_SP500
Var1 = Var_DJIA
Var2 = Var_SP500
T1 = len(DJIA.log_Rt)-1
T2 = len(SP500.log_Rt)-1
T_stats = (Mu1-Mu2)/np.sqrt(Var1/T1+Var2/T2)
Dof = (Var1/T1+Var2/T2)**2/((Var1/T1)**2/(T1-1)+(Var2/T2)**2/(T2-1))
alpha = 0.05

if np.abs(T_stats) > stats.t.ppf(1-alpha/2,Dof):
    print('We reject the null hypothesis H0: mu_DJI=mu_SP500, and accept the alternative hypothesis Ha: mu_DJI≠mu_SP500')
elif T_stats > stats.t.ppf(1-alpha,Dof):
    print('We reject the null hypothesis H0: mu_DJI=mu_SP500, and accept the alternative hypothesis Ha: mu_DJI>mu_SP500')
elif T_stats < stats.t.ppf(alpha,Dof):
    print('We reject the null hypothesis H0: mu_DJI=mu_SP500, and accept the alternative hypothesis Ha: mu_DJI<mu_SP500')
else:
    print('We can not reject the null hypothesis H0: mu_DJI=mu_SP500')
    
#2.3 F-test for equality of 2 variances at the alpha=5% level of significance
F_stats = Var1/Var2
Dof1 = T1-1
Dof2 = T2-1
alpha = 0.05

if F_stats < stats.f.ppf(1-alpha/2,Dof1,Dof2) or F_stats > stats.f.ppf(alpha/2,Dof1,Dof2):
    print('We reject the null hypothesis H0: sigma_DJI=sigma_SP500, and accept the alternative hypothesis Ha: sigma_DJI≠sigma_SP500')
else:
    print('We can not reject the null hypothesis H0: sigma_DJI=sigma_SP500')

from scipy.stats import linregress as ols
class stat_SLR:
    def __init__(self,x,y):
        x=np.array(x)
        y=np.array(y)
        self.b, self.a, *other = ols(x,y)
        self.T = len(x)
        n=self.T
        # sample property
        self.x_mean = x.mean()
        self.y_mean = y.mean()
        self.x_sigma = x.std(ddof=1)
        self.y_sigma = y.std(ddof=1)
        self.y_hat = self.b*x+self.a
        self.u_hat = y-self.y_hat
        # prediction check
        self.RSS = ((self.u_hat)**2).sum()
        self.TSS = ((y-self.y_mean)**2).sum()
        self.ESS = self.TSS - self.RSS
        self.R_2 = 1-self.RSS/self.TSS
        self.R_2_adj = 1- self.RSS/(self.T-2)/(self.TSS/(self.T-1))
        # estimator check
        self.u_sigma_hat = np.sqrt(self.RSS/(self.T-2))
        self.b_sigma = self.u_sigma_hat/np.sqrt(x.std(ddof=0)**2*n)
        self.a_sigma = self.b_sigma*np.sqrt(x.std(ddof=0)**2+x.mean()**2)
        # ddof = T-2
        self.a_hat_tstat = self.a/self.a_sigma
        self.b_hat_tstat = self.b/self.b_sigma
        
    @staticmethod
    def DW_test(s):
        return ((s-np.append(np.nan,s[:-1]))**2)[1:].sum()/(s**2).sum()
        
        
t3 = stat_SLR(log_r.SP500,log_r.DJI)
slope, intercept = t3.b, t3.a
print('a={}, b={}'.format(intercept, slope))
sigma_u = t3.u_sigma_hat
print('sigma_u = {}'.format(sigma_u))
a_hat_t = t3.a_hat_tstat
b_hat_t = t3.b_hat_tstat
print('t stat for a = {}, for b = {}'.format(a_hat_t, b_hat_t))

get_critical = lambda q, dof: ss.t.ppf(q, df=dof)
dof = t3.T-2
q=1-0.05/2
critical = get_critical(q,dof)
print('Critical value for a and b is {}'.format(critical))

R_2 = t3.R_2
R_2_adj = t3.R_2_adj
print('R square = {}, adjusted R square = {}'.format(R_2, R_2_adj))

JB_u = JB_test(t3.u_hat)
print('JB test result is {}'.format(JB_u))

DW_u = stat_SLR.DW_test(t3.u_hat)
print('DW test result is {}'.format(DW_u))
