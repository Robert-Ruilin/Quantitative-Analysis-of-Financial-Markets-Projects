# Quantitative-Analysis-of-Financial-Markets-Projects
1. Mini project

2. Final project - Application: Stock Picking
Given daily prices of 100 stocks from April 1999 to December 2018.
We can get: 
1. Jensen’s Alpha for each stock
2. Beta for each stock
3. Monthly simple returns for each stock

We can construct our portfolio by 1st strategy, which is:
- Long top 10 stocks which earn highest Jensen’s alpha,
- Short bottom 10 stocks which earn lowest Jensen’s alpha.

The algorithm of constructing 2nd portfolio is:
---Model filtering---
Step 1: Selecting model with different factors. The purpose is to test which model has the most significant explanatory power on common cause variation in stock returns.
Optional set = {CAPM, Fama French three factor model, Fama French five factor model}
Criterion of model selection:
1. Akaike information criterion (AIC), the smaller, the better.
2. Schwarz information criterion (SIC), the smaller, the better.

Step 2: Multicollinearity Testing
Criterion of multicollinearity: Variance Inflation Factor (VIF)
Correlation matrix for all factors.
Multicollinearity is high if VIF is greater than 10, a cutoff of 5 is commonly used.

Step 3: F-test & RESET
Purpose of F testing is to test whether themodel has been fitted to our data set.
According to the test result, 90 models have sufficient evidence to reject F test’s null hypothesis.
Purpose of RESET is to test whether non linear combinations of the fitted values help explain the response variable.
According to the test result, 89 models do not have sufficient evidence to reject Reset test’s null hypothesis.

Step 4: R-squared & JB Test
According to the value of AIC and SIC, we select Fama French three factor model.
According to the value of R squared, we set a threshold of 20 percentile of the total value as stock picking criterion, accepting the stocks which is above threshold.
The number of stocks which are picked from model filtering is 51.

---Stock filtering algorithm---
Step 1: Filtering by factors
1. alpha
Jensen’s alpha seems to be significantly none zero, but in three factor and five factor alpha, we can clearly see that alpha can be explained by holding exposure on stylized betas.
Five factor alpha appears to be more normalized than the other 2.
2. Bull Beta and Bear Beta
In the long run, we want to pick stocks that have a positive beta as market goes up, while get a low or negative beta as market goes down.
3. Treynor Ratio
In the long run, survived stocks will all take systemic risk It makes sense to accept a threshold of positive return on systemic risk.

Step 2: Portfolio Construction
Given 19 stocks which are filtered by factors we mentioned in step 1. We use the following strategy to build our portfolio.
1. Denote remaining 19 stocks as our current stock pool.
2. Let 𝑛=0, drop one of the stocks in stock pool and set the remaining as a portfolio, then we got 𝐶(19−𝑛,1)portfolios.
3. Calculate the Sharpe ratio of each portfolio.
4. Choose the portfolio with the highest Sharpe ratio to be the new stock pool, then the number of stocks in stock pool shrinks into 19−𝑛
5. Let 𝑛=𝑛+1, redo step 1 to step
6. Stop filtering when
1) only 10 stocks left in the stock pool, or
2) the Sharpe ratio does not increase after filtering.

Long-short strategy
1. Bear Bull Beta Hedging Strategy
- Long position criterion:
  1) Bull Beta - Bear Beta > 0.3
  2) Treynor ratio > 0
  3) Five factor’s Alpha > 0
Then, picking top 10 stocks with highest Bull Beta - Bear Beta.
- Short position:
  1) S&P 500 index
In a dynamic hedging strategy, where we pick stocks according only to bull bear betas factor. The return is actually quiet satisfactory.
Pick stocks base on last 12 month rank and invest for a 3 month fixed period.

2. Beta Hedging Strategy
Step 1 - Find highly correlated stock pairs.
Step 2 - Use pricing models to find alpha of stocks in stock pairs.
Step 3 - Construct a portfolio by holding a long position of higher alpha stocks and short position of lower alpha stocks in each stock pairs.

Conclusion
1. Some short term factors can not be justified in the long term, because stocks carrying these 'factors' are shifting in the long term. A change in trading strategy (such as switching hands) can generate profit.
2. In our case, the portfolio under our strategy can outperform buy the market and hold strategy.
3. According to the value of SIC and T test result of CMA and RMW, Fama-French three-factor is more relevant to stock excess return than five-factor model.
4. We can try more variable as most of the models are not well explained in Fama-French three or five factors.
