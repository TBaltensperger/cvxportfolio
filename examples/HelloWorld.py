# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import quandl
quandl.ApiConfig.api_key = "iMtoz65kcvRxswxXZTfg"


import cvxportfolio as cp

# %% [markdown]
# Download the problem data from Quandl. We select four liquid stocks, and the risk-free rate.

# %%
tickers = ['AMZN', 'GOOGL', 'TSLA', 'NKE']
start_date='2012-01-01'
end_date='2016-12-31'
returns = pd.DataFrame(dict([(ticker, quandl.get('WIKI/'+ticker, 
                                    start_date=start_date, 
                                    end_date=end_date)['Adj. Close'].pct_change())
                for ticker in tickers]))
returns[["USDOLLAR"]]=quandl.get('FRED/DTB3', start_date=start_date, end_date=end_date)/(250*100)
returns = returns.fillna(method='ffill').iloc[1:]

returns.tail()

# %% [markdown]
# We compute rolling estimates of the first and second moments of the returns using a window of 250 days. We shift them by one unit (so at every day we present the optimizer with only past data).

# %%
r_hat = returns.rolling(window=250, min_periods=250).mean().shift(1).dropna()
Sigma_hat = returns.rolling(window=250, min_periods=250, closed='neither').cov().dropna()

r_hat.tail()

# %% [markdown]
# Here we define the transaction cost and holding cost model (sections 2.3 and 2.4 [of the paper](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)). The data can be expressed 
# as 
# - a scalar (like we're doing here), the same value for all assets and all time periods;
# - a Pandas Series indexed by the asset names, for asset-specific values; 
# - a Pandas DataFrame indexed by timestamps with asset names as columns, for values that vary by asset and in time.

# %%
tcost_model=cp.TcostModel(half_spread=10E-4)
hcost_model=cp.HcostModel(borrow_costs=1E-4)

# %% [markdown]
# We define the single period optimization policy (section 4 [of the paper](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)). 

# %%
risk_model = cp.FullSigma(Sigma_hat)
gamma_risk, gamma_trade, gamma_hold = 5., 1., 1.
leverage_limit = cp.LeverageLimit(3)

spo_policy = cp.SinglePeriodOpt(return_forecast=r_hat, 
                                costs=[gamma_risk*risk_model, gamma_trade*tcost_model, gamma_hold*hcost_model],
                                constraints=[leverage_limit])

# %% [markdown]
# We run a backtest, which returns a result object. By calling its summary method we get some basic statistics.

# %%
market_sim=cp.MarketSimulator(returns, [tcost_model, hcost_model], cash_key='USDOLLAR') 
init_portfolio = pd.Series(index=returns.columns, data=250000.)
init_portfolio.USDOLLAR = 0
results = market_sim.run_multiple_backtest(init_portfolio,
                               start_time='2013-01-03',  end_time='2016-12-31',  
                               policies=[spo_policy, cp.Hold()])
results[0].summary()

# %% [markdown]
# The total value of the portfolio in time.

# %%
results[0].v.plot(figsize=(12,5))
results[1].v.plot(figsize=(12,5))

# %% [markdown]
# The weights vector of the portfolio in time.

# %%
results[0].w.plot(figsize=(12,6))


# %%



