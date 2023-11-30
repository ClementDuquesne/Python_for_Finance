import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries

api_key = '1BIM0PWI4YFFNKZX'
symbol = input('Enter stock symbol (e.g., AAPL): ')

def get_stock_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_quote_endpoint(symbol=symbol)
    return data

S0 = float(get_stock_data(symbol, api_key)['05. price'][0])
K_call = float(input('Strike long call : '))
K_put = float(input('Strike long put : '))
T = float(input('Time Horizon : '))
r = float(input('Risk-Free rate : '))
sig = float(input('Volatility : '))

print('Black and Scholes models calculator')

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return round(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return round(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 2)

sT = np.linspace(0.7 * S0, 1.3 * S0, 100)

def call_payoff(sT, strike_price, premium):
    return pd.DataFrame({'St': sT,
                         'payoff_call': np.where(sT > strike_price, sT - strike_price, 0) - premium})

Call = BS_CALL(S0, K_call, T, r, sig)
payoff_long_call = call_payoff(sT, K_call, Call)

fig = px.line(payoff_long_call, x="St", y="payoff_call", title='Payoff Long Call')
fig.show()

print(f'The Call price calculated using B-S Model is : {Call}')

def put_payoff(sT, strike_price, premium):
    return pd.DataFrame({'St': sT,
                         'payoff_put': np.where(sT < strike_price, strike_price - sT, 0) - premium})

Put = BS_PUT(S0, K_put, T, r, sig)
payoff_long_put = put_payoff(sT, K_put, Put)

fig2 = px.line(payoff_long_put, x="St", y="payoff_put", title='Payoff Long Put')
fig2.show()

print(f'The Put price calculated using B-S Model is : {Put}')
