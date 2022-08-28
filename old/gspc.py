import numpy as np
import yfinance as yf
import pandas as pd

NUM_TRADING_DAYS = 252

# download S&P500 prices
def download_data(stocks_df, start_year, end_year):
    start = str(start_year) + '-01-01'
    end = str(end_year) + '-12-31'
    
    # Get top 10 stocks in year
    stocks = stocks_df['stock'].to_numpy()

    stock_data = {}
    for stock in stocks: 
        ticker = yf.Ticker(stock)
        stock_data[stock] = (ticker.history(start=start, end=end))['Close']

    return pd.DataFrame(stock_data)


def calculate_log_daily_returns(stocks_data):
    log_daily_returns = np.log(stocks_data/stocks_data.shift(1))
    return log_daily_returns[1:]

def gspc_statistics(log_daily_returns, risk_free = 0):
    mean = np.sum(log_daily_returns.mean()) * NUM_TRADING_DAYS
    # can you calculate index volatility like this?
    risk = np.sqrt(np.dot(np.array([1]).T, np.dot(log_daily_returns.cov() * NUM_TRADING_DAYS, np.array([1]))))
    return np.array([mean, risk, (mean-risk_free)/risk])

years = [2000, 2001, 2002, 2003]

# S&P500 data for following year
# gspc_df = pd.DataFrame([[years[0], '^GSPC']], columns=['year', 'stock'])
# gspc_data_next_year = download_data(gspc_df, years[0]+1, years[0]+1)
# gspc_log_daily_returns_next_year = calculate_log_daily_returns(gspc_data_next_year)
# gspc_mean = np.sum(gspc_log_daily_returns_next_year.mean()) * NUM_TRADING_DAYS
# print(gspc_statistics(gspc_log_daily_returns_next_year))
# print(gspc_mean)

stats_table = pd.DataFrame(columns=['year', 'stock', 'mean', 'risks', 'sharpe'])
stats_table.append({'year': 2000, 'stock': 1, 'means': 2, 'sharpe': 1 })
print(stats_table)