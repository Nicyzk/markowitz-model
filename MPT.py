from threading import stack_size
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000


def import_stocks():
    return pd.read_csv('./top10stocks_5.csv')


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


def generate_portfolios(stocks_data):
    log_daily_returns = calculate_log_daily_returns(stocks_data) 
    stocks = list(stocks_data.columns)

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(log_daily_returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(log_daily_returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def show_portfolios(means, risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means / risks, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def portfolio_statistics(weights, log_daily_returns, risk_free = 0):
    mean = np.sum(log_daily_returns.mean() * weights) * NUM_TRADING_DAYS
    risk = np.sqrt(np.dot(weights.T, np.dot(log_daily_returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([mean, risk, (mean-risk_free)/risk])


# Calculates negative sharpe ratio for a single portfolio
# Scipy optimize can find the minimum of a given function. The maximum of f(x) is the minimum of -f(x)
def negative_sharpe(weights, log_daily_returns, risk_free = 0):
    sharpe = portfolio_statistics(weights, log_daily_returns, risk_free)[2]
    return -sharpe


def optimize_SR(weights, log_daily_returns, risk_free = 0):
    args = (log_daily_returns, risk_free)
    # Ensure the sum of weights = 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # the weight of each stock must be between 0 and 1
    num_stocks = len(weights[0])
    bounds = tuple((0, 1) for _ in range(num_stocks))

    #x0 is our first guess
    return optimization.minimize(fun=negative_sharpe, x0=weights[0], args=args, method='SLSQP', bounds=bounds, constraints=constraints)


def show_optimal_SR_portfolio(optimum, log_daily_returns, means, risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means / risks, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(portfolio_statistics(optimum['x'], log_daily_returns)[1], portfolio_statistics(optimum['x'], log_daily_returns)[0], 'g*', markersize=20)
    plt.show()


def print_optimal_portfolio(optimum, log_daily_returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe Ratio: ", portfolio_statistics(optimum['x'].round(3), log_daily_returns))


def gspc_statistics(log_daily_returns, risk_free = 0):
    mean = np.sum(log_daily_returns.mean()) * NUM_TRADING_DAYS
    # can you calculate index volatility like this?
    risk = np.sqrt(np.dot(np.array([1]).T, np.dot(log_daily_returns.cov() * NUM_TRADING_DAYS, np.array([1]))))
    return np.array([mean, risk, (mean-risk_free)/risk])


if __name__ == '__main__':
    stocks_df = import_stocks()
    years = stocks_df['year'].unique()

    # Table to compare stats between Max_SR portfolio and S&P500
    stats_table = pd.DataFrame(columns=['year', 'stock', 'means', 'risks', 'sharpe'])

    for i in range(len(years)):  

        # daily historical price data for past 5 years, including current year (since top 10 stocks change yearly)
        stocks_data = download_data(stocks_df[stocks_df['year'] == years[i]], years[i]-4, years[i])
        log_daily_returns = calculate_log_daily_returns(stocks_data)
        weights, means, risks = generate_portfolios(stocks_data)

        # show efficient frontier
        # show_portfolios(means, risks)

        # find portfolio that optimizes sharpe ratio
        result = optimize_SR(weights, log_daily_returns)
        # show_optimal_SR_portfolio(result, log_daily_returns, means, risks)
        print_optimal_portfolio(result, log_daily_returns)
        
        # daily historical price data for next year 
        # Assumption: stocks are top 10 largest market cap at end of years[i], so we use next year's price to prevent lookahead bias
        stocks_data_next_year = download_data(stocks_df[stocks_df['year'] == years[i]], years[i]+1, years[i]+1) 
        log_daily_returns_next_year = calculate_log_daily_returns(stocks_data_next_year)
        stats = portfolio_statistics(result['x'], log_daily_returns_next_year)

        # S&P500 data for following year
        gspc_df = pd.DataFrame([[years[i], '^GSPC']], columns=['year', 'stock'])
        gspc_data_next_year = download_data(gspc_df, years[i]+1, years[i]+1)
        gspc_log_daily_returns_next_year = calculate_log_daily_returns(gspc_data_next_year)
        gspc_stats = gspc_statistics(gspc_log_daily_returns_next_year)

        # insert rows into stats table
        portfolio_row = pd.DataFrame( {'year':[years[i]+1], 'stock': ['Max_SR'], 'means': [stats[0]], 'risks': [stats[1]], 'sharpe': [stats[2]] })
        stats_table = pd.concat([stats_table, portfolio_row], ignore_index=True)
        gspc_row = pd.DataFrame( {'year':[years[i]+1], 'stock': ['^GSPC'], 'means': [gspc_stats[0]], 'risks': [gspc_stats[1]], 'sharpe': [gspc_stats[2]] })
        stats_table = pd.concat([stats_table, gspc_row], ignore_index=True)
        
    print(stats_table)
    stats_table.to_csv('./stats.csv', mode='a', index=False, header=False)


    # Compare total log returns of Max_SR and GSPC over entire strategy time period 

   



    
