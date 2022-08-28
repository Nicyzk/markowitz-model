import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

# stocks we are going to handle
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

# historical data - define START and END dates
start_date = '2012-01-01'
end_date = '2017-01-01'


def download_data():
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = (ticker.history(start=start_date, end=end_date))['Close']

    return pd.DataFrame(stock_data)


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_return_1 = np.sum(returns.mean() * NUM_TRADING_DAYS * weights)
    print(returns.mean())
    print(portfolio_return)
    print(portfolio_return_1)

    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    print(returns.cov())

    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)


def show_portfolio(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    # if list is separated by commas, if np array no commas when printed?
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def statistics(weights, returns):
    # print(weights)
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    # print('one portfolio', portfolio_return)
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


# scipy optimize can find the minimum of a given function
# the maximum of f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


# What are the constraints? The sum of the weights = 1!
# sum w - 1 = 0     f(x) = 0 is the function we want to minimize    Meaning???
def optimize_portfolio(weights, returns):
    # Ensure the sum of weights = 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # the weights must be between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds,
                          constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe Ratio: ", statistics(optimum['x'].round(3), returns))


def show_optimal_portfolio(optimum, rets, returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(optimum['x'], rets)[1], statistics(optimum['x'], rets)[0], 'g*', markersize=20)
    plt.show()


if __name__ == '__main__':
    dataset = download_data()
    # show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)  # Average log returns is equal to geometric mean of total % change

    weights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolio(means, risks)

    optimum = optimize_portfolio(weights, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
