import helios_equity
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm

plt.style.use('seaborn-deep')

# Define Stock Tickers
tickers = ['AAPL', 'MSFT']
weights = [.5, .5]
start_date = '2015-01-01'

# Generate stock prices and returns
equities = Helios(tickers, start_date)
prices, returns, monthly_returns = equities.returns()
correlation_matrix = returns.corr()

# Conduct technical analysis on stock returns
technicals = Technicals(returns, window = 10)
rsi = technicals.rsi(prices)
moving_avg = technicals.moving_average(prices)
#log_returns = technicals.log_return()
bollinger = technicals.bollinger_band(prices)
exponential_average = technicals.exponential_average(prices)
beta = technicals.beta(monthly_returns, start_date)


# Technical Plots

for tick in tickers:
    # Moving Averages
    plt.plot(moving_avg[tick]. shift(20), label = '20 day SMA')
    plt.plot(exponential_average[tick].shift(20), label = '30D')
    plt.title(str(tick) + ' Moving Averages')
    plt.legend()
    plt.ylabel('Price')
    plt.show()
    
    # Returns Histogram
    plt.hist(returns[tick], bins = 50)
    plt.title(tick)
    plt.xlabel('Adjusted Close 1 Day % Chg')
    plt.show()
    
    # RSI
    rsi[tick].tail(20).plot(linewidth = 2)
    plt.axhline(y=30, color= 'r')
    plt.axhline(y=70, color = 'g')
    plt.title(label = tick, loc = 'center')
    plt.show()
    
    # Bollinger Bands
    bollinger[tick].tail(90).plot(linewidth = 2)
    plt.fill_between(bollinger[tick].index.get_level_values(0),bollinger[tick]['Upper'], bollinger[tick]['Lower'], color='grey')
    plt.plot(bollinger[tick].index.get_level_values(0),bollinger[tick]['Mean'],color = 'green', lw=2)
    plt.title(str(tick) + ' Bollinger Band')
    plt.ylabel('Price')
    plt.axis('tight')
    plt.show()
    
    # Mean Reversion
    plt.scatter(returns[tick].shift(5).corr(returns[tick]).shift(-5))
    #plt.scatter(returns[tick].shift(5), returns[tick].shift(-5))
    plt.title(str(tick) + ' Mean Reversion')
    plt.show()
    
    # Momentum
    plt.plot(prices[tick] - prices[tick].shift(10))
    plt.title(str(tick) + ' 10 Day Momentum')
    plt.show()
    
    # Market Correlation
    returns[[tick, 'SPY']].plot(x = 'SPY', y=tick, kind= 'scatter')

    ols = sm.OLS(returns[tick].values, returns['SPY'].values).fit()
    plt.plot(returns['SPY'], ols.fittedvalues,'r')
    
    # Rolling Correlation
    returns['SPY'].rolling(252).corr(other = returns['AAPL']).plot()
  

# Pull company financials and perform financial analysis
equities = Financials(tickers)
balance_sheets = equities.balance_sheet()
income_statements = equities.income_statement()
cash_flow_statements = equities.cash_flow()
