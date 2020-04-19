import HeliosEquity
import matplotlib.pyplot as plt

tickers = ['AAPL', 'MSFT']

# Pull Equity data and generate returns
Equity_Data = HeliosEquity.Equity_Data(tickers) # Initialize Equity Data

prices = Equity_Data.prices() # Pull equity prices from Yahoo DataReader
returns = Equity_Data.returns(prices) # Daily Returns - non log
future_returns = Equity_Data.future_returns(prices) # 5 Day Future Returns
monthly_returns = Equity_Data.monthly_returns(prices) # Monthly Returns


# Use prices to create equity technicals
Technicals = HeliosEquity.Technicals(df = prices, window_length =  20)

for tick in tickers:
    df = prices['Adj Close'][tick].tail(100)
    bband = Technicals.bollinger_band(df, window_length = 20, sd = 2)
    
    fig, axs = plt.subplots(2, 2, figsize = (12,6))
    fig.autofmt_xdate()
    
    # Exponential Moving Average
    axs[0, 0].plot(df, label = 'Closing Price')
    axs[0, 0].plot(Technicals.ewm(df, window_length = 20), label = '20D EWM')
    axs[0, 0].plot(Technicals.ewm(df, window_length = 100), label = '100D EWM')
    axs[0, 0].set_title(str(tick) + ' EWM')
    axs[0, 0].legend()
    
    # Volaility
    axs[0, 1].hist(monthly_returns[tick])
  
    
    # RSI - > 70 overbought < 30 oversold
    axs[1, 0].plot(Technicals.rsi(df, window_length = 14).tail(14), color= 'orange')
    axs[1, 0].axhline(70, linestyle = '--')
    axs[1, 0].axhline(80, linestyle = '--', alpha = .5)
    axs[1, 0].axhline(30, linestyle = '--')
    axs[1, 0].axhline(20, linestyle = '--', alpha = .5)
    axs[1, 0].set_title( str(tick) + ' RSI')
    
    # Bollinger Band 
    axs[1, 1].fill_between(bband.index.get_level_values(0), bband['Upper'], bband['Lower'], color = 'grey')
    axs[1, 1].plot(bband.index.get_level_values(0), bband['Mean'], color='black', lw=2, label = 'Mean')
    axs[1, 1].plot(df.tail(80), label = 'Price')
    axs[1, 1].set_title(str(tick) + ' 20D Bollinger Band')
    axs[1, 1].legend()
    

# Create features and test/training set
train_features, train_targets, test_features, test_targets = hle.features(prices)

# Random Forest Model
rfr_best_idx, rfr_best_grid = hle.random_forest()
