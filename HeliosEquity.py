import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import probplot

tickers = ['AAPL', 'MSFT']

class Helios_Equity:
    def __init__(self, tickers = []):
        self.__tickers = tickers
        
    def prices(self):
        prices = pdr.data.DataReader(tickers, 'yahoo', '2010-01-01')
        return prices

    def returns(self,df):
        returns = df['Adj Close'].pct_change().dropna()
        return returns
    
    def future_returns(self, df):
        future_returns = df['Adj Close'].shift(-5).pct_change(5).dropna()
         #----- Quantile Plot to show if returns normally distributed -----#
        for tick in tickers:
            
            figure = plt.figure()
            ax = figure.add_subplot(111)
            stats.probplot(returns[tick], dist='norm', plot=ax)
            plt.title(str(tick) + 'Probability Plot')
            plt.show()
                     
        return future_returns
    
    def monthly_returns(self, df):
        monthly_returns = df.resample('BMS').first()
        return monthly_returns
    
    # Technical Analysis Calculations
    
    def rsi(self,df, window_length):
        delta = df.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0 ] = 0
        rsi =  100-100/(1+(+up.rolling(window_length).mean() / down.abs().rolling(window_length).mean()))
        return(rsi)

    def ma(self,df, window_length):
        ma = df.rolling(window = window_length).mean()
        return(ma)

    def log_return(self,df):
        log = np.log(df).diff().cumsum()
        return(log)

    def total_return(self,df): # Not Working
        total = 100 * (np.exp(log_return(df))) - 1
        return(total)

    def bollinger_band(self,df, window_length, sd): # Not Working
        
        mean = df.rolling(window_length).mean()
        std = df.rolling(window_length).std()
        upperband = mean + (std * sd)
        lowerband = mean - (std * sd)
        bband= pd.DataFrame({'Mean':mean, 'Upper':upperband,'Lower':lowerband})
        return(bband)

    def ewm(self,df, window_length):
        ewm = df.ewm(span = window_length).mean()
        return(ewm)


    def plots(self, df):
   
        corr_5d = {name: name for name in tickers}
        bband = {name: name for name in tickers}
          
        for tick in tickers:
            bband[tick] = pd.DataFrame()
            
            # Historgram
            plt.hist(df['Adj Close'][tick].pct_change(), bins = 50)
            plt.title(tick)
            plt.xlabel('Adjusted Close 1 Day % Chg')
            plt.show()
            
            # 5 Day Correlation
            corr_5d[tick] = df['Adj Close'][tick].pct_change(10).corr(df['Adj Close'][tick].shift(-5).pct_change(5))
    
            #------ Scatter Plots ------#
            plt.scatter(df['Adj Close'][tick].pct_change(5),  df['Adj Close'][tick].shift(-5).pct_change(5))
            plt.title(str(tick) + ' Mean Reversion')
            plt.show()
            
            # Simple Moving Average 
            plt.subplots(figsize =(16,9))

            plt.plot(df['Adj Close'][tick], label = 'Closing Price')
            
            plt.plot(Helios_Equity.ma(self,  df = df['Adj Close'][tick], window_length = 20), label = '20 day SMA')
            plt.plot(Helios_Equity.ma(self, df['Adj Close'][tick],window_length =  100), label = '100 day SMA')
            plt.title(str(tick) + ' Simple Moving Averages')
            plt.legend()
            plt.ylabel('Adjusted Closing Price')
            plt.show()
            
              #--------- Exponential Moving Averages -------#
            plt.plot(Helios_Equity.ewm( self, df = df['Adj Close'][tick],window_length = 5), label = '5D')
            plt.plot(Helios_Equity.ewm(self,df['Adj Close'][tick],window_length = 30), label =' 30D')
            plt.title(str(tick) + ' Exponential Moving Average')
            plt.legend()
            plt.show()
            
            #----- Momentum ------#
            plt.plot(df['Adj Close'][tick] - df['Adj Close'][tick].shift(-10))
            plt.title(str(tick) + '10 Day Momentum')
            plt.show()
            
               #------ RSI -------#
            plt.plot( Helios_Equity.rsi(self,df['Adj Close'][tick], window_length = 14).tail(14) )
            plt.title(str(tick) + ' RSI')
            plt.show()
            
            # Bollinger Band Plots
            bband[tick] = Helios_Equity.bollinger_band(self,df['Adj Close'][tick],window_length = 20, sd = 2)
            # Create Bollinger Band Plot
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)
            ax.fill_between(bband[tick].index.get_level_values(0),bband[tick]['Upper'], bband[tick]['Lower'], color='grey')
            ax.plot(bband[tick].index.get_level_values(0),bband[tick]['Mean'], color='blue', lw=2)
            ax.set_title('20 Day Bollinger Band For '+str(tick))
            ax.set_ylabel('Price')
            plt.show()
            
            
     def features(self, df)         
            
            
            


