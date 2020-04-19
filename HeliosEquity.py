import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import probplot

<<<<<<< HEAD

class Helios_Equity:
    def __init__(self, tickers = []):
        self.tickers = tickers
        
class Equity_Data(Helios_Equity):
    
    def __init__(self, tickers = []):
        self.tickers = tickers
           
    def prices(self):
        prices = pdr.data.DataReader(self.tickers, 'yahoo', '2010-01-01')
=======
tickers = ['AAPL', 'MSFT']

class Helios_Equity:
    def __init__(self, tickers = []):
        self.__tickers = tickers
        
    def prices(self):
        prices = pdr.data.DataReader(tickers, 'yahoo', '2010-01-01')
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
        return prices

    def returns(self,df):
        returns = df['Adj Close'].pct_change().dropna()
        return returns
    
    def future_returns(self, df):
        future_returns = df['Adj Close'].shift(-5).pct_change(5).dropna()
<<<<<<< HEAD
        return future_returns
    
    def monthly_returns(self, df):
        monthly_returns = df['Adj Close'].resample('BMS').first()
        monthly_returns = monthly_returns.pct_change().dropna()
        return monthly_returns

# Technical Indicators
class Technicals(Helios_Equity):
    
    def __init__(self, window_length, df):
        self.window_length = window_length
        self.df = df
        
    def rsi(self, df, window_length):
        
=======
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
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
        delta = df.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0 ] = 0
<<<<<<< HEAD
        rsi =  100-100/(1 + ( +up.rolling(window_length).mean() / down.abs().rolling(window_length).mean() ) )
=======
        rsi =  100-100/(1+(+up.rolling(window_length).mean() / down.abs().rolling(window_length).mean()))
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
        return(rsi)

    def ma(self,df, window_length):
        ma = df.rolling(window = window_length).mean()
        return(ma)

<<<<<<< HEAD
    def log_return(self, df):
        log = np.log(df).diff().cumsum()
        return(log)

    def total_return(self, df): # Not Working
        total = 100 * (np.exp(Technicals.log_return( df))) - 1
        return(total)

    def bollinger_band(self, df, window_length, sd): 
=======
    def log_return(self,df):
        log = np.log(df).diff().cumsum()
        return(log)

    def total_return(self,df): # Not Working
        total = 100 * (np.exp(log_return(df))) - 1
        return(total)

    def bollinger_band(self,df, window_length, sd): # Not Working
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
        
        mean = df.rolling(window_length).mean()
        std = df.rolling(window_length).std()
        upperband = mean + (std * sd)
        lowerband = mean - (std * sd)
<<<<<<< HEAD
        bband = pd.DataFrame({'Mean':mean, 'Upper':upperband,'Lower':lowerband})
        #bband = bband.dropna()
=======
        bband= pd.DataFrame({'Mean':mean, 'Upper':upperband,'Lower':lowerband})
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
        return(bband)

    def ewm(self,df, window_length):
        ewm = df.ewm(span = window_length).mean()
        return(ewm)
<<<<<<< HEAD
   


            
            
    def features(self, df):
        features = {name: name for name in tickers}
        train_size = {name: name for name in tickers}
        train_features = {name: name for name in tickers}   
        train_targets = {name: name for name in tickers}
        test_features = {name: name for name in tickers}
        test_targets = {name: name for name in tickers}
        scaled_train_features = {name: name for name in tickers}
        scaled_test_features = {name: name for name in tickers}
        analysis = {name: name for name in tickers}
        
        for tick in tickers:
            features[tick] = pd.DataFrame()
            analysis[tick] = pd.DataFrame()
        
            for n in [14,30,50,200]:
                analysis[tick]['ma' + str(n)] = Helios_Equity.ma(self, df['Adj Close'][tick],n)
                analysis[tick]['rsi' + str(n)] = Helios_Equity.rsi(self, df['Adj Close'][tick],n)
                
            features[tick] = pd.merge(pd.DataFrame(returns[tick]), analysis[tick],how = 'inner', right_index=True, left_index = True )
            features[tick].rename(columns={features[tick].columns[0]: "10D Return" }, inplace = True)
            features[tick]['Volume 1D Pct'] = prices['Volume'][tick].pct_change()
            features[tick]['Volume 1D Pct MA'] = Helios_Equity.ma(self, prices['Volume'][tick].pct_change(),5)
        
            features[tick] = features[tick].dropna()
        
            # Format future returns to set length equal to features
            future_returns = future_returns.loc[features[tick].index.to_series()]
        
            # Subset test and training sets
            train_size[tick] = int(.85 * features[tick].shape[0])
            # Train Set
            train_features[tick] = pd.DataFrame(features[tick][ :train_size[tick]])
            scaled_train_features[tick] = pd.DataFrame(scale(train_features[tick])) # Scaled features to normalize ranges. Standardizing data.
            train_targets[tick] = pd.DataFrame(future_returns[tick][ :train_size[tick]])
           
            # Test Set
            test_features[tick] = pd.DataFrame(features[tick][train_size[tick]: ])
            scaled_test_features[tick] = pd.DataFrame(scale(test_features[tick]))
            test_targets[tick] = pd.DataFrame(future_returns[tick][train_size[tick]: ])
            
            return train_features, train_targets, test_features, test_targets
    
    def random_forest(self):
        rfr_test = {name: name for name in tickers}
        rfr_train = {name: name for name in tickers}
        rfr_test_scores = {name: name for name in tickers}
        rfr_best_idx = {name: name for name in tickers}
        rfr_feature_importance = {name: name for name in tickers}
        rfr_best_grid = {name: name for name in tickers}
        rfr_predictions = {name: name for name in tickers}
        grid = {'n_estimators': [50,200,400], 'max_depth': [3,5,10,20,50], 'max_features': [4,8], 'random_state': [42]}
       
        for tick in tickers:
            rfr_test_scores[tick] = []
            rfr_best_grid[tick] ={}
    # Loop through the parameter grid, set the hyperparameters, and save the scores
            for g in ParameterGrid(grid):
                rfr = RandomForestRegressor()
                rfr.set_params(**g)
                rfr.fit(train_features[tick], train_targets[tick])
                rfr_test_scores[tick].append(rfr.score(test_features[tick], test_targets[tick]))
    
        # Find best hyperparameters from the test score and store in dictionary
        rfr_best_idx[tick] = np.argmax(rfr_test_scores[tick])
        rfr_best_grid[tick] = dict(ParameterGrid(grid)[rfr_best_idx[tick]])
        print(rfr_test_scores[tick][rfr_best_idx[tick]], ParameterGrid(grid)[rfr_best_idx[tick]])
    
     
    # Train and test RF model using optimized parameters
        for tick in tickers:
            rfr = RandomForestRegressor(**rfr_best_grid[tick]).fit(train_features[tick], train_targets[tick])
            rfr_train[tick] = rfr.predict(train_features[tick])
            rfr_test[tick] = rfr.predict(test_features[tick])
            
            plt.scatter(train_targets[tick],rfr_train[tick], label = str(tick) + ' train ')
            plt.scatter(test_targets[tick], rfr_test[tick], label = str(tick) + ' test')
            plt.legend()
            plt.show()
        
        # Determine most important features
            rfr_feature_importance[tick] = rfr.feature_importances_
            plt.bar(range(len(rfr_feature_importance[tick])), np.argsort(rfr_feature_importance[tick])[::-1], tick_label = np.array(features[tick].columns)[np.argsort(rfr_feature_importance[tick])[::-1]])
            plt.xticks(rotation = 90)
            plt.title(str(tick) + ' RFR Feature Importance')
            plt.show()
       
        return rfr_best_idx, rfr_best_grid
            
    def gradient_boost(self):
        
           
          gbr_test_scores = {name: name for name in tickers}
          gbr_best_idx = {name: name for name in tickers}
          gbr_feature_importance = {name: name for name in tickers}
          gbr_best_grid = {name: name for name in tickers}
           
          grid = {'max_features': [4,8], 'learning_rate':[.01,.03,.05,1],'n_estimators': [50,200,400], 'subsample': [.6,.8], 'random_state': [42]}
   
          for tick in tickers:
              gbr_test_scores[tick] = []
              gbr_best_grid[tick] ={}
                
            # Loop through the parameter grid, set the hyperparameters, and save the scores
              for g in ParameterGrid(grid):
                  
                    gbr = GradientBoostingRegressor()
                    gbr.set_params(**g)
                    gbr.fit(train_features[tick], train_targets[tick])
                    gbr_test_scores[tick].append(gbr.score(test_features[tick], test_targets[tick]))
            
                # Find best hyperparameters from the test score and store in dictionary
              gbr_best_idx[tick] = np.argmax(gbr_test_scores[tick])
              gbr_best_grid[tick] = dict(ParameterGrid(grid)[gbr_best_idx[tick]])
              print(gbr_test_scores[tick][gbr_best_idx[tick]], ParameterGrid(grid)[gbr_best_idx[tick]])
            
            # Gradient Boosting Model
          for tick in tickers:
              gbr = GradientBoostingRegressor(**gbr_best_grid[tick])
              gbr.fit(train_features[tick], train_targets[tick])
                
              print(gbr.score(train_features[tick], train_targets[tick]))
              print(gbr.score(test_features[tick], test_targets[tick]))    
                
              gbr_feature_importance[tick] = gbr.feature_importances_
                
              plt.bar(range(features[tick].shape[1]), np.argsort(gbr_feature_importance[tick])[::-1], tick_label = np.array(features[tick].columns)[np.argsort(gbr_feature_importance[tick])[::-1]])
              plt.xticks(rotation = 90)
              plt.title(str(tick) + ' GBR Feature Importance')
              plt.show()
              
          return gbr_best_idx, gbr_best_grid
                   
=======


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
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
            
            
            


<<<<<<< HEAD



def neural_dictionaries():

    scaled_train_features = {name: name for name in tickers}
    scaled_test_features = {name: name for name in tickers}
    nn_training_pred = {name: name for name in tickers}
    nn_test_pred = {name: name for name in tickers}
    
    return scaled_train_features, scaled_test_features, nn_training_pred, nn_test_pred

=======
>>>>>>> 4834fe06c1c4ce9cd3888ca806dea55bd882c3c2
