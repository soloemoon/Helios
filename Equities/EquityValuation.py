import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras.losses
from keras.layers import Dense, Dropout
from sklearn.metrics import r2_score

# Define Functions

def rsi(df, window):
    delta = df.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0 ] = 0
    RSI =  100-100/(1+(+up.rolling(window).mean() / down.abs().rolling(window).mean()))
    return(RSI)

def ma(df, window_length):
    ma = df.rolling(window = window_length).mean()
    return(ma)

def logreturn(df):
    log = np.log(df).diff().cumsum()
    return(log)

def totalreturn(df):
    total = 100 * (np.exp(logreturn(df))) - 1
    return(total)

def bollingerband(df, window, sd):
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    upperband = mean + (std * sd)
    lowerband = mean - (std * sd)
    bband = pd.DataFrame({'Mean':mean, 'Upper':upperband,'Lower':lowerband})
    return(bband)

# Penalizies ML model for getting predicted direction of stock movements wrong
def lossfunction(y_true, y_pred):
    loss = tf.where(tf.less(y_true * y_pred, 0), 100 * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
    return tf.reduce_mean(loss, axis=-1)
# Enable use of custom loss function
keras.losses.lossfunction = lossfunction

# Set Chart Style
plt.style.use('fivethirtyeight')

# Define stock prices to download
tickers = ['AAPL', 'MSFT']

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
prices = pdr.data.DataReader(tickers, 'yahoo', '2010-01-01')
returns = prices['Adj Close'].pct_change().dropna()
future_returns = prices['Adj Close'].shift(-5).pct_change(5).dropna()
monthly_returns = returns.resample('BMS').first()

#--------------- Dictionaries of Dataframes---------------#
corr_5d = {name: name for name in tickers}
bband = {name: name for name in tickers}
analysis = {name: name for name in tickers}
features = {name: name for name in tickers}
train_size = {name: name for name in tickers}
train_features = {name: name for name in tickers}
train_targets = {name: name for name in tickers}
test_features = {name: name for name in tickers}
test_targets = {name: name for name in tickers}
rfr_test = {name: name for name in tickers}
rfr_train = {name: name for name in tickers}
rfr_test_scores = {name: name for name in tickers}
rfr_best_idx = {name: name for name in tickers}
rfr_feature_importance = {name: name for name in tickers}
rfr_best_grid = {name: name for name in tickers}
gbr_test_scores = {name: name for name in tickers}
gbr_best_idx = {name: name for name in tickers}
gbr_feature_importance = {name: name for name in tickers}
gbr_best_grid = {name: name for name in tickers}
gbr_feature_importance = {name: name for name in tickers}
scaled_train_features = {name: name for name in tickers}
scaled_test_features = {name: name for name in tickers}
nn_training_pred = {name: name for name in tickers}
nn_test_pred = {name: name for name in tickers}

#------------- Portfolio Return Plots --------------#
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9))
# Plot cumulative log returns
for tick in tickers:
    ax1.plot(logreturn(prices['Adj Close'][tick]), label = str(tick))
    ax1.set_ylabel('Cumulative Log Returns')
    ax1.legend()
# Plot Total Return
for tick in tickers:
    ax2.plot(totalreturn(prices['Adj Close'][tick]),label = str(tick))
    ax2.set_ylabel('Total relative returns (%')
    ax2.legend(loc='best')
plt.show()

# Generate visualizations
for tick in tickers:
    #---------- Simple Moving Averages --------#
    plt.subplots(figsize =(16,9))
    plt.plot(prices['Adj Close'][tick], label = 'Closing Price')
    plt.plot(ma(prices['Adj Close'][tick], 20), label = '20 day SMA')
    plt.plot(ma(prices['Adj Close'][tick],100), label = '100 day SMA')
    plt.title(str(tick) + ' Simple Moving Averages')
    plt.legend()
    plt.ylabel('Adjusted Closing Price')
    plt.show()
    
    #------- Histogram ---------#
    plt.hist(prices['Adj Close'][tick].pct_change(), bins = 50)
    plt.title(tick)
    plt.xlabel('Adjusted Close 1 Day % Chg')
    plt.show()
    
    #------ RSI -------#
    plt.plot(rsi(prices['Adj Close'][tick], 14).tail(14))
    plt.title(str(tick) + ' RSI')
    plt.show()
    
    # Compute Future close correlations with current close to determine mean reversion
    corr_5d[tick] = prices['Adj Close'][tick].pct_change(10).corr(prices['Adj Close'][tick].shift(-5).pct_change(5))
    
    #------ Scatter Plots ------#
    plt.scatter(prices['Adj Close'][tick].pct_change(5),  prices['Adj Close'][tick].shift(-5).pct_change(5))
    plt.title(str(tick) + ' Mean Reversion')
    plt.show()
    
    #----- Bollinger Bands------#
    bband[tick] = bollingerband(prices['Adj Close'][tick],20, 2)
    # Create Bollinger Band Plot
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.fill_between(bband[tick].index.get_level_values(0),bband[tick]['Upper'], bband[tick]['Lower'], color='grey')
    ax.plot(bband[tick].index.get_level_values(0),bband[tick]['Mean'], color='blue', lw=2)
    ax.set_title('20 Day Bollinger Band For '+str(tick))
    ax.set_ylabel('Price')
    plt.show()

# --------------- Create Features------------#    
for tick in tickers:
    analysis[tick] = pd.DataFrame()
    # Generate moving averages and RSIs for range of days
for tick in tickers:
    for n in [14,30,50,200]:
        analysis[tick]['ma' + str(n)] = ma(prices['Adj Close'][tick],n)
        analysis[tick]['rsi' + str(n)] = rsi(prices['Adj Close'][tick],n)

# Plot features and Targets
for tick in tickers:
    # Add MA/RSI by day and returns to features
    features[tick] = pd.merge(pd.DataFrame(returns[tick]), analysis[tick],how = 'inner', right_index=True, left_index = True )
    features[tick].rename(columns={features[tick].columns[0]: "10D Return" }, inplace = True)
    # Add volumes to features
    features[tick]['Volume 1D Pct'] = prices['Volume'][tick].pct_change()
    features[tick]['Volume 1D Pct MA'] = ma(prices['Volume'][tick].pct_change(),5)
    
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
   
# --------------------- Model Development ---------------#
#--------------------------------------------------------#
    
#---------------------- Random Forest Model --------------#
# Find best Random Forest Parameters      
grid = {'n_estimators': [50,200,400], 'max_depth': [3,5,10,20,50], 
        'max_features': [4,8], 'random_state': [42]}
   
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
    
#-----------------Gradient Boost Model----------------------#
# Find optimal gbr params
grid = {'max_features': [4,8], 'learning_rate':[.01,.03,.05,1],
        'n_estimators': [50,200,400], 'subsample': [.6,.8], 'random_state': [42]}
   
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
    
#------------------- Neural Network ----------------------#
for tick in tickers:
    nn = Sequential()
    nn.add(Dense(100, input_dim = scaled_train_features[tick].shape[1], activation = 'relu'))
    nn.add(Dropout(rate=0.2))
    nn.add(Dense(20, activation = 'relu'))
    nn.add(Dense(1, activation='linear'))
    nn.compile(optimizer='adam', loss='lossfunction')
    history = nn.fit(scaled_train_features[tick], train_targets[tick], epochs=25)
    
    plt.plot(history.history['loss'])
    
    # Use the last loss as the title - should trend downward
    plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
    plt.show()
    
    # Score model
    nn_training_pred[tick] = nn.predict(scaled_train_features[tick])
    nn_test_pred[tick] = nn.predict(scaled_test_features[tick])
    print(r2_score(train_targets[tick], nn_training_pred[tick]))
    print(r2_score(nn_test_pred[tick], test_targets[tick]))
    
    # Training Set
    plt.scatter(nn_training_pred[tick], train_targets[tick], label ='Train')
    plt.title(str(tick) + ' Prediction v Actual')
    plt.legend(); plt.show()
    
    # Test Set
    plt.scatter(nn_test_pred[tick], test_targets[tick], label ='Test')
    plt.title(str(tick) + ' Prediction v Actual')
    plt.legend(); plt.show()

# Optimize Portfolio
covariances = {}

for i in monthly_returns.index:
    mask = (returns.index.month == i.month)&(returns.index.year == i.year)
    covariances[i] = returns[mask].cov()


port_returns, port_vol, port_weight = {}, {}, {}

# Generate random weights and set to sum to 1. 
for date in sorted(covariances.keys()):
    cov = covariances[date]
    for portfolio in range(10):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        returns = np.dot(weights, monthly_returns.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        port_returns.setdefault(date, []).append(returns)
        port_vol.setdefault(date, []).append(volatility)
        port_weight.setdefault(date, []).append(weights)
               
date = sorted(covariances.keys())[-1]  

# Plot efficient frontier
plt.scatter(x=port_vol[date], y=port_returns[date],  alpha=.1)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()      
        
# Sharpe Optimization
sharpe_ratio, max_sharpe_idxs = {}, {}

for date in port_returns.keys():
    for i, ret in enumerate(port_returns[date]):
        sharpe_ratio.setdefault(date, []).append(returns / port_vol[date][i])
        max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date][i])
print(port_returns[date][max_sharpe_idxs[date]])

targets, features = [],[]

 
    # Make Heatmap
    #sns.heatmap(round(feat_targ[tick].corr(),2), annot=True, annot_kws={"size":5})
    #plt.yticks(rotation=0, size=5)
    #plt.xticks(rotation=90, size=5)
    #plt.tight_layout()
    #plt.show()
    #Add scatter plot with two highest correlated pairs
   #t =  max(feat_targ['MSFT'].corr().unstack().sort_values()
   #[max(t.index[t != 1])]

