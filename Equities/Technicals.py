import numpy as np
import pandas as pd

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