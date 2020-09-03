import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import probplot
import requests
import json
import numpy as np
from json import loads
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


class Helios:
    def __init__(self, tickers, start_date):
        self.tickers = tickers
        self.start_date = start_date
        
    def returns(self):  
        ''' Download stock prices and calculate returns based on adjust close'''
        prices = pdr.data.DataReader(self.tickers, 'yahoo', self.start_date)
        returns = prices['Adj Close'].pct_change().dropna()
        monthly_returns = prices['Adj Close'].resample('BMS').first()
        monthly_returns = monthly_returns.pct_change().dropna()
        return prices, returns, monthly_returns
     
class Technicals:
    def __init__(self, df,window = 10, standard_deviation = 2):
        self.df = df
        self.window = window
        self.standard_deviation = standard_deviation
       
        
    def rsi(self):
        delta = self.df.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0 ] = 0
        rsi =  100-100/(1 + ( +up.rolling(self.window).mean() / down.abs().rolling(self.window).mean() ) )
        return rsi
    
    def moving_average(self):
        ma = self.df.rolling(window = self.window).mean()
        return ma
    
    def log_return(self):
        logr = np.log(self.df).diff().cumsum()
        return logr
    
    def bollinger_band(self):
        mean = self.df.rolling(self.window).mean()
        sd = self.df.rolling(self.window).std()
        upperband = mean + (sd * self.standard_deviation)
        lowerband = mean - (sd * self.standard_deviation)
        bband = pd.DataFrame({'Mean':mean, 'Upper':upperband, 'Lower':lowerband})
        return bband
    
    def exponential_average(self):
        ewm = self.df.ewm(span = self.window).mean()
        return ewm
    
    
    def beta(self,start_date):
        market = pdr.get_data_yahoo('SPY',start_date)
        market_returns = market['Adj Close'].resample('BMS').first()
        market_returns = market_returns.pct_change().dropna()
        
        betas = {col:col for col in self.df.columns}
        tickers = [col for col in self.df.columns]
        
        for tick in tickers:
            covariance = np.cov(self.df[tick], market_returns)
            covariance = covariance[0,1] / covariance[1,1]
            betas[tick] = betas.append(covariance)
               
        return betas
        
        
        
    
    
class Financials:
    def __init__(self, tickers):
        self.tickers = tickers
        
    def income_statement(self):
        income_statements = {name: name for name in self.tickers}
        
        for tick in self.tickers:
            incst_link = requests.get(f'https://financialmodelingprep.com/api/v3/financials/income-statement/'+ {self.ticker} +'?period=quarter')
            
            incst = incst_link.json()
            incst = incst['financials']
            income_statement = pd.DataFrame.from_dict(incst)
            income_statement = income_statement.T
            income_statement.columns = income_statement.iloc[0]
            income_statement = income_statement[income_statement.columns]
            income_statements[tick] = income_statement
        return income_statements
    
    def balance_sheet(self):
        
        balance_sheets = {name: name for name in self.tickers}
           
        for tick in self.tickers:
            blncsht_link = requests.get(f'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/'+ {self.ticker} +'?period=quarter')
            
            bs = blncsht_link.json()
            bs = bs['financials']
            balance_sheet = pd.DataFrame.from_dict(bs)
            balance_sheet = balance_sheet.T
            balance_sheet.columns = balance_sheet.iloc[0]
            balance_sheet = balance_sheet[balance_sheet.columns]
            balance_sheets[tick] = balance_sheet
        return balance_sheets
    
    def cash_flow(self):
        cash_flows = {name: name for name in self.tickers}
           
        for tick in self.tickers:
            cf_link = requests.get(f'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/'+ {self.ticker} +'?period=quarter')
            
            cfs = cf_link.json()
            cfs = cfs['financials']
            cash_flow = pd.DataFrame.from_dict(cfs)
            cash_flow = cash_flow.T
            cash_flow.columns = cash_flow.iloc[0]
            cash_flow = cash_flow[cash_flow.columns]
            cash_flows[tick] = cash_flow
        return cash_flows
    
    def common_size_bs(self, balance_sheet):
        cols = balance_sheet.columns
        commonsize_bs = balance_sheet[cols].div(balance_sheet.loc['Total assets'])
        return commonsize_bs
    
    def common_size_is(self, income_statement): # NOT WORKING
        cols = income_statement.columns
        commonsize_is = income_statement[cols].div(income_statement.loc['Revenue'])
        return commonsize_is
    
    def ratios(self, df, item):
        ratio = pd.to_numeric(df.loc[item])
        return(ratio)
    
    def growth_rates(self, income_statement, balance_sheet):# NOT WORKING
        
        is_growth_rates = income_statement.apply(lambda x: (x /x.shift(-1)) - 1, axis=1)
        bs_growth_rates = balance_sheet.apply(lambda x: (x /x.shift(-1)) - 1, axis=1)
        
        return is_growth_rates, bs_growth_rates
    
    def discretionary_accrual_model(self, income_statement, balance_sheet, cashflow_statement):
        discretionary_accrual = pd.DataFrame()
        
        discretionary_accrual['Accruals'] = (Financials.ratios(income_statement,'Net Income') - Financials.ratios(cashflow_statement,'Operating Cash Flow')) / Financials.ratios(balance_sheet,'Total assets').shift(-1)
        discretionary_accrual['Cash Revenue Growth'] = ( (Financials.ratios(income_statement,'Revenue') - Financials.ratios(income_statement, 'Revenue').shift(-1))- ( Financials.ratios(balance_sheet, 'Receivables') - Financials.ratios(balance_sheet, 'Receivables').shift(-1) ) ) / Financials.ratios(balance_sheet,'Total assets').shift(-1)
     # Need to use gross PP&E. Not currently available. Pull from Yahoo Finance?
        discretionary_accrual['PP&E'] = Financials.ratios(balance_sheet, 'Property, Plant & Equipment Net') / Financials.ratios(balance_sheet,'Total assets').shift(-1)
    
        discretionary_accrual_train = discretionary_accrual.tail(len(discretionary_accrual) - 4 )
        discretionary_accrual_train = discretionary_accrual_train.replace([np.inf, -np.inf], np.nan)
        discretionary_accrual_train = discretionary_accrual_train.dropna()
        
        regression = LinearRegression().fit(discretionary_accrual_train[['Cash Revenue Growth', 'PP&E']] , discretionary_accrual_train['Accruals'] )
        print(regression.score(discretionary_accrual_train[['Cash Revenue Growth', 'PP&E']] , discretionary_accrual_train['Accruals']))
        
        discretionary_accrual_estimate = discretionary_accrual['Accruals'].head(4) - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual['Cash Revenue Growth'].head(4)) + ( regression.coef_[1] * discretionary_accrual['PP&E'].head(4)))
        discretionary_accrual_estimate['Max'] = max(discretionary_accrual_train['Accruals'] - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual_train['Cash Revenue Growth']) + ( regression.coef_[1] * discretionary_accrual_train['PP&E'])))
        discretionary_accrual_estimate['Min'] = min(discretionary_accrual_train['Accruals'] - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual_train['Cash Revenue Growth']) + ( regression.coef_[1] * discretionary_accrual_train['PP&E'])))
               
        return discretionary_accrual_estimate
                                 
         # Add LTSM Model for Stock Price Prediction
        
       


       
