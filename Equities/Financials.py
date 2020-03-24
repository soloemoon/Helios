import pandas as pd
import requests
import json
import numpy as np
from json import loads
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Define tickers to analyze
ticker = ['JBLU', 'AAPL']

# Set Financial Statement Output Path(s)

# Define Data Dictionaries    
income_statements = {tick: tick for tick in ticker}
balance_sheets = {tick: tick for tick in ticker}
cash_flow_statements = {tick: tick for tick in ticker}
ratios = {tick: tick for tick in ticker}
bs_growth_rates = {tick: tick for tick in ticker}
is_growth_rates = {tick: tick for tick in ticker}
commonsize_balance_sheets = {tick: tick for tick in ticker}
commonsize_income_statements = {tick: tick for tick in ticker}
discretionary_accrual = {tick: tick for tick in ticker}
discretionary_accrual_train = {tick: tick for tick in ticker}
discretionary_accrual_estimate = {tick: tick for tick in ticker}
discretionary_expenditures = {tick: tick for tick in ticker}
discretionary_expenditures_train = {tick: tick for tick in ticker}
discretionary_expenditures_estimate = {tick: tick for tick in ticker}


#Define Functions - Financial data pulled from financial modeling prep
def finstatement(ticker, statement):
    
    if statement == 'bs':
        statement = requests.get(f'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/'+str(ticker)+'?period=quarter')
      
    elif statement == 'is': 
        statement = requests.get(f'https://financialmodelingprep.com/api/v3/financials/income-statement/'+str(ticker)+'?period=quarter')
        
    elif statement == 'cf':
        statement = requests.get(f'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/'+str(ticker)+'?period=quarter')
                 
    statement = statement.json()
    statement = statement['financials']
    statement = pd.DataFrame.from_dict(statement)
    statement = statement.T
    statement.columns = statement.iloc[0]
    
    cols = statement.columns
    statement = statement[cols].apply(pd.to_numeric, errors='coerce')
    return(statement)

# Download Financial Statements
for tick in ticker:
     income_statements[tick] = finstatement(tick,'is')
     balance_sheets[tick] = finstatement(tick,'bs')
     cash_flow_statements[tick] = finstatement(tick,'cf')

# Ratio Analysis Function
def ra(statement,tick,item):
    if statement == 'is':
        ratio = income_statements[tick].loc[item]
    elif statement == 'bs':
        ratio = balance_sheets[tick].loc[item]
    elif statement == 'cf':
        ratio = cash_flow_statements[tick].loc[item]
    return(ratio)

# Format Dictionaries as Dataframes
for tick in ticker:
    ratios[tick] = pd.DataFrame()
    discretionary_accrual[tick] = pd.DataFrame()
    discretionary_accrual_train[tick] = pd.DataFrame()
    discretionary_expenditures[tick] = pd.DataFrame()
    discretionary_expenditures_train[tick] = pd.DataFrame()
    bs_growth_rates[tick] = pd.DataFrame()
    is_growth_rates[tick] = pd.DataFrame()
        
# Accounting Analytics
for tick in ticker:
# Dupont Analysis - ROE Decomposition
    ratios[tick]['Return on Sales'] = (ra('is',tick,'Net Income') + ra('is', tick,'Income Tax Expense'))/ ra('is',tick,'Revenue')
    ratios[tick]['Asset Turnover'] = ra('is',tick,'Revenue') / ra('bs',tick,'Total assets')
    ratios[tick]['Financial Leverage'] = ra('bs',tick,'Total assets') / ra('bs',tick,'Total shareholders equity')
    ratios[tick]['Return on Equity'] = ratios[tick]['Return on Sales'] * ratios[tick]['Asset Turnover'] * ratios[tick]['Financial Leverage']

# Tax Rates
    ratios[tick]['Tax Rate'] =ra('is', tick, 'Income Tax Expense') / ra('is', tick, 'Earnings before Tax')
    ratios[tick]['Interest Expense to Sales'] =ra('is', tick, 'Interest Expense') / ra('is', tick, 'Revenue')

# Turnover Ratios
    ratios[tick]['Days Receivable'] = 365 * (ra('bs', tick, 'Receivables') / ra('is', tick, 'Revenue'))
    ratios[tick]['Days Inventory'] = 365 * (ra('bs', tick, 'Inventories') / ra('is', tick,'Cost of Revenue') )
    ratios[tick]['Days Payable'] = 365 * ( ra('bs',tick,'Payables') / ra('is', tick, 'Cost of Revenue')  )
    ratios[tick]['Fixed Asset Turnover'] = ra('is', tick, 'Revenue') / ra('bs', tick,'Property, Plant & Equipment Net')
    ratios[tick]['Net Trade Cycle'] =ratios[tick]['Days Receivable'] + ratios[tick]['Days Inventory'] - ratios[tick]['Days Payable']

# Liquidity Ratios Ideally >1
    ratios[tick]['Quick Ratio'] = (ra('bs', tick,'Cash and cash equivalents') + ra('bs', tick, 'Receivables')) / ra('bs', tick, 'Total current liabilities')
    ratios[tick]['CFO to Current Liabilities'] = ra('cf',tick,'Operating Cash Flow') / ra('bs', tick, 'Total current liabilities')

# Interest Coverage Ratios Ideally > 1
    ratios[tick]['Interest Coverage'] = ra('is', tick, 'EBITDA') / ra('is', tick, 'Interest Expense')
    ratios[tick]['Cash Interest Coverage'] = (ra('cf', tick, 'Operating Cash Flow') + ra('is', tick, 'Interest Expense') + ra('is', tick, 'Income Tax Expense') ) / ra('is', tick, 'Interest Expense')

# Long-Term Debt Ratios
    ratios[tick]['Debt to Equity'] = ra('bs',tick,'Total liabilities') / ra('bs',tick,'Total shareholders equity')
    ratios[tick]['Term Debt to Equity'] = ra('bs',tick,'Long-term debt') / ra('bs',tick,'Total shareholders equity')
    ratios[tick]['Debt to Tangible Assets'] = ra('bs',tick,'Long-term debt') / (ra('bs',tick,'Total assets') - ra('bs',tick,'Goodwill and Intangible Assets' ) )

# Common size financial statements
for tick in ticker:
    cols = balance_sheets[tick].columns
    commonsize_balance_sheets[tick] = balance_sheets[tick][cols].div(balance_sheets[tick].loc['Total assets'])
    cols = income_statements[tick].columns
    commonsize_income_statements[tick] = income_statements[tick][cols].div(income_statements[tick].loc['Revenue'])

# Discretionary Accruals Model. Regression to estimate what accruals should be given revenue growth. Used to identify management inflation.
for tick in ticker:
# Scaled by total assets to remove firm size effect
    discretionary_accrual[tick]['Accruals'] = (ra('is',tick,'Net Income') - ra('cf',tick,'Operating Cash Flow')) / ra('bs', tick, 'Total assets').shift(-1)
    discretionary_accrual[tick]['Cash Revenue Growth'] = ( (ra('is', tick, 'Revenue') - ra('is', tick, 'Revenue').shift(-1))- ( ra('bs', tick, 'Receivables') - ra('bs', tick, 'Receivables').shift(-1) ) ) / ra('bs',tick,'Total assets').shift(-1)
 # Need to use gross PP&E. Not currently available. Pull from Yahoo Finance?
    discretionary_accrual[tick]['PP&E'] = ra('bs', tick,'Property, Plant & Equipment Net') / ra('bs', tick, 'Total assets').shift(-1)

    discretionary_accrual_train[tick] = discretionary_accrual[tick].tail(len(discretionary_accrual[tick]) - 4 )
    discretionary_accrual_train[tick] = discretionary_accrual_train[tick].replace([np.inf, -np.inf], np.nan)
    discretionary_accrual_train[tick] = discretionary_accrual_train[tick].dropna()
    
    regression = LinearRegression().fit(discretionary_accrual_train[tick][['Cash Revenue Growth', 'PP&E']] , discretionary_accrual_train[tick]['Accruals'] )
    print(regression.score(discretionary_accrual_train[tick][['Cash Revenue Growth', 'PP&E']] , discretionary_accrual_train[tick]['Accruals']))
    
    discretionary_accrual_estimate[tick] = discretionary_accrual[tick]['Accruals'].head(4) - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual[tick]['Cash Revenue Growth'].head(4)) + ( regression.coef_[1] * discretionary_accrual[tick]['PP&E'].head(4)))
    discretionary_accrual_estimate[tick]['Max'] = max(discretionary_accrual_train[tick]['Accruals'] - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual_train[tick]['Cash Revenue Growth']) + ( regression.coef_[1] * discretionary_accrual_train[tick]['PP&E'])))
    discretionary_accrual_estimate[tick]['Min'] = min(discretionary_accrual_train[tick]['Accruals'] - (regression.intercept_ + (regression.coef_[0] * discretionary_accrual_train[tick]['Cash Revenue Growth']) + ( regression.coef_[1] * discretionary_accrual_train[tick]['PP&E'])))
                                                      
# Discreitonary Expenditures Model - ADD SG&A as a separate regression. Negative is an indication of trying to boost earnings.
for tick in ticker:
    discretionary_expenditures['R&D'] = ( ra('is',tick,'R&D Expenses') - ra('is',tick,'R&D Expenses').shift(-1) ) / ra('bs', tick, 'Total assets').shift(-1)
    discretionary_expenditures['Revenue'] = (ra('is', tick, 'Revenue') - ra('is', tick, 'Revenue').shift(-1) ) / ra('bs', tick, 'Total assets').shift(-1)
    discretionary_expenditures['PriorRevenue'] = ra('is', tick, 'Revenue').shift(-1)  / ra('bs', tick, 'Total assets').shift(-1)
    
    discretionary_expenditures_train[tick] = discretionary_expenditures[tick].tail(len(discretionary_expenditures[tick]) - 4 )
    discretionary_expenditures_train[tick] = discretionary_expenditures_train[tick].replace([np.inf, -np.inf], np.nan)
    discretionary_expenditures_train[tick] = discretionary_expenditures_train[tick].dropna()
    
    regression = LinearRegression().fit(discretionary_expenditures_train[tick][['Revenue', 'PriorRevenue']] , discretionary_accrual_train[tick]['R&D'] )
    print(regression.score(discretionary_expenditures_train[tick][['Revenue', 'PriorRevenue']] , discretionary_accrual_train[tick]['R&D']))
    
    discretionary_expenditures_estimate[tick] = discretionary_expenditures[tick]['R&D'].head(4) - (regression.intercept_ + (regression.coef_[0] * discretionary_expenditures[tick]['Revenue'].head(4))  + (regression.coef_[1] * discretionary_accrual[tick]['PriorRevenue'].head(4)) )
    discretionary_expenditures_estimate[tick]['Max'] = max(discretionary_expenditures_train[tick]['R&D'] - (regression.intercept_ + (regression.coef_[0] * discretionary_expenditures_train[tick]['Revenue']) + ( regression.coef_[1] * discretionary_accrual_train[tick]['PriorRevenue'])))
    discretionary_expenditures_estimate[tick]['Min'] = min(discretionary_expenditures_train[tick]['R&D'] - (regression.intercept_ + (regression.coef_[0] * discretionary_expenditures_train[tick]['Revenue']) + ( regression.coef_[1] * discretionary_accrual_train[tick]['PriorRevenue'])))
    
# Remove unneeded training variables
del [discretionary_accrual_train, discretionary_expenditures_train]

# Forecast Financial Statements

# Calculate Growth Rates
for tick in ticker:
    is_growth_rates[tick] = income_statements[tick].apply(lambda x: (x /x.shift(-1)) - 1, axis=1)
    bs_growth_rates[tick] = balance_sheets[tick].apply(lambda x: (x /x.shift(-1)) - 1, axis=1)