import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import probplot
import numpy as np
from json import loads
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

tickers = ['AAPL', 'MSFT']

equities = Helios(tickers, '2020-01-01')
prices, returns, monthly_returns = equities.returns()

equities = Technicals(returns, 10)
rsi = equities.rsi()
moving_avg = equities.moving_average()
log_returns = equities.log_return()
bollinger = equities.bollinger_band()
exponential_average = equities.exponential_average()


class Helios:
    def __init__(self, tickers, start_date):
        self.tickers = tickers
        self.start_date = start_date
        
    def returns(self):  
        ''' Download stock prices and calculate returns based on adjust close'''
        prices = pdr.data.DataReader(self.tickers, 'yahoo', self.start_date)
        returns = prices['Adj Close'].pct_change().dropna()
        monthly_returns = prices['Adj Close'].resample('BMS').first()
        monthly_reutnrs = monthly_returns.pct_change().dropna()
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
        
        bband = {name: name for name in self.tickers}
        
        for tick in self.tickers:
            mean = self.df[tick].rolling(self.window).mean()
            sd = self.df[tick].rolling(self.window).std()
            upperband = mean + (sd * self.standard_deviation)
            lowerband = mean - (sd * self.standard_deviation)
            bband[tick] = pd.DataFrame({'Mean':mean, 'Upper':upperband, 'Lower':lowerband})
        return bband
    
    def exponential_average(self):
        ewm = self.df.ewm(span = self.window).mean()
        return ewm
    
class Forecast:
    def __init__(self, prices, batch_size, num_unroll):
        self.prices = prices['Adj Close']
        self.price_length = len(self.prices) - num_unroll
        self.batch_size = batch_size
        self.segments = self.price_length // self.batch_size
        self.cursor = [offset * self.segments for offset in range(self.batch_size)]
        
        
    def next_batch(self):    
        batch_data = np.zeros(self.batch_size)
        batch_labels = npzeros(self.batch_size)
        
        for b in range(self.batch_size):
            if self.cursor[b] + 1 >= self.price_length:
                self.cursor[b] = np.random.randint(0, (b + 1) * self.segments)
                
        batch_data[b] = self.prices[self.cursor[b]]
        batch_labels[b] = self.prices[self.cursor[b] + np.random.randint(0,5)]
        self.cursor[b] = (self.cursor[b] + 1) % self.price_length
        
        return batch_data, batch_labels
    
    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        
        for i in range(self.num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)
            
        return unroll_data, unroll_labels
        
    def reset_indices(self):
        for b in range(self.batch_size):
            self.cursor[b] = np.random.randint(0, min((b + 1) *self.segments, self.price_length - 1))
          
    def forecast(self):
        train = self.prices
        dg = DataGneratorSeq(train, 5, 5)
        u_data, u_labels = dg.unroll_batches()

        for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
            print('\n\nUnrolled index %d'%ui)
            dat_ind = dat
            lbl_ind = lbl
            print('\tInputs: ', dat)
            print('\n\tOutput:',lbl)
            
        parameters ={'D':1, 'num_unrollings': 50, 'batch_size':500, 'num_nodes':[200,200,150],'n_layers':3, 'dropout':.2}    
        tf.reset_defualt_graph()
        
        train_inputs, train_outputs = [], []
        
        for i in range(parameters['num_unrollings']):
            train_inputs.append(tf.placeholder(tf.float32, shape=[parameters['batch_size'], parameters['D']], name='train_inputs_%d'%ui))
            train_outputs.append(tf.placeholder(tf.float32, shape=[parameters['batch_size'], 1], name='train_outputs_%d'%ui))
            
        cells = [tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],state_is_tuple=True,initializer= tf.contrib.layers.xavier_initializer() ) for li in range(n_layers)]

        drop_cells = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout) for lstm in cells]
        drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_cells)
        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
        
        w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))
        
        c, h, initial = [], [],[]
        
        for i in range(n_layers):
            c.append(tf.Variable(tf.zeros([parameters['batch_size'], parameters['num_nodes'][i]]), trainable = False))
            h.append(tf.Variable(tf.zeros([parameters['batch_size'], parameters['num_nodes'][i]]), trainable = False))
            initial.append(tf.contrib.rnn.LSTMStateTuple(c[i], h[i]))
            
            all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis = 0)
            
            all_lstm_outputs, state = tf.nn.dynamic_rnn(drop_multi_cell, all_inputs, initial_state = tuple(initial), time_major = True, dtype = tf.float32 )
            
            all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)
            
            split_outputs = tf.split(all_outputs, num_unrollings, axis = 0)
            
            
        loss = 0
         
        with tf.control_dependencies([tf.assign(c[i], state[i][0]) for i in range(parameters['n_layers'])] + [tf.assign(h[i], state[i][1]) for i in range(parameters['n_layers'])]):
             
            for ui in range(parameters['num_unrollings']):
                 loss += tf.reduce_mean(.5 * (split_outputs[i] - train_outputs[i]) ** 2)
                 
        global_step = tf.Variable(0, trainable = False)
        inc_gstep = tf.assign(global_step, global_step + 1)
        tf_learning_rate = tf.placeholder(shape = None, dtype = tf.float32)
        tf_min_learning_rate = tf.placeholder(shape = Non, dtype = tf.float32)
        
        learning_rate = tf.maximum( tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps =1, decay_rate = .5, staircase = True), tf_min_learning_rate)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = optimizer.apply_gradients( zip(gradients, v))
        
        sample_inputs = tf.placeholder(tf.float32, shape=[1, parameters['D']])
        
        sample_c, sample_h, initial_sample_state = [], [], []
        
        for i in range(parameters['n_layers']):
            sample_c.append(tf.Variable(tf.zeros([1, parameters['num_nodes'][i]]), trainable= False))
            sample_h.append(tf.Variable(tf.zeros([1, parameters['num_nodes'][i]]), trainable= False))
            initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))

        reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                               *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

        sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                   initial_state=tuple(initial_sample_state),
                                   time_major = True,
                                   dtype=tf.float32)

        with tf.control_dependencies([tf.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                                      [tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
          sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)
             
        epochs = 30
        valid_summary = 1
        n_predict_once = 50
        train_seq_length = train_data_size
        
        train_mse_ot = []
        test_mse_ot = []
        predictions_over_time = []
        
        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    # Used for decaying learning rate
    loss_nondecrease_count = 0
    loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate
    
    print('Initialized')
    average_loss = 0
    
    # Define data generator
    data_gen = tf.DataGeneratorSeq(train_data,batch_size,num_unrollings)
    
    x_axis_seq = []
    
    # Points you start your test predictions from
    test_points_seq = np.arange(11000,12000,50).tolist()
    
    for ep in range(epochs):       
    
        # ========================= Training =====================================
        for step in range(train_seq_length//batch_size):
    
            u_data, u_labels = data_gen.unroll_batches()
    
            feed_dict = {}
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
                feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)
    
            feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})
    
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    
            average_loss += l
    
        # ============================ Validation ==============================
        if (ep+1) % valid_summary == 0:
    
          average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))
    
          # The average loss
          if (ep+1)%valid_summary==0:
            print('Average loss at step %d: %f' % (ep+1, average_loss))
    
          train_mse_ot.append(average_loss)
    
          average_loss = 0 # reset loss
    
          predictions_seq = []
    
          mse_test_loss_seq = []
    
          # ===================== Updating State and Making Predicitons ========================
          for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []
    
            if (ep+1)-valid_summary==0:
              # Only calculate x_axis values in the first validation epoch
              x_axis=[]
    
            # Feed in the recent past behavior of stock prices
            # to make predictions from that point onwards
            for tr_i in range(w_i-num_unrollings+1,w_i-1):
              current_price = all_mid_data[tr_i]
              feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)    
              _ = session.run(sample_prediction,feed_dict=feed_dict)
    
            feed_dict = {}
    
            current_price = all_mid_data[w_i-1]
    
            feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
    
            # Make predictions for this many steps
            # Each prediction uses previous prediciton as it's current input
            for pred_i in range(n_predict_once):
    
              pred = session.run(sample_prediction,feed_dict=feed_dict)
    
              our_predictions.append(np.asscalar(pred))
    
              feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)
    
              if (ep+1)-valid_summary==0:
                # Only calculate x_axis values in the first validation epoch
                x_axis.append(w_i+pred_i)
    
              mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2
    
            session.run(reset_sample_states)
    
            predictions_seq.append(np.array(our_predictions))
    
            mse_test_loss /= n_predict_once
            mse_test_loss_seq.append(mse_test_loss)
    
            if (ep+1)-valid_summary==0:
              x_axis_seq.append(x_axis)
    
          current_test_mse = np.mean(mse_test_loss_seq)
    
          # Learning rate decay logic
          if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
              loss_nondecrease_count += 1
          else:
              loss_nondecrease_count = 0
    
          if loss_nondecrease_count > loss_nondecrease_threshold :
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')
    
          test_mse_ot.append(current_test_mse)
          print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
          predictions_over_time.append(predictions_seq)
          print('\tFinished Predictions')
        
         
                                      
        



        
    def price_forecast(self):
        df = self.df['Adj Close']
            
        train = df[ :round(len(df) * .8)]
        test = df[round(len(df) * .8): ]
        
        scaled_train = pd.DataFrame(MinMaxScaler().fit_transform(df))
        scaled_train.columns = self.tickers
            
        EMA = 0.0
        
        for i in len(scaled_train):
            EMA = 0.1 * scaled_train[i] + (1 - 0.1) * EMA
            scaled_train = train[i] = EMA
            
    
      
        
       

     
        
    
    
class Financials:
    def __init__(self):
        
    
    
    
    
    
     
    def financial_statement(self, ticker):
        self.ticker = ticker
        
        balance = requests.get(f'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/'+str(ticker)+'?period=quarter')
        income = requests.get(f'https://financialmodelingprep.com/api/v3/financials/income-statement/'+str(ticker)+'?period=quarter')
        cashflow = requests.get(f'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/'+str(ticker)+'?period=quarter')
        
        def download_statement(statement):
            statement = statement.json()
            statement = statement['financials']
            statement = pd.DataFrame.from_dict(statement)
            statement = statement.T
            statement.columns = statement.iloc[0]
            
            cols = statement.columns
            statement = statement[cols]
            return statement
        
        balance_sheet = download_statement(balance)
        income_statement = download_statement(income)
        cashflow_statement = download_statement(cashflow)
        
        return balance_sheet, income_statement, cashflow_statement
    
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
                                 
            
        
       


       
