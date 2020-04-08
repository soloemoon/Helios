import math
import scipy
import scipy.optimize as opt
import numpy as np

class bootstrap(object):
    def __init__(self):
        self.zero_rates = {}
        self.instruments = {}
        
    def add_instrument(self, par, T, cpn, price, freq = 2):
        # freq is compounding frequency
        self.instruments[T] = (par, cpn, price, freq)
        
        
    def maturities(self):
        return (sorted(self.instruments.keys()))
    
    def zero_rates(self):
        self.bootstrap_zero_cpn()
        self.spot_rate()
        return([self.zero_rates[T] for T in self.get_maturities()])
        
    def bootstrap_zero_cpn(self):
        for(T, instrument) in self.instruments.items():
            (par,cpn,price,freq) = instrument
            if cpn == 0:
                spot_rate = self.zero_cpn_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate
                
    def zero_cpn_spot_rate(self, par, price, T):
        spot_rate = math.log(par/price) / T
        return(spot_rate)
    
    def spot_rate(self):
        for T in self.maturities():
            instrument = self.instruments[T]
            (par, cpn, price, freq) = instrument
            if cpn != 0:
                spot_rate = self.calc_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate
                
    def calc_spot_rate(self, T, instrument):
        try:
            (par, cpn, price, freq) = instrument
            periods = T * freq
            value = price
            per_cpn = cpn / freq
            
            for i in range(int(periods) -1):
                t = (i+1) / float(freq)
                spot_rate = self.zero_rates[t]
                disc_cpn = per_cpn * math.exp(-spot_rate * t)
                value  = -disc_cpn
                last_per = int(periods) / float(freq)
                spot_rate = -math.log(value/ (par + per_cpn)) / last_per
                
                return(spot_rate)
        except:
                print('ERROR: Spot Rate Not Found for T =' + str(t))
                
class implied_forward(object):
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = {}
        
        # Used to add spot rates
    def spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate
     
        
    def forward_rate(self):
        per = sorted(self.spot_rates.keys())
        
        for T2, T1 in zip(per, per[1:]):
            forward_rate = self.calc_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)
            
            return(self.forward_rates)
        
    def calc_forward_rate(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = ((R2 * T2) - (R1 * T1)) / (T2 - T1)
        return(forward_rate)
        

       
        
        