import math
import numpy as np
import scipy.optimize as opt

def yield_to_maturity(price, par, T, cpn, freq, guess = .05):
    freq = float(freq)
    per = T * 2
    
    if isinstance(cpn , int):
        cpn = cpn /100 * par
    else:
        cpn = cpn
        
    dt = [(i + 1) / freq for i in range(int(per))]
    
    ytm = lambda y: sum([cpn / freq / (1 + y / freq) ** (freq * t) for t in dt]) + par / (1 + y/freq) ** (freq * T) - price
    return opt.newton(ytm, guess)
    
              
def price_bond(par, T, ytm, cpn, freq):
    freq = float(freq)
    per = T * 2
    cpn = cpn /100 * par
    dt = [(i + 1) / freq for i in range(int(per))]
    price = sum( [cpn / freq / (1 + ytm/freq) ** (freq * t) for t in dt]) + par/(1 + ytm / freq) ** (freq * T)
    return(price)

def mod_duration(price, par, T, cpn, freq, dy = .01):
    ytm = yield_to_maturity(price, par, T, cpn, freq)
    
    ytm_minus = ytm - dy
    price_minus = price_bond(par, T, ytm_minus, cpn, freq)
    
    ytm_plus = ytm + dy
    price_plus = price_bond(par, T, ytm_plus, cpn, freq)
    
    mduration = (price_minus - price_plus) / (2 * price * dy)
    return (mduration)


def convexity(price, par, T, cpn, freq, dy=.01):
    # higher convexity = less affected by rate volatility. Priced more expensive than low convexity.
    ytm = yield_to_maturity(price, par, T, cpn, freq)
    
    ytm_minus = ytm - dy
    price_minus = price_bond(par, T, ytm_minus, cpn, freq)
    
    ytm_plus = ytm + dy
    price_plus = price_bond(par, T, ytm_plus, cpn, freq)
    
    convexity = (price_minus + price_plus - 2 * price) / (price * dy ** 2)
    return(convexity)
           