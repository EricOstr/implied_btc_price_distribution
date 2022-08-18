import numpy as np
from scipy.stats import norm
import datetime
import pandas as pd


def blackScholes(flag, S0, K, T, r, σ):
    "Calculate BS price of call/put"
    d1 = (np.log(S0/K) + (r + σ**2/2)*T)/(σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    if flag == "C":
        price = S0*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif flag == "P":
        price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S0*norm.cdf(-d1, 0, 1)
    return price


def delta(flag, S0, K, T, r, σ):
    d1 = (np.log(S0/K) + T*(r + σ**2/2))/(σ*np.sqrt(T))
    if flag == "C":
        delta = norm.cdf(d1, 0, 1)
    elif flag == "P":
        delta = -norm.cdf(-d1, 0, 1)

    return delta


def gamma(flag, S0, K, T, r, σ):
    d1 = (np.log(S0/K) + T*(r + σ**2/2))/(σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    gamma_calc = norm.pdf(d1, 0, 1)/(S0*σ*np.sqrt(T))
    
    return gamma_calc


def vega(flag, S0, K, T, r, σ):
    d1 = (np.log(S0/K) + T*(r + σ**2/2))/(σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    
    vega_calc = S0*norm.pdf(d1, 0, 1)*np.sqrt(T)
    
    return vega_calc*0.01


def theta(flag, S0, K, T, r, σ):
    d1 = (np.log(S0/K) + T*(r + σ**2/2))/(σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    
    if flag == "C":
        theta_calc = -S0*norm.pdf(d1, 0, 1)*σ/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif flag == "P":
        theta_calc = -S0*norm.pdf(d1, 0, 1)*σ/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        
    return theta_calc/365


def rho(flag, S0, K, T, r, σ):
    d1 = (np.log(S0/K) + T*(r + σ**2/2))/(σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    
    if flag == "C":
        rho_calc = K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif flag == "P":
        rho_calc = -K*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        
    return rho_calc*0.01


def iv_newton_raphson(flag, S0, K, T, r, market_price, tol=0.01):
    """
    Compute the implied volatility of a European Option
    S0: initial stock price
    K:  strike price
    T:  maturity
    r:  risk-free rate
    market_price: market observed price
    tol: user choosen tolerance
    """
    max_iter = 20 #max number of iterations
    vol_old = 0.30 #initial guess

    for k in range(max_iter):
        
        bs_price = blackScholes(flag, S0, K, T, r, vol_old)
        Cprime =  vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        bs_new = blackScholes(flag, S0, K, T, r, vol_new)

        if (abs(vol_old - vol_new) < tol or abs(bs_new - market_price) < tol):
            break
        vol_old = vol_new

    implied_vol = vol_old
    
    return implied_vol


def iv_binary_search(flag, S0, K, T, r, market_price):
    
    ONE_CENT = 0.01
    step = 0.001    
    _sigma = 0.5 # initial sigma guess
    
    for i in range(1000): #max number of calculations is 10000
        bs_price = blackScholes(flag, S0, K, T, r, σ=_sigma)
        diff = market_price - bs_price
        if diff > ONE_CENT:
            _sigma = _sigma + step
        elif diff < 0 and abs(diff) > ONE_CENT:
            _sigma = _sigma - step
        elif abs(diff) < ONE_CENT:
            return _sigma
    return _sigma
    

def get_iv(flag, S0, K, T, r, market_price):
    res = iv_newton_raphson(flag, S0, K, T, r, market_price, tol=0.01)
    
    return res if res is not None else iv_binary_search(flag, S0, K, T, r, market_price)


def find_sensitivities(data : pd.DataFrame, current_underlying : float):
    current_underlying = 20050

    data['IV'] = data.apply(
                            lambda row: get_iv(
                                            flag = row['instrument_name'][-1:],                            
                                            S0=current_underlying,
                                            K=row['strike'],
                                            T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                            r=0,
                                            market_price = row['last_price'] if row['last_price'] is not None else 0
                                            )
                            , axis=1
                                    )
    data['delta'] = data.apply(
                            lambda row: delta(
                                flag = row['instrument_name'][-1:],
                                S0=current_underlying,
                                K=row['strike'],
                                T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                r=0,
                                σ = row['IV']
                            ), axis=1
    )


    data['gamma'] = data.apply(
                            lambda row: gamma(
                                flag = row['instrument_name'][-1:],
                                S0=current_underlying,
                                K=row['strike'],
                                T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                r=0,
                                σ = row['IV']
                            ), axis=1
    )

    data['vega'] = data.apply(
                            lambda row: vega(
                                flag = row['instrument_name'][-1:],
                                S0=current_underlying,
                                K=row['strike'],
                                T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                r=0,
                                σ = row['IV']
                            ), axis=1
    )

    data['theta'] = data.apply(
                            lambda row: theta(
                                flag = row['instrument_name'][-1:],
                                S0=current_underlying,
                                K=row['strike'],
                                T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                r=0,
                                σ = row['IV']
                            ), axis=1
    )

    data['rho'] = data.apply(
                            lambda row: rho(
                                flag = row['instrument_name'][-1:],
                                S0=current_underlying,
                                K=row['strike'],
                                T=(row['expiration_time'] - datetime.datetime.now()).days / 365,            
                                r=0,
                                σ = row['IV']
                            ), axis=1
    )    

    return data