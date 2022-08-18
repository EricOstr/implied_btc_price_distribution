from datetime import datetime
import numpy as np
import pandas as pd
from sensitivities import delta


def predict_mc_model(flag, S0, K, vol, expr, r=0.0, N=10, M=1000):

    '''
    Find the value of an option using Monte Carlo simulation with a control variate

    '''

    if vol is None: return [], 0, 0

    if type(expr) == str: expr = datetime.strptime(expr, '%Y-%m-%dT%H:%M:%S')
    T = (expr - datetime.now()).total_seconds() / (365*24*60*60)
    dt = T/N
    nudt = (r - 0.5*vol**2)*dt
    volsdt = vol*np.sqrt(dt)
    erdt = np.exp(r*dt)
    
    CT_res = np.empty(M)

    # Monte Carlo Method
    for i in range(M):
        St = S0 
        cv = 0
        for j in range(N):
            epsilon = np.random.normal()
            deltaSt = delta(flag, St, K, T-j*dt, r, vol)
            Stn = St*np.exp( nudt + volsdt*epsilon )
            cv = cv + deltaSt*(Stn - St*erdt)
            St = Stn
        
        CT_res[i] = (max(0, St - K) - cv ) if flag=='C' else (max(0, K-St) - cv)

    # Compute Expectation and SE
    C0 = np.exp(-r*T)*np.mean(CT_res)
    sigma = np.std(CT_res)
    SE = sigma/np.sqrt(M)
    
    return  CT_res, np.round(C0, 3), np.round(SE,3)