import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

from sensitivities import get_iv


def get_gradient(max_val, min_val, n):
    first_half = np.linspace(min_val, max_val, (n//2)+1)
    grad = list(first_half) + list(first_half[::-1])[1:]
    return grad


def get_options(options_data: pd.DataFrame,
                expr : np.datetime64,
                S0: float,
                T: float,
                kind: str
                ):
    """
    Cleans options data and calculates implied volatility for each option.
    Calculates implied volatility using Black Scholes.
    We use Newton-Raphson methods as the root finding algorithm - if this does not work, resort to Binary Search
    """

    options_mask = (
        (options_data['expiration_time'] == expr)
        & ((options_data['best_ask_price'] != 0) | (options_data['best_bid_price'] != 0) | (options_data['last_price'] != 0))
        & (options_data['instrument_name'].apply(lambda x: x[-1:] == kind))
        & options_data['volume'].apply(lambda x: x is not None)
    )

    options = options_data[options_mask].sort_values(['strike'])

    options['midprice'] = options.apply(
        lambda x: float((x.best_bid_price + x.best_ask_price) / 2)
        if (x.best_bid_price != 0 and x.best_ask_price != 0)
        else x.last_price,
        axis=1
    )
    options.strike = options.strike.apply(lambda x: float(x))
    options = options.reset_index(drop=True)
    options["iv"] = options.apply(
        lambda row: get_iv('C', S0=S0, K=row.strike, T=T, r=0.0, market_price=row.midprice), axis=1)

    return options[['strike', 'expiration_time', 'midprice', 'volume', 'open_interest']]


def get_bflys(options_data: pd.DataFrame):
    '''
    Calculate implied probability of strike prices using price of the butterfly vs max profit potential
    '''
    bf_data = []

    for (_, left), (_, centre), (_, right) in zip(options_data.iterrows(),
                                                  options_data.iloc[1:].iterrows(),
                                                  options_data.iloc[2:].iterrows()):

        if centre.strike - left.strike != right.strike - centre.strike:
            continue
        butterfly_price = left.midprice - 2 * centre.midprice + right.midprice
        max_profit = centre.strike - left.strike
        bf_data.append([centre.strike, butterfly_price, max_profit])

    bflys_data = pd.DataFrame(
        bf_data, columns=["strike", "price", "max_profit"])
    bflys_data = bflys_data.dropna()
    bflys_data["prob"] = bflys_data.price / bflys_data.max_profit
    bflys_data = bflys_data[(abs(bflys_data.prob) < 1) & (bflys_data.prob > 0)]
    bflys_data = bflys_data.reset_index(drop=True)

    return bflys_data[['strike', 'prob']]


def inverse_CDF(CDF: scipy.interpolate._cubic.CubicSpline,
                mi: float,
                ma: float,
                pct: float):

    domain = np.linspace(mi, ma, 1000)
    for x in domain:
        if CDF(x) > pct:
            return x

    return ma


def get_implied_price_distribution(options_data: pd.DataFrame(),
                                   percentiles: np.array([]),
                                   S0 : float):

    unique_expirations = options_data['expiration_time'].unique()
    mask = ((unique_expirations.astype('datetime64[s]').astype(
        'int') - datetime.now().timestamp()) / (60*60*24*7)) < 4
    unique_expirations = unique_expirations[mask]
    unique_expirations.sort()

    res = pd.DataFrame(columns=['expiration_time'] +
                       [pct for pct in percentiles])

    for expr in unique_expirations:

        expr_dt = datetime.fromtimestamp(
            expr.astype('datetime64[s]').astype('int'))
        T = (expr_dt - datetime.now()
             ).total_seconds() / (365*24*60*60)

        # Clean data and calculate IV
        calls = get_options(options_data, expr, T, S0, 'C')
        puts = get_options(options_data, expr, T, S0, 'P')

        # Calculate implied probability of various strike prices using call and put data
        bflys_calls = get_bflys(calls)
        bflys_puts = get_bflys(puts)
        bflys_combined = pd.concat(
            [bflys_calls, bflys_puts], axis=0, sort=False)
        if len(bflys_combined) == 0:
            continue

        # Combine data points for each strike price and apply gaussian filter
        bflys_combined = bflys_combined.sort_values(
            ['strike']).groupby(['strike'], as_index=False).mean()
        bflys_combined['prob'] = gaussian_filter1d(bflys_combined.prob, 2)

        # Interpolate implied price distribution using Splines
        # use bc_type='clamped' so first derivative at curve ends are 0 (conforms to notion of PDF)
        cubic_spline = scipy.interpolate.CubicSpline(
            bflys_combined.strike,
            bflys_combined.prob,
            axis=0,
            bc_type='clamped')

        mi = bflys_combined.strike.min()
        ma = bflys_combined.strike.max()

        # Find area under the curve by integrating between min and max strike price
        # The auc figure is later used to normalize the probabilities
        auc = cubic_spline.integrate(mi, ma)
        cubic_spline_mod = scipy.interpolate.CubicSpline(
            bflys_combined.strike, bflys_combined.prob/auc, axis=0, bc_type='clamped')

        # Calculate anti-derivative and use this for: percentile -> strike price
        cubic_spline_mod_anti = cubic_spline_mod.antiderivative()
        percentile_prices = [inverse_CDF(
            cubic_spline_mod_anti, mi, ma, pct) for pct in percentiles]

        row = [expr] + percentile_prices
        res.loc[len(res)] = row

    # Apply final gaussian filter
    for col in res.columns[1:]:
        res[col] = gaussian_filter1d(res[col], 1)

    return res