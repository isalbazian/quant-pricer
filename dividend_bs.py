"""
Black-Scholes with Continuous Dividend Yield (Merton, 1973)
=============================================================
Extends Black-Scholes to handle dividend-paying stocks.

The key insight: a stock paying continuous dividend yield q
is equivalent to a non-dividend stock with price S*exp(-q*T).
The dividend reduces the effective stock price because holders
know the stock will drop by the dividend amount on ex-date.

Effect on options:
    - Dividends REDUCE call value (stock drops on ex-dividend)
    - Dividends INCREASE put value (same reason, reversed)
    - Deep ITM calls on high-dividend stocks can be worth less
      than their intrinsic value — early exercise may be optimal

This is the version used for index options (S&P 500, etc.)
where dividends are roughly continuous across all components.

Reference: Merton, R.C. (1973) "Theory of Rational Option Pricing"
           Hull, Chapter 17 — "Options on Stock Indices and Currencies"
"""

import numpy as np
from scipy.stats import norm


def d1_d2_div(S, K, T, r, sigma, q):
    """
    Modified d1 and d2 with continuous dividend yield q.
    
    The only change from plain BS: (r + sigma^2/2) becomes (r - q + sigma^2/2)
    The dividend yield reduces the drift of the stock.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price_div(S, K, T, r, sigma, q=0.0):
    """
    European call price with continuous dividend yield.
    
    Formula: C = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
    
    When q=0, reduces exactly to standard Black-Scholes.
    The exp(-q*T) term discounts the stock price by the
    present value of dividends paid over the option's life.
    """
    d1, d2 = d1_d2_div(S, K, T, r, sigma, q)
    price = (S * np.exp(-q * T) * norm.cdf(d1) 
             - K * np.exp(-r * T) * norm.cdf(d2))
    return price


def put_price_div(S, K, T, r, sigma, q=0.0):
    """
    European put price with continuous dividend yield.
    
    Formula: P = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
    """
    d1, d2 = d1_d2_div(S, K, T, r, sigma, q)
    price = (K * np.exp(-r * T) * norm.cdf(-d2) 
             - S * np.exp(-q * T) * norm.cdf(-d1))
    return price


def delta_call_div(S, K, T, r, sigma, q=0.0):
    """Delta for call with dividends: exp(-q*T) * N(d1)"""
    d1, _ = d1_d2_div(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d1)


def delta_put_div(S, K, T, r, sigma, q=0.0):
    """Delta for put with dividends: exp(-q*T) * (N(d1) - 1)"""
    d1, _ = d1_d2_div(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def get_dividend_yield(ticker_symbol):
    """
    Fetch the trailing dividend yield from Yahoo Finance.
    Returns as a decimal (e.g. 0.005 for 0.5%).
    """
    import yfinance as yf
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    div_yield = info.get("dividendYield", 0.0) or 0.0
    return div_yield


if __name__ == "__main__":
    import yfinance as yf

    print("=" * 65)
    print("Black-Scholes with Dividends — AAPL Pricing Comparison")
    print("=" * 65)

    # Fetch live AAPL data
    ticker = yf.Ticker("AAPL")
    S = ticker.history(period="1d")["Close"].iloc[-1]
    
    # Get dividend yield
    info = ticker.info
    # trailingAnnualDividendYield is the correct field (e.g. 0.0035 = 0.35%)
    # dividendYield returns value already in percent form (0.37 = 0.37%), confusing
    q = info.get('trailingAnnualDividendYield', 0.0) or 0.0
    
    # Risk-free rate
    tnx = yf.Ticker("^TNX")
    r = tnx.history(period="1d")["Close"].iloc[-1] / 100

    # Use same params as our implied vol analysis
    K = round(S / 5) * 5  # nearest $5 strike
    T = 0.5               # 6 months
    sigma = 0.235         # ATM implied vol

    print(f"\nAAPL price:        ${S:.2f}")
    print(f"Strike (ATM):      ${K:.2f}")
    print(f"Time (T):          {T} years")
    print(f"Risk-free (r):     {r*100:.2f}%")
    print(f"Volatility:        {sigma*100:.1f}%")
    print(f"Dividend yield:    {q*100:.2f}%")

    print(f"\n{'Method':<35} {'Call':>10} {'Put':>10} {'Call Delta':>12}")
    print("-" * 70)

    # Without dividends
    from black_scholes import call_price, put_price, delta_call
    c_nodiv = call_price(S, K, T, r, sigma)
    p_nodiv = put_price(S, K, T, r, sigma)
    d_nodiv = delta_call(S, K, T, r, sigma)
    print(f"{'Plain BS (no dividends)':<35} {c_nodiv:>10.4f} {p_nodiv:>10.4f} {d_nodiv:>12.4f}")

    # With dividends
    c_div = call_price_div(S, K, T, r, sigma, q)
    p_div = put_price_div(S, K, T, r, sigma, q)
    d_div = delta_call_div(S, K, T, r, sigma, q)
    print(f"{'Merton (with dividends)':<35} {c_div:>10.4f} {p_div:>10.4f} {d_div:>12.4f}")

    # Difference
    call_diff = c_nodiv - c_div
    put_diff  = p_div - p_nodiv
    print(f"\nDividend impact on call: -${call_diff:.4f} "
          f"({'as expected — dividends reduce call value' if call_diff > 0 else 'check data'})")
    print(f"Dividend impact on put:  +${put_diff:.4f} "
          f"({'as expected — dividends increase put value' if put_diff > 0 else 'check data'})")

    print(f"\nNOTE: AAPL's dividend yield is {q*100:.2f}%.")
    if q < 0.005:
        print("Very low dividend — the adjustment is small but non-zero.")
        print("For high-dividend stocks (utilities, REITs), this matters a lot.")
    else:
        print("This adjustment is material for accurate pricing.")

    print("\n" + "=" * 65)
    print("Merton dividend model: identical to BS when q=0.")
    print("For index options (SPX), q = index dividend yield ~1.3%.")
    print("For currency options, q = foreign risk-free rate.")
    print("Same formula, different interpretation of q.")
    print("=" * 65)
    