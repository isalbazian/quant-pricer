"""
Price a real AAPL option using live market data + Black-Scholes.
"""

import yfinance as yf
import numpy as np
from datetime import datetime
from black_scholes import call_price, put_price


def get_historical_volatility(ticker_symbol, lookback_days=252):
    """Estimate annualized volatility from historical daily returns."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=f"{lookback_days}d")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    daily_vol = log_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
    return annualized_vol


def get_risk_free_rate():
    """Use the 10-year Treasury yield as the risk-free rate proxy."""
    tnx = yf.Ticker("^TNX")
    current_yield_pct = tnx.history(period="1d")["Close"].iloc[-1]
    return current_yield_pct / 100


def price_real_option(ticker_symbol="AAPL"):
    """Pull real market data and price an at-the-money option."""
    
    print("=" * 60)
    print(f"Pricing a real {ticker_symbol} option with Black-Scholes")
    print("=" * 60)
    
    ticker = yf.Ticker(ticker_symbol)
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    S = round(current_price, 2)
    print(f"Current {ticker_symbol} price (S):  ${S}")
    
    expirations = ticker.options
    if not expirations:
        print("No options available for this ticker")
        return
    
    today = datetime.now().date()
    target_exp = None
    target_days = None
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_out = (exp_date - today).days
        if days_out >= 30:
            target_exp = exp_str
            target_days = days_out
            break
    
    if not target_exp:
        target_exp = expirations[-1]
        target_days = (datetime.strptime(target_exp, "%Y-%m-%d").date() - today).days
    
    T = target_days / 365
    print(f"Expiration date:        {target_exp} ({target_days} days)")
    print(f"Time to expiry (T):     {T:.4f} years")
    
    chain = ticker.option_chain(target_exp)
    calls = chain.calls
    calls = calls.copy()
    calls["dist"] = abs(calls["strike"] - S)
    atm_row = calls.loc[calls["dist"].idxmin()]
    K = atm_row["strike"]
    market_price = atm_row["lastPrice"]
    market_iv = atm_row["impliedVolatility"]
    
    print(f"Strike price (K):       ${K}")
    
    sigma_hist = get_historical_volatility(ticker_symbol)
    print(f"Historical volatility:  {sigma_hist*100:.2f}%")
    print(f"Market implied vol:     {market_iv*100:.2f}%")
    
    r = get_risk_free_rate()
    print(f"Risk-free rate (r):     {r*100:.2f}%")
    
    print("-" * 60)
    
    our_call_hist = call_price(S, K, T, r, sigma_hist)
    our_call_iv = call_price(S, K, T, r, market_iv)
    
    print(f"\nOur BS price (historical vol): ${our_call_hist:.2f}")
    print(f"Our BS price (market IV):      ${our_call_iv:.2f}")
    print(f"Actual market price:           ${market_price:.2f}")
    print()
    
    if market_price > 0:
        diff_pct = ((our_call_iv - market_price) / market_price) * 100
        print(f"Difference (using market IV): {diff_pct:+.2f}%")
    print()
    print("Note: Using market's implied vol should match market price exactly,")
    print("because IV is the vol that MAKES Black-Scholes equal market price.")


if __name__ == "__main__":
    price_real_option("AAPL")
