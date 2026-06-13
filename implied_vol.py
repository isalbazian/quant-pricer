"""
Implied Volatility Solver
==========================
Inverts the Black-Scholes formula to find the volatility (sigma)
that makes the BS price equal to the observed market price.

This is the most important calculation in options trading.
Every quoted option has an "implied vol" — the market's consensus
forecast of future realized volatility embedded in the price.

Method: Bisection search (binary search over sigma space)
    - We know sigma must be between 0 and some upper bound
    - Test midpoint: if BS price too high, sigma is too high
    - Cut the interval in half and repeat
    - Converges in ~50 iterations to machine precision

Reference: Hull, Chapter 20 — "Volatility Smiles"
"""

import numpy as np
from black_scholes import call_price, put_price


def implied_vol_call(S, K, T, r, market_price,
                     sigma_low=1e-6, sigma_high=5.0,
                     tol=1e-6, max_iter=1000):
    """
    Find the implied volatility for a European call option.

    Parameters:
        S            : current stock price
        K            : strike price
        T            : time to expiration in years
        r            : risk-free rate
        market_price : observed market price of the call
        sigma_low    : lower bound for vol search (near zero)
        sigma_high   : upper bound for vol search (500%)
        tol          : convergence tolerance
        max_iter     : maximum iterations before giving up

    Returns:
        Implied volatility as a decimal (e.g. 0.25 for 25%)
        or None if no solution found (deep ITM/OTM options
        can have no valid IV due to arbitrage bounds)
    """
    # Quick sanity check: is the market price within valid BS range?
    # Call price must be > max(S - K*exp(-rT), 0) and < S
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if market_price <= intrinsic:
        return None  # Price below intrinsic — arbitrage or bad data
    if market_price >= S:
        return None  # Price above stock — impossible for a call

    # Bisection search
    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = call_price(S, K, T, r, sigma_mid)

        if abs(price_mid - market_price) < tol:
            return sigma_mid  # Converged

        if price_mid < market_price:
            sigma_low = sigma_mid   # Need higher vol
        else:
            sigma_high = sigma_mid  # Need lower vol

    return (sigma_low + sigma_high) / 2  # Best estimate after max_iter


def implied_vol_put(S, K, T, r, market_price,
                    sigma_low=1e-6, sigma_high=5.0,
                    tol=1e-6, max_iter=1000):
    """
    Find the implied volatility for a European put option.
    Same algorithm as the call solver but uses put_price().
    """
    intrinsic = max(K * np.exp(-r * T) - S, 0)
    if market_price <= intrinsic:
        return None
    if market_price >= K:
        return None

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = put_price(S, K, T, r, sigma_mid)

        if abs(price_mid - market_price) < tol:
            return sigma_mid

        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return (sigma_low + sigma_high) / 2


def iv_from_chain(chain_df, S, T, r, option_type="call"):
    """
    Compute implied vol for an entire options chain DataFrame.

    Parameters:
        chain_df    : pandas DataFrame with columns 'strike' and 'lastPrice'
        S           : current stock price
        T           : time to expiration in years
        r           : risk-free rate
        option_type : "call" or "put"

    Returns:
        DataFrame with added 'impliedVol' and 'moneyness' columns
    """
    import pandas as pd

    df = chain_df.copy()
    ivs = []

    solver = implied_vol_call if option_type == "call" else implied_vol_put

    for _, row in df.iterrows():
        iv = solver(S, row["strike"], T, r, row["lastPrice"])
        ivs.append(iv)

    df["impliedVol"] = ivs
    df["moneyness"] = df["strike"] / S  # 1.0 = ATM, >1 = OTM call, <1 = ITM call

    return df


if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime

    print("=" * 60)
    print("Implied Volatility Solver — Validation + Live AAPL IV")
    print("=" * 60)

    # PART 1: Validate the solver works
    print("\n1. SOLVER VALIDATION")
    print("-" * 40)
    S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.25

    # Compute a BS price at known vol
    from black_scholes import call_price
    known_price = call_price(S, K, T, r, true_sigma)
    print(f"Known sigma:      {true_sigma:.4f} (25%)")
    print(f"BS price at 25%:  ${known_price:.4f}")

    # Now recover sigma from the price
    recovered = implied_vol_call(S, K, T, r, known_price)
    print(f"Recovered sigma:  {recovered:.6f}")
    print(f"Error:            {abs(recovered - true_sigma):.2e}")
    print("✅ Solver inverts Black-Scholes correctly" if abs(recovered - true_sigma) < 1e-4 else "❌ Check solver")

    # PART 2: Live AAPL implied vol across strikes
    print("\n2. LIVE AAPL IMPLIED VOLATILITY BY STRIKE")
    print("-" * 40)

    ticker = yf.Ticker("AAPL")
    S = ticker.history(period="1d")["Close"].iloc[-1]
    expirations = ticker.options
    today = datetime.now().date()

    # Pick first expiration 20-45 days out
    target_exp = None
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        days = (exp_date - today).days
        if 20 <= days <= 45:
            target_exp = exp_str
            T = days / 365
            break

    if not target_exp:
        target_exp = expirations[1]
        T = (datetime.strptime(target_exp, "%Y-%m-%d").date() - today).days / 365

    # Risk-free rate
    tnx = yf.Ticker("^TNX")
    r = tnx.history(period="1d")["Close"].iloc[-1] / 100

    print(f"AAPL price:    ${S:.2f}")
    print(f"Expiration:    {target_exp} (T={T:.4f} years)")
    print(f"Risk-free:     {r*100:.2f}%")
    print()

    chain = ticker.option_chain(target_exp)
    calls = chain.calls.copy()

    # Filter to strikes within 15% of current price
    calls = calls[
        (calls["strike"] >= S * 0.85) &
        (calls["strike"] <= S * 1.15) &
        (calls["lastPrice"] > 0.10)
    ].copy()

    result = iv_from_chain(calls, S, T, r, option_type="call")
    result = result.dropna(subset=["impliedVol"])

    print(f"{'Strike':>8} {'Moneyness':>10} {'Mkt Price':>10} {'Impl Vol':>10}")
    print("-" * 45)
    for _, row in result.iterrows():
        flag = " ← ATM" if abs(row["moneyness"] - 1.0) < 0.02 else ""
        print(f"${row['strike']:>7.1f} {row['moneyness']:>10.3f} "
              f"${row['lastPrice']:>9.2f} {row['impliedVol']*100:>9.2f}%{flag}")

    print()
    atm = result.iloc[(result["moneyness"] - 1.0).abs().argsort()[:1]]
    print(f"ATM implied vol: {atm['impliedVol'].values[0]*100:.2f}%")
    print()
    print("NOTE: If IV changes with strike (smile/skew), that's the")
    print("'volatility smile' — a real market phenomenon Black-Scholes")
    print("can't explain. It's why this research matters.")
    