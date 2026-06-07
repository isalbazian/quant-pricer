"""
Black-Scholes Options Pricer
============================
Implements the Black-Scholes-Merton formula for pricing
European call and put options.

The formula:
    C = S * N(d1) - K * exp(-r*T) * N(d2)
    P = K * exp(-r*T) * N(-d2) - S * N(-d1)

where:
    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

Inputs:
    S      : current stock price
    K      : strike price
    T      : time to expiration (in years)
    r      : risk-free interest rate (decimal, e.g. 0.045 for 4.5%)
    sigma  : volatility (decimal, e.g. 0.25 for 25%)

Returns:
    Option price (premium)

Reference: Hull, "Options, Futures, and Other Derivatives", Chapter 15
"""

import numpy as np
from scipy.stats import norm

def d1_d2(S, K, T, r, sigma):
    """
    Calculate d1 and d2 — the two key intermediate values in Black-Scholes.
    
    Intuition:
        d1 measures "moneyness" adjusted for the drift and volatility
        d2 is d1 minus the volatility scaling term
        
    Both feed into the normal distribution N() to give us probabilities.
    
    Parameters:
        S     : current stock price
        K     : strike price
        T     : time to expiration in years
        r     : risk-free rate (decimal)
        sigma : volatility (decimal)
    
    Returns:
        tuple (d1, d2)
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def call_price(S, K, T, r, sigma):
    """
    Price a European call option using Black-Scholes.
    
    Formula:
        C = S * N(d1) - K * exp(-r*T) * N(d2)
    
    Intuition:
        S * N(d1)         = expected stock value if option ends ITM
        K * exp(-r*T) * N(d2) = present value of strike, weighted by probability of exercise
        
        Call value = "what you get" minus "what you pay" (in expected, discounted terms)
    
    Parameters:
        S     : current stock price
        K     : strike price
        T     : time to expiration in years
        r     : risk-free rate (decimal)
        sigma : volatility (decimal)
    
    Returns:
        Call option price (the premium)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def put_price(S, K, T, r, sigma):
    """
    Price a European put option using Black-Scholes.
    
    Formula:
        P = K * exp(-r*T) * N(-d2) - S * N(-d1)
    
    Intuition:
        K * exp(-r*T) * N(-d2) = expected strike receipt if option ends ITM (stock < strike)
        S * N(-d1)             = expected stock value given to seller
        
        Put value = "what you receive" minus "what you give up"
        
    Note: N(-x) = 1 - N(x), so these capture probability of stock ending BELOW strike.
    
    Parameters:
        S     : current stock price
        K     : strike price
        T     : time to expiration in years
        r     : risk-free rate (decimal)
        sigma : volatility (decimal)
    
    Returns:
        Put option price (the premium)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


# ============================================================
# THE GREEKS - sensitivities of option price to each input
# ============================================================

def delta_call(S, K, T, r, sigma):
    """
    Delta for a call option: dC/dS
    
    Measures how much the call price changes per $1 move in stock.
    Ranges 0 to 1. Deep ITM call has Delta near 1 (acts like owning stock).
    Deep OTM call has Delta near 0 (almost no exposure).
    ATM call has Delta around 0.5.
    
    Formula: Delta = N(d1)
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def delta_put(S, K, T, r, sigma):
    """
    Delta for a put option: dP/dS
    
    Ranges -1 to 0. Deep ITM put has Delta near -1 (inversely tracks stock).
    Deep OTM put has Delta near 0. ATM put has Delta around -0.5.
    
    Formula: Delta = N(d1) - 1
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """
    Gamma: d2C/dS2 (same for calls and puts)
    
    Rate of change of Delta. Highest for at-the-money options.
    High Gamma means Delta changes rapidly, so hedging needs to be adjusted often.
    
    Formula: Gamma = N'(d1) / (S * sigma * sqrt(T))
    where N'(x) is the standard normal PDF (density), not CDF.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega: dC/d(sigma) (same for calls and puts)
    
    Sensitivity to a 1.00 change in volatility.
    We divide by 100 so the output represents a 1% (0.01) change in vol,
    which matches how traders quote it.
    
    Formula: Vega = S * sqrt(T) * N'(d1)
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1) / 100  # per 1% vol change


def theta_call(S, K, T, r, sigma):
    """
    Theta for a call option: dC/dT (note: t = time elapsed, so we negate)
    
    Returns time decay PER DAY (divided by 365).
    Almost always negative — options lose value as time passes.
    
    Formula:
        Theta = -S*N'(d1)*sigma / (2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (term1 + term2) / 365  # per day


def theta_put(S, K, T, r, sigma):
    """
    Theta for a put option: per day.
    
    Formula:
        Theta = -S*N'(d1)*sigma / (2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365  # per day


def rho_call(S, K, T, r, sigma):
    """
    Rho for a call option: dC/dr
    
    Sensitivity to a 1% change in interest rate.
    Calls have positive Rho — higher rates = higher call value.
    
    Formula: Rho = K * T * exp(-r*T) * N(d2) / 100
    """
    _, d2 = d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) / 100


def rho_put(S, K, T, r, sigma):
    """
    Rho for a put option: dP/dr
    
    Puts have negative Rho — higher rates = lower put value.
    
    Formula: Rho = -K * T * exp(-r*T) * N(-d2) / 100
    """
    _, d2 = d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def all_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all Greeks at once and return as a dictionary.
    
    Parameters:
        option_type : "call" or "put"
    
    Returns:
        dict with keys: price, delta, gamma, vega, theta, rho
    """
    if option_type == "call":
        return {
            "price": call_price(S, K, T, r, sigma),
            "delta": delta_call(S, K, T, r, sigma),
            "gamma": gamma(S, K, T, r, sigma),
            "vega":  vega(S, K, T, r, sigma),
            "theta": theta_call(S, K, T, r, sigma),
            "rho":   rho_call(S, K, T, r, sigma),
        }
    elif option_type == "put":
        return {
            "price": put_price(S, K, T, r, sigma),
            "delta": delta_put(S, K, T, r, sigma),
            "gamma": gamma(S, K, T, r, sigma),
            "vega":  vega(S, K, T, r, sigma),
            "theta": theta_put(S, K, T, r, sigma),
            "rho":   rho_put(S, K, T, r, sigma),
        }
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    # Standard textbook example to validate against published values
    
    S = 100      # Stock at $100
    K = 100      # Strike at $100 (at the money)
    T = 1.0      # 1 year to expiration
    r = 0.05     # 5% risk-free rate
    sigma = 0.20 # 20% volatility
    
    print("=" * 60)
    print("Black-Scholes Pricer + Greeks — Smoke Test")
    print("=" * 60)
    print(f"Stock price (S):      ${S}")
    print(f"Strike price (K):     ${K}")
    print(f"Time to expiry (T):   {T} years")
    print(f"Risk-free rate (r):   {r*100}%")
    print(f"Volatility (sigma):   {sigma*100}%")
    print("-" * 60)
    
    # Prices
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    print(f"Call price:           ${call:.4f}")
    print(f"Put price:            ${put:.4f}")
    print("Expected (Hull):       Call ~$10.45, Put ~$5.57")
    print("-" * 60)
    
    # Greeks for the call
    print("\nGREEKS — Call Option")
    print(f"  Delta:    {delta_call(S, K, T, r, sigma):+.4f}  (per $1 move in stock)")
    print(f"  Gamma:    {gamma(S, K, T, r, sigma):+.4f}  (rate of change of Delta)")
    print(f"  Vega:     {vega(S, K, T, r, sigma):+.4f}  (per 1% change in vol)")
    print(f"  Theta:    {theta_call(S, K, T, r, sigma):+.4f}  (per day decay)")
    print(f"  Rho:      {rho_call(S, K, T, r, sigma):+.4f}  (per 1% change in rates)")
    
    print("\nGREEKS — Put Option")
    print(f"  Delta:    {delta_put(S, K, T, r, sigma):+.4f}")
    print(f"  Gamma:    {gamma(S, K, T, r, sigma):+.4f}")
    print(f"  Vega:     {vega(S, K, T, r, sigma):+.4f}")
    print(f"  Theta:    {theta_put(S, K, T, r, sigma):+.4f}")
    print(f"  Rho:      {rho_put(S, K, T, r, sigma):+.4f}")
    
    print("\n" + "=" * 60)
    print("Expected ATM call Greeks (Hull, approximate):")
    print("  Delta ~0.64, Gamma ~0.019, Vega ~0.376, Theta ~-0.018, Rho ~0.532")
    print("=" * 60)