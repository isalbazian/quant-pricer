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


if __name__ == "__main__":
    # Standard textbook example to validate against published values
    # Hull, Chapter 15, Example 15.6: AAPL-style example
    
    S = 100      # Stock at $100
    K = 100      # Strike at $100 (at the money)
    T = 1.0      # 1 year to expiration
    r = 0.05     # 5% risk-free rate
    sigma = 0.20 # 20% volatility
    
    print("=" * 50)
    print("Black-Scholes Pricer — Smoke Test")
    print("=" * 50)
    print(f"Stock price (S):      ${S}")
    print(f"Strike price (K):     ${K}")
    print(f"Time to expiry (T):   {T} years")
    print(f"Risk-free rate (r):   {r*100}%")
    print(f"Volatility (sigma):   {sigma*100}%")
    print("-" * 50)
    
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    
    print(f"Call price:           ${call:.4f}")
    print(f"Put price:            ${put:.4f}")
    print("-" * 50)
    print("Expected (Hull textbook):")
    print(f"  Call: ~$10.45")
    print(f"  Put:  ~$5.57")