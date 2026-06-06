"""
Quick smoke test — verify Python environment works
and we can pull live market data from Yahoo Finance.
"""

import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd

print("=" * 50)
print("Quant pricer setup verification")
print("=" * 50)

# Verify libraries
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"yfinance loaded successfully")

# Pull live market data — AAPL
print("\nFetching AAPL data...")
aapl = yf.Ticker("AAPL")
info = aapl.history(period="5d")

print(f"\nAAPL last 5 days of closing prices:")
print(info["Close"].round(2))

# Quick math check — normal distribution
print(f"\nN(0) = {norm.cdf(0):.4f} (should be 0.5)")
print(f"N(1.96) = {norm.cdf(1.96):.4f} (should be ~0.975)")

print("\n✅ Environment fully working — ready to build Black-Scholes")
