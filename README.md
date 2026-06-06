# Quant Pricer

Black-Scholes options pricing in Python with live market data integration.

## What This Does

- Implements the Black-Scholes-Merton formula for European call and put options
- Pulls live market data (stock prices, options chains, Treasury yields) via yfinance
- Estimates historical volatility from daily returns
- Compares model prices against live market quotes
- Validated against textbook reference values (Hull, Chapter 15)

## Quick Start

```bash
git clone https://github.com/isalbazian/quant-pricer.git
cd quant-pricer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 black_scholes.py    # textbook validation
python3 price_aapl.py       # live AAPL pricing
```

## Files

| File | Description |
|------|-------------|
| `black_scholes.py` | Core pricing functions: call_price, put_price, d1_d2 |
| `price_aapl.py` | Live AAPL options pricing using real market data |
| `test_setup.py` | Environment verification |
| `requirements.txt` | Pinned dependencies |

## Sample Output