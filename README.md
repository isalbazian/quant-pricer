# Quant Pricer

Black-Scholes options pricing in Python with live market data integration.

## What This Does

- Implements the Black-Scholes-Merton formula for European call and put options
- Computes all five Greeks (Delta, Gamma, Vega, Theta, Rho) for risk analysis
- Pulls live market data (stock prices, options chains, Treasury yields) via yfinance
- Estimates historical volatility from daily returns
- Compares model prices against live market quotes
- Validated against textbook reference values (Hull, Chapter 15)

## Quick Start

    git clone https://github.com/isalbazian/quant-pricer.git
    cd quant-pricer
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python3 black_scholes.py    # textbook validation + Greeks
    python3 price_aapl.py       # live AAPL pricing

## Files

| File | Description |
|------|-------------|
| `black_scholes.py` | Core pricing functions + all Greeks |
| `price_aapl.py` | Live AAPL options pricing using real market data |
| `test_setup.py` | Environment verification |
| `requirements.txt` | Pinned dependencies |

## Sample Output

    Pricing a real AAPL option with Black-Scholes
    ============================================================
    Current AAPL price (S):  $307.34
    Expiration date:        2026-07-10 (34 days)
    Strike price (K):       $305.00
    Historical volatility:  22.28%
    Market implied vol:     27.60%
    Risk-free rate (r):     4.54%
    ------------------------------------------------------------
    Our BS price (market IV):      $12.18
    Actual market price:           $11.94
    Difference: +2.00%

    GREEKS — Call Option (ATM, 1yr, sigma=20%)
      Delta:    +0.6368  (per $1 move in stock)
      Gamma:    +0.0188  (rate of change of Delta)
      Vega:     +0.3752  (per 1% change in vol)
      Theta:    -0.0176  (per day decay)
      Rho:      +0.5323  (per 1% change in rates)

## Roadmap

- [x] Black-Scholes European call/put
- [x] Live market data integration
- [x] Historical volatility estimation
- [x] Greeks (Delta, Gamma, Vega, Theta, Rho)
- [x] Monte Carlo for path-dependent options
- [ ] Dividend adjustments
- [x] Implied volatility solver
- [ ] Portfolio Greeks aggregation

## References

- Hull, J. C. *Options, Futures, and Other Derivatives* (10th ed., Chapter 15)
- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Merton, R. C. (1973). "Theory of Rational Option Pricing"

## License

MIT
