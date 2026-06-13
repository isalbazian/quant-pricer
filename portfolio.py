"""
Portfolio Greeks Aggregation
==============================
Real options traders don't hold one option — they hold portfolios.
This module aggregates Greeks across multiple positions to give
net risk exposure: total Delta, Gamma, Vega, Theta, Rho.

Key insight: Greeks are additive across positions (with sign).
    - Long call: positive Delta, positive Gamma, positive Vega
    - Long put: negative Delta, positive Gamma, positive Vega
    - Short anything: flip all signs

Example use case:
    A trader sells 10 AAPL ATM calls (collects premium, negative Gamma)
    and buys 5 OTM calls to cap upside risk.
    Net Greeks tell them: what is my actual exposure right now?

Reference: Hull, Chapter 19 — "The Greek Letters"
"""

import numpy as np
from black_scholes import call_price, put_price, all_greeks


class OptionPosition:
    """
    A single option position in a portfolio.
    
    Parameters:
        S          : current stock price
        K          : strike price
        T          : time to expiration in years
        r          : risk-free rate
        sigma      : volatility
        option_type: "call" or "put"
        quantity   : number of contracts (negative = short)
        multiplier : shares per contract (standard = 100)
    """

    def __init__(self, S, K, T, r, sigma, option_type="call",
                 quantity=1, multiplier=100, label=None):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.quantity = quantity      # negative = short
        self.multiplier = multiplier  # 100 shares per contract
        self.label = label or f"{option_type.upper()} K={K}"

        # Compute Greeks for one contract
        self._greeks = all_greeks(S, K, T, r, sigma, option_type)

    @property
    def price(self):
        return self._greeks["price"]

    @property
    def delta(self):
        """Net Delta: Greek * quantity * multiplier"""
        return self._greeks["delta"] * self.quantity * self.multiplier

    @property
    def gamma(self):
        return self._greeks["gamma"] * self.quantity * self.multiplier

    @property
    def vega(self):
        return self._greeks["vega"] * self.quantity * self.multiplier

    @property
    def theta(self):
        return self._greeks["theta"] * self.quantity * self.multiplier

    @property
    def rho(self):
        return self._greeks["rho"] * self.quantity * self.multiplier

    @property
    def market_value(self):
        """Total market value of the position"""
        return self.price * self.quantity * self.multiplier

    def summary(self):
        direction = "LONG" if self.quantity > 0 else "SHORT"
        return (f"{direction} {abs(self.quantity)}x {self.label} "
                f"@ ${self.price:.2f} | "
                f"Delta={self.delta:+.1f} "
                f"Gamma={self.gamma:+.4f} "
                f"Vega={self.vega:+.2f} "
                f"Theta={self.theta:+.2f}")


class OptionsPortfolio:
    """
    A portfolio of option positions with aggregated Greeks.
    
    Aggregation rules:
        - Delta: sum across all positions. Net Delta tells you
          how many shares of stock you effectively hold.
        - Gamma: sum. Positive Gamma = you benefit from big moves.
          Negative Gamma = you're hurt by big moves (seller's curse).
        - Vega: sum. Net Vega tells you if you profit or lose
          when implied vol changes.
        - Theta: sum. Almost always negative for long options —
          you pay time decay. Positive Theta = you're a net seller.
    """

    def __init__(self, name="Portfolio"):
        self.name = name
        self.positions = []

    def add_position(self, position):
        self.positions.append(position)
        return self

    def net_delta(self):
        return sum(p.delta for p in self.positions)

    def net_gamma(self):
        return sum(p.gamma for p in self.positions)

    def net_vega(self):
        return sum(p.vega for p in self.positions)

    def net_theta(self):
        return sum(p.theta for p in self.positions)

    def net_rho(self):
        return sum(p.rho for p in self.positions)

    def total_value(self):
        return sum(p.market_value for p in self.positions)

    def delta_dollars(self, S):
        """
        Dollar Delta: how much does portfolio value change per $1 in S?
        Also called 'dollar delta' or 'delta exposure'.
        """
        return self.net_delta()

    def pnl_estimate(self, dS=0, dsigma=0, dt_days=0):
        """
        Estimate P&L for small moves using first-order Greeks.
        
        This is the 'Taylor expansion' approach traders use for
        quick P&L estimation without repricing:
        
        dP ≈ Delta*dS + 0.5*Gamma*dS^2 + Vega*dsigma + Theta*dt
        
        Parameters:
            dS       : stock price change in dollars
            dsigma   : vol change in percentage points (e.g. 1 = 1%)
            dt_days  : days elapsed
        """
        delta_pnl = self.net_delta() * dS
        gamma_pnl = 0.5 * self.net_gamma() * dS**2
        vega_pnl  = self.net_vega() * dsigma
        theta_pnl = self.net_theta() * dt_days

        total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        return {
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl":  vega_pnl,
            "theta_pnl": theta_pnl,
            "total_pnl": total
        }

    def report(self):
        print(f"\n{'='*65}")
        print(f"PORTFOLIO: {self.name}")
        print(f"{'='*65}")
        print(f"{'Position':<40} {'Value':>10}")
        print(f"{'-'*65}")

        for p in self.positions:
            print(f"  {p.summary()}")

        print(f"\n{'-'*65}")
        print(f"AGGREGATED RISK (Net Greeks)")
        print(f"{'-'*65}")
        print(f"  Net Delta:  {self.net_delta():>+10.2f}  "
              f"(≈ holding {self.net_delta():.0f} shares)")
        print(f"  Net Gamma:  {self.net_gamma():>+10.4f}  "
              f"({'benefits' if self.net_gamma() > 0 else 'hurt by'} large moves)")
        print(f"  Net Vega:   {self.net_vega():>+10.2f}  "
              f"(per 1% vol change)")
        print(f"  Net Theta:  {self.net_theta():>+10.2f}  "
              f"(per day)")
        print(f"  Net Rho:    {self.net_rho():>+10.2f}  "
              f"(per 1% rate change)")
        print(f"\n  Total MV:   ${self.total_value():>+10.2f}")
        print(f"{'='*65}")


if __name__ == "__main__":
    # Real-world example: a common options strategy
    # COVERED CALL + PROTECTIVE PUT on AAPL

    S = 291.13   # AAPL current price
    r = 0.0449   # risk-free rate
    T = 0.074    # ~27 days to expiry
    sigma = 0.235  # ATM implied vol from our solver

    print("Building a realistic AAPL options portfolio...")
    print(f"AAPL @ ${S}, T={T:.3f}yr, r={r*100:.2f}%, sigma={sigma*100:.1f}%")

    portfolio = OptionsPortfolio("AAPL Hedged Position")

    # Position 1: Long 100 shares (represented as delta=1 per share)
    # We'll simulate this by adding a deep ITM call as a proxy
    # Actually let's build a real multi-option strategy:

    # STRATEGY: SHORT STRANGLE
    # Sell OTM call + sell OTM put. Collect premium, profit if stock stays flat.
    # Classic income strategy used by sophisticated investors.

    # Leg 1: Short 2 OTM calls (strike $310, ~6% OTM)
    short_call = OptionPosition(
        S=S, K=310, T=T, r=r, sigma=sigma,
        option_type="call", quantity=-2,
        label="AAPL 310 Call (Jul)"
    )

    # Leg 2: Short 2 OTM puts (strike $270, ~7% OTM)
    short_put = OptionPosition(
        S=S, K=270, T=T, r=r, sigma=sigma,
        option_type="put", quantity=-2,
        label="AAPL 270 Put (Jul)"
    )

    # Leg 3: Long 1 far OTM call as hedge (strike $320)
    hedge_call = OptionPosition(
        S=S, K=320, T=T, r=r, sigma=sigma,
        option_type="call", quantity=1,
        label="AAPL 320 Call hedge (Jul)"
    )

    portfolio.add_position(short_call)
    portfolio.add_position(short_put)
    portfolio.add_position(hedge_call)

    portfolio.report()

    # P&L scenarios
    print("\nP&L SCENARIO ANALYSIS")
    print("-" * 50)
    scenarios = [
        {"dS": 0,   "dsigma": 0,    "dt_days": 1,  "desc": "1 day passes, stock flat"},
        {"dS": 5,   "dsigma": 0,    "dt_days": 0,  "desc": "AAPL up $5 today"},
        {"dS": -5,  "dsigma": 0,    "dt_days": 0,  "desc": "AAPL down $5 today"},
        {"dS": 0,   "dsigma": 2,    "dt_days": 0,  "desc": "Vol spikes +2% (fear)"},
        {"dS": 0,   "dsigma": -2,   "dt_days": 0,  "desc": "Vol drops -2% (calm)"},
        {"dS": -15, "dsigma": 5,    "dt_days": 0,  "desc": "AAPL crash -$15, vol +5%"},
    ]

    for s in scenarios:
        pnl = portfolio.pnl_estimate(
            dS=s["dS"], dsigma=s["dsigma"], dt_days=s["dt_days"]
        )
        sign = "+" if pnl["total_pnl"] >= 0 else ""
        print(f"  {s['desc']:<40} ${sign}{pnl['total_pnl']:>7.2f}")

    print()
    print("NOTE: Short strangle collects Theta daily (positive for seller)")
    print("but is hurt by Gamma (big moves kill you) and Vega spikes.")
    print("This is the core tradeoff every options market maker manages.")