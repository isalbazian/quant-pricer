"""
Monte Carlo Options Pricing
============================
Prices European call and put options using Monte Carlo simulation.

The stock is modeled as geometric Brownian motion (GBM):
    dS = r*S*dt + sigma*S*dW

Discretized:
    S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0, 1) is a standard normal random variable.

The option payoff is averaged across many simulated paths,
then discounted back to present value.

For European options, results should match Black-Scholes
(differences come from the finite number of simulations).

Reference: Hull, "Options, Futures, and Other Derivatives", Chapter 21
"""

import numpy as np
from black_scholes import call_price, put_price


def simulate_terminal_prices(S, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Simulate the stock price at expiration for many paths.
    
    Uses the "exact" GBM solution (one-step simulation to T).
    For European options, this is sufficient — we only care about S(T),
    not the path. For path-dependent options, use simulate_paths() instead.
    
    Parameters:
        S             : current stock price
        T             : time to expiration in years
        r             : risk-free rate
        sigma         : volatility
        n_simulations : number of paths to simulate
        seed          : optional random seed for reproducibility
    
    Returns:
        Array of simulated terminal stock prices (length n_simulations)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Draw n random normals
    Z = np.random.standard_normal(n_simulations)
    
    # Apply GBM formula in one step (since we only need terminal price)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_terminal = S * np.exp(drift + diffusion)
    
    return S_terminal


def mc_call_price(S, K, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Price a European call option via Monte Carlo simulation.
    
    Algorithm:
        1. Simulate n stock prices at expiration
        2. Compute payoff for each: max(S_T - K, 0)
        3. Average the payoffs
        4. Discount back to present: PV = mean * exp(-r*T)
    
    Returns:
        Estimated call price.
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_simulations, seed)
    payoffs = np.maximum(S_T - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


def mc_put_price(S, K, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Price a European put option via Monte Carlo simulation.
    
    Same algorithm as call but with payoff = max(K - S_T, 0).
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_simulations, seed)
    payoffs = np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


def mc_call_with_stderr(S, K, T, r, sigma, n_simulations=100_000, seed=None):
    """
    Price a European call with Monte Carlo AND return standard error.
    
    Standard error tells you how confident you should be in the estimate.
    Smaller SE = more reliable estimate. SE shrinks like 1/sqrt(n).
    
    Returns:
        tuple (price, standard_error, 95% confidence interval as tuple)
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_simulations, seed)
    payoffs = np.maximum(S_T - K, 0)
    discounted = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted)
    std_err = np.std(discounted, ddof=1) / np.sqrt(n_simulations)
    ci_95 = (price - 1.96 * std_err, price + 1.96 * std_err)
    
    return price, std_err, ci_95


def simulate_paths(S, T, r, sigma, n_simulations=10_000, n_steps=252, seed=None):
    """
    Simulate full price paths (not just terminal prices).
    
    Needed for path-dependent options (Asian, lookback, barrier).
    
    Returns:
        2D array of shape (n_simulations, n_steps + 1)
        Each row is one simulated path from t=0 to t=T.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Initialize paths: each row is a path, each column is a time step
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S
    
    # Draw all random numbers at once for efficiency
    Z = np.random.standard_normal((n_simulations, n_steps))
    
    # Step through time, updating all paths simultaneously
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])
    
    return paths


def mc_asian_call(S, K, T, r, sigma, n_simulations=10_000, n_steps=252, seed=None):
    """
    Price an Asian (average price) call option via Monte Carlo.
    
    Asian option payoff: max(avg(S over path) - K, 0)
    
    This option's value depends on the AVERAGE price over the life of the
    contract, not just the terminal price. Black-Scholes can't price this
    directly — Monte Carlo is the standard approach.
    
    Asian calls are typically CHEAPER than European calls because averaging
    smooths out big moves, reducing volatility of the payoff.
    """
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps, seed)
    # Average price along each path (excluding the initial price)
    avg_prices = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(avg_prices - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


if __name__ == "__main__":
    # Validation: Monte Carlo for European options should match Black-Scholes
    
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    print("=" * 60)
    print("Monte Carlo vs Black-Scholes — Validation")
    print("=" * 60)
    print(f"Stock: ${S}, Strike: ${K}, T: {T}yr, r: {r*100}%, sigma: {sigma*100}%")
    print("-" * 60)
    
    # Black-Scholes (closed form, exact)
    bs_call = call_price(S, K, T, r, sigma)
    bs_put = put_price(S, K, T, r, sigma)
    
    # Monte Carlo with increasing sample sizes
    print(f"\n{'Method':<30} {'Call':>10} {'Put':>10}")
    print("-" * 60)
    print(f"{'Black-Scholes (exact)':<30} {bs_call:>10.4f} {bs_put:>10.4f}")
    
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        mc_call = mc_call_price(S, K, T, r, sigma, n_simulations=n, seed=42)
        mc_put = mc_put_price(S, K, T, r, sigma, n_simulations=n, seed=42)
        print(f"{'Monte Carlo (' + f'{n:,}' + ' paths)':<30} {mc_call:>10.4f} {mc_put:>10.4f}")
    
    # Show convergence with standard error
    print("\n" + "-" * 60)
    print("Monte Carlo with confidence interval (100K paths):")
    price, se, ci = mc_call_with_stderr(S, K, T, r, sigma, n_simulations=100_000, seed=42)
    print(f"  Estimate: ${price:.4f}")
    print(f"  Std error: ${se:.4f}")
    print(f"  95% CI: (${ci[0]:.4f}, ${ci[1]:.4f})")
    print(f"  Black-Scholes is at ${bs_call:.4f} — should be inside the CI.")
    
    # Now do something Black-Scholes CAN'T — an Asian option
    print("\n" + "=" * 60)
    print("Asian (average price) call — Black-Scholes CAN'T price this")
    print("=" * 60)
    asian_call = mc_asian_call(S, K, T, r, sigma, n_simulations=10_000, n_steps=252, seed=42)
    print(f"  Asian call price:    ${asian_call:.4f}")
    print(f"  European call price: ${bs_call:.4f}")
    print(f"  Asian is cheaper because averaging reduces payoff volatility.")
    print(f"  This is exactly what we'd expect — Monte Carlo working correctly.")

    