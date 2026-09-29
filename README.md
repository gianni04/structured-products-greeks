# Structured Products & Greeks

Python library (`structrisk`) for pricing options and structured products and
measuring their risk: Black-Scholes greeks up to third order, binomial tree,
implied volatility, Monte Carlo pricing of autocallables and other notes,
portfolio greeks and a delta hedging simulation.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![tests: 14 passed](https://img.shields.io/badge/tests-14%20passed-brightgreen)

## What it covers

- **Greeks:** delta, gamma, vega, theta, rho, and higher order (vanna, volga,
  charm, speed, zomma, color, veta), all from closed-form formulas.
- **Pricing:** Black-Scholes-Merton, Cox-Ross-Rubinstein tree, implied vol
  (Newton-Raphson with a Brent fallback).
- **Structured products by Monte Carlo:** Phoenix autocall (memory coupon),
  reverse convertible, capped bonus certificate, capital protected note,
  worst-of basket option. Greeks by bump-and-revalue with common random
  numbers, so they stay stable on discontinuous payoffs.
- **Portfolio:** aggregated greeks, delta-equivalent exposure, spot x vol
  stress grid.

Tests cover put-call parity, analytic delta vs finite differences, tree
convergence to Black-Scholes, implied vol round trips, the Monte Carlo
confidence interval and no-arbitrage bounds on the autocall.

## Run

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q
python examples/03_autocall_pricing.py
```

```python
from structrisk.blackscholes import all_greeks
g = all_greeks(spot=100, strike=105, maturity=1.0, rate=0.03, dividend=0.02, vol=0.22, option_type="call")
```

## Results

**Greeks:** ATM call, 1 year (r = 3%, q = 2%, vol = 22%): delta 0.551,
gamma 0.0176, vega 38.6, theta -0.0124 per day. ATM gamma at 1 month is 5.3x
the gamma at 2 years.

![Greeks](docs/img/01_greeks_surface.png)

**Phoenix autocall** (4 semi-annual observations, 4% coupon, barriers
100% / 70% / 60%, 200,000 paths): price 100.71 (95% CI 100.64 to 100.77).
Probability of early redemption at the first date 47.6%; probability of
reaching maturity 34.4%, and of a capital loss in that case 22.7%.

![Autocall](docs/img/03_autocall_pricing.png)

**Portfolio** (long call, short put, protected note, bonus certificate):
delta-equivalent exposure 66,931 EUR, vega P&L +236 EUR per vol point. The
spot x vol stress grid goes from -12,827 EUR to +18,663 EUR.

**Delta hedging:** the replication error falls from 5.40 (one rehedge) to
0.39 (daily rehedging), in line with 1/sqrt(n). With 10 bp costs, the average
cost grows from -0.11 to -0.56, so hedging more often is not always better.

![Delta hedging](docs/img/05_delta_hedging.png)

## Limitations

- Constant volatility per asset (no stochastic vol, no jumps). The vol surface
  in example 2 is a synthetic parametrisation.
- No issuer credit risk in the structured product prices.
- Barriers are monitored on the simulation time grid, not continuously.

## References

Hull, *Options, Futures, and Other Derivatives*; Haug, *The Complete Guide to
Option Pricing Formulas*; Glasserman, *Monte Carlo Methods in Financial
Engineering*.
