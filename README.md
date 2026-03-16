# Black-Scholes Options Pricer

A clean, interactive pricer for European vanilla options (call/put) built with Streamlit. Computes BS price, full Greeks, and renders sensitivity charts in real time.

## Getting started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens on `localhost:Lien BS Pricer (8501)`.

## Inputs
| Parameter | Description |
| Spot S | Current underlying price |
| Strike K | Option exercise price |
| Maturity T | Days to expiration |
| Rate r | Risk-free rate (annualized) |
| Volatility σ | Implied vol (annualized) |
| Dividend q | Continuous dividend yield — set to 0 if none |
| Premium paid | Optional. If filled, P&L is computed against your actual entry cost rather than the theoretical BS price |

## Outputs
**Pricing**
- Black-Scholes price, intrinsic value, time value
- Break-even at expiration
- Risk-neutral probability of finishing ITM

**Greeks — 1st order**
- **Delta** — price sensitivity to spot; also reads as hedge ratio
- **Gamma** — rate of change of delta; peaks at ATM
- **Vega** — price change per +1% move in implied vol
- **Theta** — daily time decay; always negative for the option buyer
- **Rho** — sensitivity to the risk-free rate

**Greeks — 2nd order**
- **Vanna** — how delta changes when vol moves
- **Volga** — convexity with respect to vol (vol of vol exposure)
- **Charm** — how delta drifts over time

## Charts
- **P&L at expiration** — payoff profile with break-even, strike, and current value overlaid
- **Price vs Volatility** — how the premium reacts to vol shocks
- **Price vs Time** — time decay curve from today to expiration
- **Delta / Gamma / Vega / Theta vs Spot** — full sensitivity curves across the strike range

## Stack
Python · NumPy · SciPy · Matplotlib · Streamlit

## Limitations
Black-Scholes assumes constant volatility and log-normal returns. In practice, implied vol varies across strikes and maturities (volatility smile/skew), which means BS will misprice deep OTM options or long-dated structures. This pricer is accurate for European vanilla options on equities and indices under standard assumptions.