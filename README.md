# Adaptive Factor Models for Sector ETF Long/Short Trading: Kalman vs PCA

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![JAX](https://img.shields.io/badge/JAX-green)
![License](https://img.shields.io/badge/License-MIT-purple)

This project tests whether **tradable signals** exist in **factor-model residuals** for U.S. sector ETFs.  We compare:

- **Static factor extraction** via PCA (fixed vs rolling loadings)
- **Adaptive residual extraction** via a **Bayesian State Space Model (Kalman filter)** (time-varying betas on PCA factors)

Evaluation method: **Out-of-sample walk-forward** evaluation

- Training period: 3 years
- Testing period: 1 year
- Years: 2015–2025 

Backtest engine:

- **Rebalances weekly** (by default, every Friday).
- **Builds a market-neutral long/short portfolio**: long some ETFs, short others so net dollars are approximately zero; removes SPY exposure when possible by estimating how sensitive each ETF is to SPY using recent history, then adjusts the L/S weights so the portfolio’s overall movement isn’t explained by SPY.
- **One-day delay to avoid look-ahead bias**: signals are computed using data available by the end of day $t$, and the resulting weights are applied starting day $t+1$.
- **Includes simple transaction costs**: 2 bps per one-way turnover by default.

Strategies tested:

The evaluation covers 22 long/short residual strategies across two residual sources (fixed PCA and rolling PCA) and two signals (mean-reversion and residual momentum), plus an adaptive residual variant where PCA factors are held fixed and Kalman filtering is used to track time-varying factor loadings. For the Kalman setup, a $3\times 3$ grid over the process/observation noise parameters ($\delta$, $v_e$) is run to see when dynamic loadings help relative to static PCA residuals.

---

## Summary Results

**Main result**: In this small sector-ETF universe, **mean-reversion signals from factor residuals** outperform residual momentum, and **allowing factor exposures (“betas”) to vary over time via a Kalman filter can further improve mean-reversion performance** for some $\delta$ (process noise) and $v_e$ (observation noise) parameter choices.


| Strategy | Sharpe | Total Return | MaxDD | VaR | ES | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Fixed PCA + Kalman (MeanRev, $\delta$=1e-4, $v_e$=1e-4) | 0.56 | 15.82% | -3.74% | -0.17% | -0.36% | 0.199 |
| Fixed PCA (MeanRev) | 0.40 | 10.60% | -3.43% | -0.17% | -0.37% | 0.199 |
| Rolling PCA (MeanRev) | -0.24 | -6.21% | -13.40% | -0.20% | -0.43% | 0.199 |
| Fixed PCA (Momentum) | -0.73 | -16.56% | -19.58% | -0.18% | -0.43% | 0.182 |

> Notes:
> - VaR/ES here correspond to 1-day Value-at-Risk/Expected Shortfall at the 95% confidence interval.

### Cumulative wealth under different strategies

![Cumulative wealth under different strategies](figures/wealth_curves.png)

This graph shows the cumulative wealth from a 10-fold walk-forward test (train 3 years, trade the next 1 year) under different strategies (see legend), with each test year chained together into a single out-of-sample equity curve. 

- **Mean-reversion outperforms momentum**: Both mean-reversion variants finish above their momentum counterparts across most of the sample.
- **Best-performing model**: Kalman Filtering with fixed PCA factors, mean-reversion delivers the strongest out-of-sample equity curve, finishing around +15–16%.
- **Rolling PCA underperforms**: Rolling PCA is less stable and ends worse than fixed PCA, suggesting frequent re-estimation hurts in this small ETF universe.
- **Regime-dependent edge**: Most of the performance separation appears post-2019; earlier years are flatter and closer together.
---

## Quickstart

### Prerequisites
- Python 3.12+
- Git

### Installation Steps:

1.  Clone repository

```bash
git clone https://github.com/micyiucm/Dynamic-Factor-Arbitrage-Isolating-Residual-Momentum-in-Sector-ETFs.git
cd Dynamic-Factor-Arbitrage-Isolating-Residual-Momentum-in-Sector-ETFs
```
2. Create and activate virtual environment

```bash
python -m venv .venv
# On Linux/macOS
source .venv/bin/activate
# On Windows
.\.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

### Run Experiments

1. Download data and generate returns
```bash
python src/data_loader.py
```

2. Generate PCA residuals

```bash
python src/static_pca.py
```

3. Run walk-forward validation for all strategies

```bash
python src/run_walk_forward.py
```