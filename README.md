# Dynamic Factor Arbitrage: Isolating Residual Momentum in Sector ETFs

This is a quantitative finance research project that explores **statistical arbitrage** in U.S. equity sector ETFs by extracting tradable signals from factor model residuals. We compare static PCA decomposition against an adaptive **Bayesian state-space model (SSM)** via Kalman filtering to capture time-varying factor exposures. The models are evaluated using a walk-forward validation approach.

---

## Project Overview

Traditional factor models assume static betas (factor loadings), but real markets exhibit **regime changes** where these relationships evolve over time. This project investigates whether adaptive residual extraction using Kalman filtering can generate superior trading signals compared to static PCA approaches.

### Key Research Questions:
1. Can a low-rank factor model ($k=3$) sufficiently explain common variance in sector ETFs?
2. Do factor model residuals exhibit momentum or mean-reversion characteristics?
3. Does dynamic beta estimation via Kalman filtering improve signal quality?

### Results Summary
| Strategy | Sharpe Ratio | Total Return | Max Drawdown |
|----------|--------------|--------------|--------------|
| Fixed PCA + Kalman (Momentum) | **0.16** | **37.2%** | -44.1% |
| Fixed PCA (Momentum) | -0.03 | -5.2% | -48.8% |
| Rolling PCA (Mean Reversion) | -0.85 | -81.7% | -82.7% |

The Kalman-filtered momentum strategy outperforms both static PCA approaches, suggesting that **residual persistence** (momentum) rather than mean-reversion drives alpha in the universe of sector ETFs.

---

## Methodology

### 1. Universe & Data
- **Assets**: 9 SPDR Select Sector ETFs (XLE, XLB, XLI, XLY, XLP, XLV, XLF, XLK, XLU) + SPY:
  - XLE (Energy)
  - XLB (Materials)
  - XLI (Industrials)
  - XLY (Consumer Discretionary)
  - XLP (Consumer Staples)
  - XLV (Health Care)
  - XLF (Financials)
  - XLK (Technology)
  - XLU (Utilities)
  - SPY (S&P 500 benchmark)

- **Period**: January 2012 – January 2026 (14 years of daily data)
- **Returns**: Log returns $r_t = \ln(P_t / P_{t-1})$

### 2. Factor Model Architecture

The core assumption is that asset returns can be decomposed into systematic (factor) and idiosyncratic (residual) components:

$$r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \varepsilon_{i,t}$$

Where:
- $r_{i,t}$ = return of asset $i$ at time $t$
- $\beta_{i,k}$ = exposure of asset $i$ to factor $k$
- $f_{k,t}$ = factor $k$ return at time $t$
- $\varepsilon_{i,t}$ = idiosyncratic residual (trading signal)

---

##  Mathematical Framework

### Principal Component Analysis (PCA)

PCA extracts orthogonal factors that maximise explained variance. Given a returns matrix $\mathbf{R} \in \mathbb{R}^{T \times N}$,

1. **Compute covariance matrix**: $\mathbf{\Sigma} = \frac{1}{T-1}(\mathbf{R} - \bar{\mathbf{R}})^\top(\mathbf{R} - \bar{\mathbf{R}})$

2. **Eigendecomposition**: $\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$
   - $\mathbf{V}$ = matrix of eigenvectors (principal components)
   - $\mathbf{\Lambda}$ = diagonal matrix of eigenvalues

3. **Factor extraction**: Select top $K$ eigenvectors to form factor loadings matrix $\mathbf{W} \in \mathbb{R}^{N \times K}$

4. **Residual computation**:

$$\hat{\mathbf{R}}_t = \mathbf{F}_t \mathbf{W}^\top + \boldsymbol{\mu},$$

$$\varepsilon_t = \mathbf{R}_t - \hat{\mathbf{R}}_t,$$

where $\mathbf{F}_t = (\mathbf{R}_t - \boldsymbol{\mu})\mathbf{W}$ are the factor scores.

#### Implementation Variants
- **Fixed PCA**: Fit on training data, apply fixed loadings to test period
- **Rolling PCA**: Re-estimate loadings using a 60-day rolling window

---

### Bayesian State-Space Model via Kalman Filtering

The Kalman filter enables **time-varying beta estimation**, modelling factor loadings as a random walk:

#### State-Space Representation

**State Equation** (beta evolution):

$$\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \boldsymbol{\omega}_t, \quad \boldsymbol{\omega}_t \sim \mathcal{N}(\mathbf{0}, \delta \mathbf{I})$$

**Observation Equation** (return generation):

$$r_t = \mathbf{f}_t^\top \boldsymbol{\beta}_t + v_t, \quad v_t \sim \mathcal{N}(0, v_e)$$

Where:
- $\boldsymbol{\beta}_t \in \mathbb{R}^K$ = time-varying factor loadings at time $t$
- $\mathbf{f}_t \in \mathbb{R}^K$ = PCA factor scores at time $t$
- $\delta$ = state transition noise (controls adaptation speed)
- $v_e$ = observation noise variance

#### Kalman Filter Recursion

**Prediction Step**:

$$\hat{\boldsymbol{\beta}}_{t|t-1} = \hat{\boldsymbol{\beta}}_{t-1|t-1}$$

$$\mathbf{P}_{t|t-1} = \mathbf{P}_{t-1|t-1} + \delta \mathbf{I}$$

**Update Step**:

$$e_t = r_t - \mathbf{f}_t^\top \hat{\boldsymbol{\beta}}_{t|t-1} \quad \text{(innovation/residual)}$$

$$Q_t = \mathbf{f}_t^\top \mathbf{P}_{t|t-1} \mathbf{f}_t + v_e \quad \text{(innovation variance)}$$

$$\mathbf{K}_t = \frac{\mathbf{P}_{t|t-1} \mathbf{f}_t}{Q_t} \quad \text{(Kalman gain)}$$

$$\hat{\boldsymbol{\beta}}_{t|t} = \hat{\boldsymbol{\beta}}_{t|t-1} + \mathbf{K}_t e_t \quad \text{(updated state)}$$

$$\mathbf{P}_{t|t} = \mathbf{P}_{t|t-1} - \mathbf{K}_t \mathbf{f}_t^\top \mathbf{P}_{t|t-1} \quad \text{(updated covariance)}$$

The **innovation** $e_t$ serves as the trading signal, representing the component of returns unexplained by the adaptive factor model.

#### Hyperparameter Interpretations
| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| $\delta$ (process noise) | Slow adaptation, stable betas | Fast adaptation, responsive to regime shifts |
| $v_e$ (observation noise) | Trust observations more | Smooth through noise |

---

### Trading Strategy

**Cross-Sectional Momentum on Residuals**:

1. **Rank residuals**: At each time $t$, compute percentile ranks of $\varepsilon_{i,t}$ across all assets
2. **Signal generation**:
   - Long: Assets with residual rank > 90th percentile
   - Short: Assets with residual rank < 10th percentile
3. **Portfolio construction**: Equal-weight within long/short legs, dollar-neutral
4. **Execution**: Trade at close on day $t$, positions enter on day $t+1$

---

##  Implementation

### Tech Stack
- **JAX**: GPU-accelerated Kalman filter with JIT compilation and automatic vectorization (`vmap`)
- **scikit-learn**: PCA implementation
- **pandas/numpy**: Data manipulation
- **yfinance**: Market data retrieval

### Project Structure
```
├── src/
│   ├── data_loader.py      # ETF data download & preprocessing
│   ├── static_pca.py       # Fixed and rolling PCA residual extraction
│   ├── kalman_jax.py       # JAX-accelerated Kalman filter
│   ├── backtest.py         # Backtesting engine with transaction costs
│   └── run_walk_forward.py # Walk-forward validation framework
├── data/
│   ├── etf_returns.csv     # Log returns (2012-2025)
│   ├── walk_forward_config.csv  # Fold definitions
│   └── experiment_results.csv   # Strategy performance metrics
├── Notebooks/
│   └── model_justification.ipynb  # Statistical validation & visualizations
└── requirements.txt
```

### Walk-Forward Validation
To prevent look-ahead bias, the framework uses **walk-forward validation**:
- **Training window**: 3 years
- **Test window**: 1 year
- **Folds**: 10 non-overlapping test periods (2015–2025)

```
Fold 1: Train [2012-2015) → Test [2015-2016)
Fold 2: Train [2013-2016) → Test [2016-2017)
...
Fold 10: Train [2021-2024) → Test [2024-2025)
```

---

## Installation and Setup

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

### Model Justification Notebook
The Jupyter notebook provides statistical validation:
- **Scree plot**: Confirms 3 factors capture >80% of variance
- **Q-Q plot**: Reveals fat-tailed residuals (excess kurtosis)
- **Parameter sensitivity heatmaps**: Demonstrates Kalman robustness

---

## Key Findings

1. **Momentum dominates mean-reversion**: Contrary to classical stat-arb intuition, residuals exhibit **persistence** rather than mean-reversion, suggesting information diffusion dynamics in sector rotation.

2. **Adaptive betas matter**: The Kalman filter's ability to track time-varying factor exposures produces cleaner residuals, reducing the risk of trading beta disguised as alpha.

3. **Transaction costs are critical**: High turnover (~170% daily) erodes returns significantly; the 2bp cost assumption represents an optimistic institutional scenario.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

