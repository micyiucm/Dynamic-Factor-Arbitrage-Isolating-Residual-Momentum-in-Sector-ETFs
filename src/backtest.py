# backtest.py
# Backtester class to simulate a dollar-neutral (and optionally beta-neutral) L/S strategy
# Trading signals are derived from factor model residuals.

import pandas as pd
import numpy as np


class Backtester:
    """
    Backtester class to simulate a L/S equity strategy that trades on factor model residuals.
    
    Strategy logic:
        - Momentum: Long assets with rising residuals, short falling residuals
        - Mean-reversion: Long assets with low residuals, short assets with high residuals
    
    Portfolio constraints:
        - Dollar-neutral: ∑w = 0 (equal long/short exposure)
        - Beta-neutral (optional): β'w = 0 (zero market beta)
    """
    def __init__(
        self,
        cost_bps: float = 0.0,          # Transaction cost per unit turnover (e.g. 0.0002 = 2 bps)
        long_q: float = 0.7,            # Percentile threshold for longs (e.g. 0.7 means top 30%)
        short_q: float = 0.3,           # Percentile threshold for shorts (e.g. 0.3 means bottom 30%)
        rebalance: str = "W-FRI",       # Rebalance frequency (weekly on Fridays)
        beta_neutral: bool = True,      # Enforce beta neutrality to market
        beta_window: int = 60,          # Rolling window (days) for beta estimation
        market_col: str = "SPY",        # Market index column name
        trade_market: bool = False,     # False means do not trade SPY
        # Signal hyperparams (kept here for clarity + reproducibility)
        vol_halflife: int = 20,         # EWMA vol halflife for residual standardization
        mom_lookback: int = 20,         # L-day diff lookback for residual momentum
        z_clip: float = 5.0,            # clip standardized residuals
        min_beta_obs: int = 10,         # minimum obs for beta estimation
    ):
        self.cost = float(cost_bps)
        self.long_q = float(long_q)
        self.short_q = float(short_q)
        self.rebalance = rebalance
        self.beta_neutral = bool(beta_neutral)
        self.beta_window = int(beta_window)
        self.market_col = market_col
        self.trade_market = bool(trade_market)

        self.vol_halflife = int(vol_halflife)
        self.mom_lookback = int(mom_lookback)
        self.z_clip = float(z_clip)
        self.min_beta_obs = int(min_beta_obs)

    @staticmethod
    def _project_out_constraints(w: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """
        Project weights to satisfy:
          1) Dollar neutrality: 1^T w = 0
          2) Beta neutrality:   beta^T w = 0

        w_adj = w - A (A^T A)^+ A^T w,  A = [1, beta]
        """
        n = w.shape[0]
        A = np.column_stack([np.ones(n), betas])  # (n,2)
        inv_ATA = np.linalg.pinv(A.T @ A)
        return w - A @ (inv_ATA @ (A.T @ w))

    def run(
        self,
        returns: pd.DataFrame,              # Asset log returns (index=dates, columns=tickers)
        residuals: pd.DataFrame,            # Residuals/innovations (same structure as returns)
        signal: str = "momentum",           # "momentum" or "mean_reversion"
        return_components: bool = False,    # If True, return PnL breakdown
    ):
        # -------------------------
        # 0) Align data
        # -------------------------
        common_cols = returns.columns.intersection(residuals.columns)
        returns = returns[common_cols]
        residuals = residuals[common_cols]

        common_index = returns.index.intersection(residuals.index)
        returns = returns.loc[common_index]
        residuals = residuals.loc[common_index]

        # Market column check (if beta-neutral is requested)
        if self.beta_neutral and self.market_col not in returns.columns:
            raise ValueError(f"market_col='{self.market_col}' missing from returns")

        # Separate market vs tradables
        market = returns[self.market_col] if self.market_col in returns.columns else None
        tradable_cols = list(returns.columns)
        if (not self.trade_market) and (self.market_col in tradable_cols):
            tradable_cols.remove(self.market_col)

        # Drop days with no residuals on tradables
        residuals_tr = residuals[tradable_cols].dropna(how="all")
        returns_tr = returns.loc[residuals_tr.index, tradable_cols]
        if market is not None:
            market = market.loc[residuals_tr.index]

        # -------------------------
        # 1) Rebalance dates
        # -------------------------
        reb_dates = residuals_tr.resample(self.rebalance).last().index
        reb_dates = reb_dates.intersection(residuals_tr.index)

        if len(reb_dates) == 0:
            empty = pd.Series(dtype=float)
            if return_components:
                comps = {"gross_pnl": empty, "net_pnl": empty, "turnover": empty, "costs": empty}
                return empty, residuals_tr, comps
            return empty, residuals_tr

        # -------------------------
        # 2) Signal generation (vectorized, causal)
        # -------------------------
        signal_key = signal.lower().replace("-", "_")
        X = residuals_tr[tradable_cols]

        # 2.1 Volatility normalization (EWMA std). We DO NOT shift here because:
        # - Signal is formed using information up to Close T
        # - We execute with positions.shift(1), so returns at T are never traded using signal at T.
        vol = X.ewm(halflife=self.vol_halflife, adjust=False).std()

        # Avoid divide-by-zero / garbage early windows
        vol = vol.clip(lower=1e-8)

        # Standardized residual level (winsorized)
        Z = (X / vol).replace([np.inf, -np.inf], np.nan).clip(-self.z_clip, self.z_clip)

        # 2.2 Score definition
        if signal_key == "momentum":
            # Residual momentum = L-day diff of standardized residual level
            # Positive score => residual has drifted up over last L days
            L = self.mom_lookback
            score_df = Z.diff(L)
        elif signal_key == "mean_reversion":
            # Mean reversion uses standardized level itself
            score_df = Z
        else:
            raise ValueError(f"Unknown signal: {signal}")

        # -------------------------
        # 3) Build target weights on rebalance dates
        # -------------------------
        target_w = pd.DataFrame(0.0, index=residuals_tr.index, columns=tradable_cols)

        for dt in reb_dates:
            score = score_df.loc[dt].dropna()
            if score.empty:
                continue

            ranks = score.rank(pct=True)

            if signal_key == "momentum":
                # High score -> long; low score -> short
                longs = ranks[ranks >= self.long_q].index
                shorts = ranks[ranks <= self.short_q].index
            else:  # mean_reversion
                # Low level -> long; high level -> short
                longs = ranks[ranks <= self.short_q].index
                shorts = ranks[ranks >= self.long_q].index

            # Need both legs to keep it meaningfully L/S
            if len(longs) == 0 or len(shorts) == 0:
                continue

            # Equal weight within legs
            w = pd.Series(0.0, index=tradable_cols)
            w.loc[longs] = 1.0 / len(longs)
            w.loc[shorts] = -1.0 / len(shorts)

            # Optional beta-neutrality at rebalance date
            if self.beta_neutral and market is not None:
                idx = returns_tr.index.get_loc(dt)
                start = max(0, idx - self.beta_window)

                if idx - start >= self.min_beta_obs:
                    win_r = returns_tr.iloc[start:idx]      # strictly past
                    win_m = market.iloc[start:idx]
                    var_m = win_m.var()

                    if var_m > 0:
                        covs = win_r.apply(lambda s: s.cov(win_m), axis=0)
                        betas_full = (covs / var_m).reindex(tradable_cols).fillna(0.0).values

                        w_arr = w.values
                        active = (w_arr != 0.0)

                        # Need >=3 active names to satisfy 2 constraints without degeneracy
                        if active.sum() >= 3:
                            w_act = w_arr[active]
                            b_act = betas_full[active]
                            w_act_adj = self._project_out_constraints(w_act, b_act)

                            w_new = np.zeros_like(w_arr)
                            w_new[active] = w_act_adj
                            w = pd.Series(w_new, index=tradable_cols)

            # Renormalize to 100% gross exposure (sum |w| = 1)
            gross = w.abs().sum()
            if gross > 0:
                w = w / gross
                target_w.loc[dt] = w

        # -------------------------
        # 4) Hold between rebalances + execution lag
        # -------------------------
        # Hold last active portfolio between rebalances (explicit)
        positions = target_w.ffill().fillna(0.0)

        # Execution lag: signal/weights decided at Close T -> position active for returns from T+1
        positions = positions.shift(1).fillna(0.0)

        # -------------------------
        # 5) PnL calculation
        # -------------------------
        simple_ret = np.expm1(returns_tr)                 # convert log->simple for portfolio arithmetic
        gross_simple = (positions * simple_ret).sum(axis=1)

        # One-way turnover and linear costs
        turnover = 0.5 * positions.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * self.cost

        net_simple = (gross_simple - costs).clip(lower=-0.999999)

        # Convert back to log returns
        gross_log = np.log1p(gross_simple.clip(lower=-0.999999))
        net_log = np.log1p(net_simple)

        if return_components:
            components = {
                "gross_pnl": gross_log,
                "net_pnl": net_log,
                "turnover": turnover,
                "costs": costs,
            }
            return net_log, residuals_tr, components

        return net_log, residuals_tr
