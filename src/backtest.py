import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, cost_bps: float = 0.0):
        self.cost = cost_bps

    def run(
        self,
        returns: pd.DataFrame,
        residuals: pd.DataFrame,
        signal: str = "momentum",
        return_components: bool = False,
    ) -> pd.DataFrame:
        """
        Rank-Based Momentum Strategy.
        INPUTS:
            returns: LOG Returns (from data_loader)
            residuals: Model residuals
            signal: "momentum" or "mean_reversion"
        OUTPUT:
            net_pnl: Log PnL stream
            if return_components: (net_pnl, residuals, components)
        """
        # Align Data
        common_cols = returns.columns.intersection(residuals.columns)
        returns = returns[common_cols]
        residuals = residuals[common_cols]

        common_index = returns.index.intersection(residuals.index)
        returns = returns.loc[common_index]
        residuals = residuals.loc[common_index]

        # Drop rows where residuals are entirely missing
        residuals = residuals.dropna(how="all")
        returns = returns.loc[residuals.index]

        # 1. Rank residuals daily (0.0 to 1.0)
        ranks = residuals.rank(axis=1, pct=True)

        # 2. Generate Signals
        signal = signal.lower().replace("-", "_")
        if signal == "momentum":
            longs = (ranks > 0.9).astype(int)
            shorts = (ranks < 0.1).astype(int)
        elif signal == "mean_reversion":
            longs = (ranks < 0.1).astype(int)
            shorts = (ranks > 0.9).astype(int)
        else:
            raise ValueError(f"Unknown signal: {signal}")

        # 3. Create Neutral Portfolio
        positions = longs - shorts

        # 4. Capital Allocation (100% Gross Leverage)
        daily_count = positions.abs().sum(axis=1).replace(0, 1)
        positions = positions.div(daily_count, axis=0)

        # Shift (Trade at Close: Signal T -> Position T+1)
        positions = positions.shift(1).fillna(0)

        # 5. Returns (Approximation: Position * LogRet ~= LogRet Contribution)
        asset_pnl = positions * returns
        gross_pnl = asset_pnl.sum(axis=1)

        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * self.cost
        net_pnl = gross_pnl - costs if self.cost > 0 else gross_pnl

        if return_components:
            components = {
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "turnover": turnover,
                "costs": costs,
            }
            return net_pnl, residuals, components

        return net_pnl, residuals

    def analyze(self, log_returns: pd.Series, title: str):
        """
        Print performance metrics using Log Returns logic.
        """
        # Wealth Index (Exponential for Log Returns)
        # Wealth_t = exp(Sum(r_0...r_t))
        wealth_index = np.exp(log_returns.cumsum())

        # Metrics
        # Sharpe on log returns is standard approximation
        if log_returns.std() == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252) * log_returns.mean() / log_returns.std()

        # Drawdown
        peaks = wealth_index.cummax()
        drawdowns = (wealth_index - peaks) / peaks
        max_dd = drawdowns.min()

        total_ret = wealth_index.iloc[-1] - 1

        print(f"{'='*40}")
        print(f"Results: {title}")
        print(f"{'='*40}")
        print(f"Total Return:   {total_ret:.2%}")
        print(f"Annual Sharpe:  {sharpe:.2f}")
        print(f"Max Drawdown:   {max_dd:.2%}")
        print(f"{'='*40}")

        return wealth_index

def main():
    pass

if __name__ == "__main__":
    main()
