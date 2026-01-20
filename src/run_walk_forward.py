# Walk-forward validation script that tests multiple factor model configurations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from backtest import Backtester
from kalman_jax import KalmanFilterJAX

# Configuration
DATA_PATH = 'data/etf_returns.csv'
CONFIG_PATH = 'data/walk_forward_config.csv'
PCA_FIXED_RES_PATH = "data/pca_residuals_fixed_etf.csv"
PCA_ROLLING_RES_PATH = "data/pca_residuals_rolling_etf.csv"
COST_BPS = 0.0002  # 2 bps
N_FACTORS = 3
ROLLING_WINDOW = 60
KALMAN_DELTAS = [1e-4, 1e-3, 1e-2]
KALMAN_VES = [1e-4, 1e-3, 1e-2]

def load_data():
    """
    Load returns data, walk-forward configuration, and pre-computed PCA residuals.
    
    Returns:
        tuple: (returns DataFrame, config DataFrame, dict of residuals)
    """
    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    config = pd.read_csv(CONFIG_PATH, parse_dates=['Train_Start', 'Train_End', 'Test_Start', 'Test_End'])
    try:
        fixed_res = pd.read_csv(PCA_FIXED_RES_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Warning: missing {PCA_FIXED_RES_PATH}. Run src/static_pca.py.")
        fixed_res = None

    try:
        rolling_res = pd.read_csv(PCA_ROLLING_RES_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Warning: missing {PCA_ROLLING_RES_PATH}. Run src/static_pca.py.")
        rolling_res = None

    return returns, config, {"fixed": fixed_res, "rolling": rolling_res}

def calc_sharpe(log_returns: pd.Series) -> float:
    """
    Calculate annualized Sharpe ratio from log returns.
    Parameters:
        log_returns: Series of daily log returns
    Returns:
        Annualized Sharpe ratio (assumes 252 trading days/year)
    Math:
    Sharpe Ratio = sqrt(T)*(mu/sigma), where
    T = 252 trading days per year
    mu = mean(r_t) = average daily log return
    sigma = std(r_t) = standard deviation of daily log return
    """
    if log_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * log_returns.mean() / log_returns.std()

def calc_max_drawdown(wealth: pd.Series) -> float:
    """
    Calculate maximum drawdown from wealth curve.
    Parameters:
        wealth: Cumulative wealth series (not log returns)
    Returns:
        Maximum drawdown as negative percentage (e.g., -0.25 = -25% loss)
    Math:
        MaxDD = min_t[(W_t - P_t) / P_t], where
        W_t = wealth at time t
        P_t = max(W_s for s <= t) = peak up to time t
    """
    # Track running maximum (peaks)
    peaks = wealth.cummax()
    # Calculate drawdown from peak at each point
    drawdowns = (wealth - peaks) / peaks
    # Return worst drawdown (most negative)
    return drawdowns.min()

def calc_cagr(log_returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate from log returns.
    Parameters:
        log_returns: Series of daily log returns
    Returns:
        CAGR as decimal (e.g., 0.15 = 15% annualized)
    Math:
        CAGR = exp(sum(r_t) / T) - 1, where
        r_t = daily log return at time t
        T = number of years = n_days / 252
    """
    years = len(log_returns) / 252
    if years == 0:
        return 0.0
    return np.exp(log_returns.sum() / years) - 1


def calc_var_es(simple_returns: pd.Series, confidence: float = 0.95) -> tuple:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) on simple returns.

    Returns (VaR, ES) as negative decimals (e.g., -0.05 = -5% one-day loss).
    """
    if simple_returns.empty:
        return 0.0, 0.0

    var = simple_returns.quantile(1 - confidence)
    tail_losses = simple_returns[simple_returns <= var]
    es = tail_losses.mean() if not tail_losses.empty else var
    return var, es

def calc_metrics(net_pnl: pd.Series, gross_pnl: pd.Series, turnover: pd.Series) -> dict:
    """Calculate comprehensive performance metrics from PnL series."""
    if net_pnl.empty:
        return {
            "Total_Return": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "VaR_95": 0.0,
            "ES_95": 0.0,
            "Avg_Turnover": 0.0,
            "Cost_Impact": 0.0,
        }

    # Existing metrics (net_pnl is log return series)
    wealth = np.exp(net_pnl.cumsum())
    total_return = wealth.iloc[-1] - 1
    sharpe = calc_sharpe(net_pnl)
    max_dd = calc_max_drawdown(wealth)
    avg_turnover = turnover.mean() if not turnover.empty else 0.0

    # VaR / ES should be computed on SIMPLE returns
    net_simple = np.expm1(net_pnl)
    var_95, es_95 = calc_var_es(net_simple, confidence=0.95)

    gross_return = np.exp(gross_pnl.sum()) - 1 if not gross_pnl.empty else total_return
    cost_impact = gross_return - total_return

    return {
        "Total_Return": total_return,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "VaR_95": var_95,
        "ES_95": es_95,
        "Avg_Turnover": avg_turnover,
        "Cost_Impact": cost_impact,
    }

def run_walk_forward_strategy(
    name,
    returns,
    config,
    tester,
    pca_residuals,
    pca_mode,
    signal,
    use_kalman=False,
    delta=None,
    ve=None,
    print_folds=True,
    print_summary=True,
):
    """
    Execute walk-forward validation for a single strategy configuration.
    
    For each fold: trains factor model on training period, generates residuals, then
    executes trading strategy on test period and collects performance metrics.
    
    Parameters:
        name: Strategy name for display
        returns: Full returns DataFrame
        config: Walk-forward fold configuration
        tester: Backtester instance
        pca_residuals: Dict with 'fixed' and 'rolling' pre-computed residuals
        pca_mode: 'fixed' or 'rolling' PCA variant
        signal: 'mean_reversion' or 'momentum' trading signal
        use_kalman: Whether to use Kalman filter for adaptive residuals
        delta: Kalman process noise (if use_kalman=True)
        ve: Kalman observation noise (if use_kalman=True)
        print_folds: Whether to print per-fold results
        print_summary: Whether to print aggregate metrics
    
    Returns:
        tuple: (concatenated PnL series, aggregate metrics dict, fold table DataFrame)
    """
    print(f"\n--- Running Walk-Forward: {name} ---")
    if use_kalman and pca_mode != "fixed":
        raise ValueError("Kalman requires fixed PCA factors; set pca_mode='fixed'.")

    residuals_source = None
    if not use_kalman:
        if pca_residuals is None:
            print("Warning: missing PCA residuals.")
            empty = pd.Series(dtype=float)
            metrics = calc_metrics(empty, empty, empty)
            metrics["CAGR"] = 0.0
            return empty, metrics, pd.DataFrame()
        residuals_source = pca_residuals.get(pca_mode)
        if residuals_source is None:
            print(f"Warning: missing residuals for pca_mode='{pca_mode}'.")
            empty = pd.Series(dtype=float)
            metrics = calc_metrics(empty, empty, empty)
            metrics["CAGR"] = 0.0
            return empty, metrics, pd.DataFrame()

    stitched_pnl = []
    stitched_gross = []
    stitched_turnover = []
    fold_rows = []

    for i, row in config.iterrows():
        # 1. Dates
        train_start, train_end = row["Train_Start"], row["Train_End"]
        test_start, test_end = row["Test_Start"], row["Test_End"]

        # 2. Slice Data
        # We need training data to fit the PCA and warm-start the Kalman
        train_data = returns.loc[train_start:train_end]
        test_data = returns.loc[test_start:test_end]

        if len(train_data) < ROLLING_WINDOW or len(test_data) == 0:
            continue

        # 3. Generate factors and residuals
        if use_kalman:
            # Fit PCA on train only, apply to test for Kalman factors
            pca = PCA(n_components=N_FACTORS)
            pca.fit(train_data)
            if delta is None or ve is None:
                raise ValueError("delta/ve required when use_kalman=True")
            combined_data = pd.concat([train_data, test_data])
            combined_factors = pca.transform(combined_data)
            combined_factors_df = pd.DataFrame(combined_factors, index=combined_data.index)
            combined_centered = combined_data - pca.mean_
            kf = KalmanFilterJAX(n_factors=N_FACTORS, delta=delta, ve=ve)
            combined_residuals = kf.fit_transform(combined_centered, combined_factors_df)
            test_residuals = combined_residuals.loc[test_start:test_end]
        else:
            test_residuals = residuals_source.loc[test_start:test_end]

        # 4. Execute trading strategy
        pnl, _, components = tester.run(
            test_data,
            test_residuals,
            signal=signal,
            return_components=True,
        )
        gross_pnl = components["gross_pnl"]
        turnover = components["turnover"]
        stitched_pnl.append(pnl)
        stitched_gross.append(gross_pnl)
        stitched_turnover.append(turnover)

        metrics = calc_metrics(pnl, gross_pnl, turnover)
        fold_rows.append({
            "Fold": row["Fold"],
            "Start": test_start.date().isoformat(),
            "End": test_end.date().isoformat(),
            "Total_Return": metrics["Total_Return"],
            "Sharpe": metrics["Sharpe"],
            "MaxDD": metrics["MaxDD"],
            "VaR_95": metrics["VaR_95"],
            "ES_95": metrics["ES_95"],
            "Avg_Turnover": metrics["Avg_Turnover"],
            "Cost_Impact": metrics["Cost_Impact"],
        })
        if print_folds:
            print(f"Fold {row['Fold']} [{test_start.date()} -> {test_end.date()}]: {metrics['Total_Return']:.2%}")

    if not stitched_pnl:
        empty = pd.Series(dtype=float)
        metrics = calc_metrics(empty, empty, empty)
        metrics["CAGR"] = 0.0
        return empty, metrics, pd.DataFrame()

    net_all = pd.concat(stitched_pnl)
    gross_all = pd.concat(stitched_gross)
    turnover_all = pd.concat(stitched_turnover)
    agg_metrics = calc_metrics(net_all, gross_all, turnover_all)
    cagr = calc_cagr(net_all)

    fold_table = pd.DataFrame(fold_rows)
    if print_folds:
        print("\nFold Summary:")
        print(
            fold_table.to_string(
                index=False,
                formatters={
                    "Total_Return": "{:.2%}".format,
                    "Sharpe": "{:.2f}".format,
                    "MaxDD": "{:.2%}".format,
                    "VaR_95": "{:.2%}".format,
                    "ES_95": "{:.2%}".format,
                    "Avg_Turnover": "{:.3f}".format,
                    "Cost_Impact": "{:.2%}".format,
                },
            )
        )

    if print_summary:
        print("\nAggregate Metrics:")
        print(f"Total Return:   {agg_metrics['Total_Return']:.2%}")
        print(f"CAGR:           {cagr:.2%}")
        print(f"Sharpe:         {agg_metrics['Sharpe']:.2f}")
        print(f"Max Drawdown:   {agg_metrics['MaxDD']:.2%}")
        print(f"VaR 95%:        {agg_metrics['VaR_95']:.2%}")
        print(f"ES 95%:         {agg_metrics['ES_95']:.2%}")
        print(f"Avg Turnover:   {agg_metrics['Avg_Turnover']:.3f}")
        print(f"Cost Impact:    {agg_metrics['Cost_Impact']:.2%}")

    agg_metrics["CAGR"] = cagr
    return net_all, agg_metrics, fold_table

def signal_label(signal: str) -> str:
    if signal == "mean_reversion":
        return "MeanRev"
    return "Momentum"

def build_experiments():
    experiments = []

    for pca_mode in ["fixed", "rolling"]:
        for signal in ["mean_reversion", "momentum"]:
            label = signal_label(signal)
            experiments.append({
                "name": f"{pca_mode.title()} PCA | {label}",
                "pca_mode": pca_mode,
                "signal": signal,
                "use_kalman": False,
                "print_folds": True,
                "print_summary": True,
                "plot": True,
            })

    for delta in KALMAN_DELTAS:
        for ve in KALMAN_VES:
            for signal in ["mean_reversion", "momentum"]:
                label = signal_label(signal)
                experiments.append({
                    "name": f"Fixed PCA + Kalman | {label} | d={delta:g}, ve={ve:g}",
                    "pca_mode": "fixed",
                    "signal": signal,
                    "use_kalman": True,
                    "delta": delta,
                    "ve": ve,
                    "print_folds": False,
                    "print_summary": False,
                    "plot": delta == 1e-3 and ve == 1e-3 and signal == "momentum",
                })

    return experiments

def main():
    returns, config, pca_residuals = load_data()
    tester = Backtester(cost_bps=COST_BPS)

    print("Signal rationale: residuals reflect model deviations, so mean-reversion is the primary signal; momentum is a robustness check.")

    experiments = build_experiments()
    summary_rows = []
    exp_results = {}

    for exp in experiments:
        net_pnl, metrics, _ = run_walk_forward_strategy(
            exp["name"],
            returns,
            config,
            tester,
            pca_residuals,
            pca_mode=exp["pca_mode"],
            signal=exp["signal"],
            use_kalman=exp.get("use_kalman", False),
            delta=exp.get("delta"),
            ve=exp.get("ve"),
            print_folds=exp.get("print_folds", True),
            print_summary=exp.get("print_summary", True),
        )

        exp_results[exp["name"]] = {
            "pnl": net_pnl,
            "metrics": metrics,
            "kalman": exp.get("use_kalman", False),
        }

        summary_rows.append({
            "Strategy": exp["name"],
            "PCA": exp["pca_mode"],
            "Signal": signal_label(exp["signal"]),
            "Kalman": exp.get("use_kalman", False),
            "Delta": exp.get("delta", np.nan),
            "VE": exp.get("ve", np.nan),
            "Total_Return": metrics["Total_Return"],
            "CAGR": metrics["CAGR"],
            "Sharpe": metrics["Sharpe"],
            "MaxDD": metrics["MaxDD"],
            "VaR_95": metrics["VaR_95"],
            "ES_95": metrics["ES_95"],
            "Avg_Turnover": metrics["Avg_Turnover"],
            "Cost_Impact": metrics["Cost_Impact"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nExperiment Summary:")
    # In run_walk_forward.py
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "Total_Return": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "MaxDD": "{:.2%}".format,
                "VaR_95": "{:.2%}".format,      # NEW
                "ES_95": "{:.2%}".format,       # NEW
                "Avg_Turnover": "{:.3f}".format,
                "Cost_Impact": "{:.2%}".format,
            },
        )
    )
    summary_df.to_csv("data/experiment_results.csv", index=False)
    print("Results saved to data/experiment_results.csv")
    
    # Plotting graphs
    # Base strategies requested for plotting
    base_plot_names = [
        "Fixed PCA | MeanRev",
        "Fixed PCA | Momentum",
        "Rolling PCA | MeanRev",
        "Rolling PCA | Momentum",
    ]

    selected_names = []
    for name in base_plot_names:
        res = exp_results.get(name)
        if res and not res["pnl"].empty:
            selected_names.append(name)

    # Add best/worst Kalman by Sharpe if available
    kalman_df = summary_df[summary_df["Kalman"] == True]
    if not kalman_df.empty:
        best_row = kalman_df.loc[kalman_df["Sharpe"].idxmax()]
        worst_row = kalman_df.loc[kalman_df["Sharpe"].idxmin()]
        for row in (best_row, worst_row):
            name = row["Strategy"]
            res = exp_results.get(name)
            if res and not res["pnl"].empty and name not in selected_names:
                selected_names.append(name)

    os.makedirs("figures", exist_ok=True)

    def short_label(name: str) -> str:
        name = name.replace("Fixed PCA |", "PCA")
        name = name.replace("Rolling PCA |", "RollPCA")
        name = name.replace("Fixed PCA + Kalman |", "PCA+KF")
        name = name.replace("MeanRev", "MR")
        name = name.replace("Momentum", "MOM")
        name = name.replace("ve=", "R=")   # optional: R = observation noise
        name = name.replace("d=", "Q=")    # optional: Q = process noise
        return name

    if selected_names:
        fig, ax = plt.subplots(figsize=(12, 5))

        for name in selected_names:
            res = exp_results[name]
            pnl = res["pnl"]
            if pnl.empty:
                continue

            wealth = np.exp(pnl.cumsum())
            ax.plot(wealth.index, wealth, label=short_label(name), linewidth=1.8)

        ax.set_title("Walk-forward cumulative wealth (selected strategies)")
        ax.set_ylabel("Cumulative wealth")
        ax.grid(True, alpha=0.25)

        # Put legend below plot so it doesn't cover lines
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        # Save performance graph as png
        fig.savefig("figures/wealth_curves.png", dpi=220, bbox_inches="tight")

        plt.show()

if __name__ == "__main__":
    main()
