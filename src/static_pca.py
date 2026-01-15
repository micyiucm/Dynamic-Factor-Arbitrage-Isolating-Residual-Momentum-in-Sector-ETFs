import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Configuration
DATA_PATH = "data/etf_returns.csv"
CONFIG_PATH = "data/walk_forward_config.csv"
FIXED_OUT_PATH = "data/pca_residuals_fixed_etf.csv"
ROLLING_OUT_PATH = "data/pca_residuals_rolling_etf.csv"
N_FACTORS = 3
ROLLING_WINDOW = 60

def rolling_pca_residuals(
    returns: pd.DataFrame,
    window: int,
    n_components: int,
) -> pd.DataFrame:
    """
    Compute PCA residuals using a rolling window approach.
    
    At each time t, fits PCA on the previous 'window' days, then calculates
    the residual (actual return minus PCA-reconstructed return) for day t.
    This is an adaptive method that continuously updates factor loadings.
    
    Parameters:
        returns: DataFrame of log returns (n_days x n_assets)
        window: Number of historical days to fit PCA on
        n_components: Number of principal components to extract
    
    Returns:
        DataFrame of residuals with same shape as input (first 'window' rows are NaN)
    """
    data = returns.values
    timestamps = returns.index
    tickers = returns.columns
    n_samples, n_assets = data.shape
    k = min(n_components, n_assets)

    # Initialise residuals array filled with NaNs with same shape as data
    residuals = np.full_like(data, np.nan, dtype=float)

    print(f"Running Rolling PCA (Window={window}, Factors={k})...")

    # Rolling window PCA loop
    for t in range(window, n_samples):
        # Extract the sliding window
        window_data = data[t - window : t]
        # Fit PCA on historical window
        pca = PCA(n_components=k)
        pca.fit(window_data)
        # Get return of day t (current day) and reshape for sklearn
        current_return = data[t].reshape(1, -1)
        # Project current return onto factor space
        factors = pca.transform(current_return)
        # Reconstruct expected return from factors
        expected_return = np.dot(factors, pca.components_) + pca.mean_
        # Residual = Actual returns - Expected returns
        residuals[t] = current_return - expected_return

    return pd.DataFrame(residuals, index=timestamps, columns=tickers)

def fixed_pca_residuals_by_fold(
    returns: pd.DataFrame,
    config: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    """
    Compute PCA residuals using fixed factors per walk-forward fold.
    
    For each fold: fits PCA on training period, then applies those fixed
    loadings to the test period. This prevents lookahead bias while
    maintaining stable factor definitions within each test period.
    
    Parameters:
        returns: DataFrame of log returns (n_days x n_assets)
        config: Walk-forward configuration with Train_Start, Train_End, 
                Test_Start, Test_End columns
        n_components: Number of principal components to extract
    
    Returns:
        DataFrame of residuals (only test periods are filled, training periods are NaN)
    """
    # Edge case: Ensures number of PCs is less than number of assets
    k = min(n_components, returns.shape[1])
    # Initialise residuals dataframe with same shape as returns
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    print(f"Running Fixed PCA by Fold (Factors={k})...")

    # Fixed PCA loop over each walk-forward fold
    for _, row in config.iterrows():
        # Extract fold date boundaries from configuration
        train_start, train_end = row["Train_Start"], row["Train_End"]
        test_start, test_end = row["Test_Start"], row["Test_End"]
        # Slice returns data for this particular fold
        train_data = returns.loc[train_start:train_end]
        test_data = returns.loc[test_start:test_end]
        # Skip fold if training/test data is missing
        if train_data.empty or test_data.empty:
            continue
        # Fit PCA on training data (learns factor structure from historical data, no look-ahead bias)
        pca = PCA(n_components=k)
        pca.fit(train_data)
        # Transform test data into factor space using training PC loadings
        test_factors = pca.transform(test_data)
        # Reconstruct expected returns from factors
        test_reconstructed = np.dot(test_factors, pca.components_) + pca.mean_
        # Calculate residuals for test period
        residuals.loc[test_data.index] = test_data - test_reconstructed

    return residuals

def main():
    print("--- Loading ETF Data... ---")
    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    config = pd.read_csv(
        CONFIG_PATH,
        parse_dates=["Train_Start", "Train_End", "Test_Start", "Test_End"],
    )

    print(f"Data Shape: {returns.shape}")

    fixed_residuals = fixed_pca_residuals_by_fold(returns, config, n_components=N_FACTORS)
    rolling_residuals = rolling_pca_residuals(
        returns,
        window=ROLLING_WINDOW,
        n_components=N_FACTORS,
    )

    fixed_residuals.to_csv(FIXED_OUT_PATH)
    rolling_residuals.to_csv(ROLLING_OUT_PATH)
    print(f"Saved fixed PCA residuals to {FIXED_OUT_PATH}")
    print(f"Saved rolling PCA residuals to {ROLLING_OUT_PATH}")

if __name__ == "__main__":
    main()
