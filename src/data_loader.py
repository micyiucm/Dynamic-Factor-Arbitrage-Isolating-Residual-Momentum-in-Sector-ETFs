import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import List

def get_liquid_etfs() -> List[str]:
    """
    Function that returns a list containing the classic 9 SPDR Select Sector ETFs + SPY.
    Excludes XLRE (launched Oct 7, 2015) and XLC (launched June 18, 2018) to ensure consistent history of data (>10 years).
    """
    return ['XLE', 'XLB', 'XLI', 'XLY', 'XLP', 'XLV', 'XLF', 'XLK', 'XLU', 'SPY']

def download_etf_data(start_date: str = "2015-01-01", 
                      end_date: str = "2026-01-01") -> pd.DataFrame:
    """
    Function that downloads adjusted close prices of liquid sector ETFs + SPY with pinned dates for reproducibility.
    Parameters:
        start_date (str): a string of the start date of the dataset
        end_date (str): a string of the end date of the dataset
    Output: A pandas DataFrame of adjusted close prices of liquid ETFs
    """
    tickers = get_liquid_etfs()
    
    # Download data
    print(f"Downloading data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)
    
    # Handle MultiIndex output from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']] # Edge case: tickers contains only 1 element
    
    # Sort index to ensure prices are in chronological order
    prices = prices.sort_index()

    before_len = len(prices)
    prices = prices.dropna(how='any') # If any asset is missing, drop the row
    dropped_rows = before_len - len(prices)
    
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows containing missing values to maintain data integrity.")
    else:
        print("No missing values found.")
    
    print(f"Final dataset: {prices.shape[1]} assets over {prices.shape[0]} days.")
    return prices

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate log returns from simple returns
    Parameters:
        prices (pd.DataFrame): A pandas DataFrame containing adjusted close prices
    Output: A pandas DataFrame containing log_returns instead of prices
    Math:
        log returns at time t r_t = log(P_t/P_{t-1}) = log(P_t)-log(P_{t-1})
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna() # Drops first row containing NaNs
    return log_returns

def generate_walk_forward_validation_split(returns: pd.DataFrame, 
                                   train_years: int = 3, 
                                   test_years: int = 1) -> pd.DataFrame:
    """
    Function to generate walk_forward validation splits using calendar-based time windows.
    Parameters:
        returns (pd.DataFrame): DataFrame of log returns with DateTime Index.
        train_years (int, optional): Number of years in each training window.
        test_years (int, optional): Number of years in each testing window.
    Output: A pandas DataFrame containing splitted data with columns:
        Fold (int): Fold identifier, 1-indexed
        Train_Start (Timestamp): First trading day in training period
        Train_End (Timestamp): Last trading day in training period
        Test_Start (Timestamp): First trading day in testing period
        Test_End (Timestamp): Last trading day in testing period
    """
    folds = []
    
    # Start from the first available date in the dataset
    current_start = returns.index[0]
    fold_id = 1
    
    while True:
        # Define the exact calendar cutoff points
        train_end_date = current_start + pd.DateOffset(years=train_years)
        test_end_date = train_end_date + pd.DateOffset(years=test_years)
        
        # Stop if the test set extends beyond available data
        if test_end_date > returns.index[-1]:
            break
            
        # Create Boolean masks to select data
        # Train: [start, train_end) -> Inclusive start, Exclusive end
        train_mask = (returns.index >= current_start) & (returns.index < train_end_date)
        # Test: [train_end, test_end) -> Inclusive start, Exclusive end
        test_mask = (returns.index >= train_end_date) & (returns.index < test_end_date)
        
        # Get the actual min/max dates that exist in the data for logging
        train_idx = returns.index[train_mask]
        test_idx = returns.index[test_mask]
        
        # Skip current fold if train_idx or test_idx is empty and move forward by test_years
        if len(train_idx) == 0 or len(test_idx) == 0:
            current_start += pd.DateOffset(years=test_years)
            continue

        # Store fold configuration as dictionary
        folds.append({
            'Fold': fold_id,
            'Train_Start': train_idx[0],
            'Train_End': train_idx[-1],
            'Test_Start': test_idx[0],
            'Test_End': test_idx[-1]
        })
        
        print(f"Fold {fold_id}: "
              f"Train[{train_idx[0].date()} -> {train_idx[-1].date()}] "
              f"Test[{test_idx[0].date()} -> {test_idx[-1].date()}]")
        
        # Roll forward by the test window size
        current_start += pd.DateOffset(years=test_years)
        fold_id += 1
        
    return pd.DataFrame(folds)

def save_data(prices, returns, splits, output_dir='data'):
    """
    Function to save processed ETF data and walk-forward split configuration to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    prices.to_csv(f'{output_dir}/etf_prices.csv')
    returns.to_csv(f'{output_dir}/etf_returns.csv')
    splits.to_csv(f'{output_dir}/walk_forward_config.csv', index=False)
    print(f"\nData and Config saved to {output_dir}/")

def main():
    print("=" * 80)
    print("Liquid ETF Data Loader (Calendar Walk-Forward)")
    print("=" * 80)
    

    # Explicitly set the end date to ensure reproducible results
    prices = download_etf_data(start_date="2012-01-01", end_date="2026-01-01")
    
    returns = calculate_log_returns(prices)
    
    # Generate Splits
    splits = generate_walk_forward_validation_split(returns, train_years=3, test_years=1)
    
    if not splits.empty:
        save_data(prices, returns, splits)
    else:
        print("Error: Not enough data to generate folds.")

if __name__ == "__main__":
    main()
