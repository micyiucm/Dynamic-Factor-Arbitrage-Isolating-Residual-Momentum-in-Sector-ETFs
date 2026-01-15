import jax
import jax.numpy as jnp
from jax import jit, vmap
import pandas as pd
import numpy as np

# Enable 64-bit precision for Kalman stability (JAX defaults to float32)
jax.config.update("jax_enable_x64", True)

@jit # Just-In-Time compile for speed improvements
def run_kalman_static(Y, X, delta, ve):
    """
    Static JIT-compiled Kalman Filter engine.
    This function is compiled once by JAX and reused for every fold, 
    providing speed improvements over pure NumPy implementations.
    Parameters:
        Y: Asset returns matrix, shape (T, N): T time steps, N assets
        X: Factor matrix, shape (T, K): T time steps, K factors
        delta: State transition noise (controls adaptation speed)
        ve: Observation noise variance
    
    Returns:
        residuals_matrix: Shape (T, N)
    """
    T, N = Y.shape # T = number of time steps, N = number of assets
    K = X.shape[1] # K = number of factors
    
    # Initialise Kalman State
    # Beta = factor loadings (how much each factor affects this asset)
    # P = covariance matrix of Beta (uncertainty about loadings)
    initial_beta = jnp.zeros(K) # Start with zero loadings
    initial_P = jnp.eye(K) # Start with identity covariance matrix, high uncertainty

    def kalman_step(carry, data_t):
        """
        Single time step of Kalman Filter.
        This function updates factor loadings (beta) based on new observations,
        tracking how factor sensitivities evolve over time.
        """
        # Previous state estimate
        beta_prev, P_prev = carry
        # Current observations
        y_t, x_t = data_t  # y_t is scalar (returns), x_t is vector (K factors)

        # Predict step:
        # Assume Beta follows a Random Walk: Beta_t = Beta_{t-1} + noise
        beta_pred = beta_prev
        P_pred = P_prev + delta * jnp.eye(K)

        # Update step:
        # Expected Return = Dot product of Factor exposures and Current Beta loadings
        y_pred = jnp.dot(x_t, beta_pred)
        e_t = y_t - y_pred  # Residual/Innovation

        # Compute Kalman Gain (optimal weight between prediction and observation)
        Q_t = jnp.dot(jnp.dot(x_t, P_pred), x_t.T) + ve # Variance of prediction error
        K_gain = jnp.dot(P_pred, x_t.T) / Q_t # Kalman gain

        # Correct state estimate using innvoation
        beta_new = beta_pred + K_gain * e_t # Updated factor loadings
        P_new = P_pred - jnp.dot(jnp.outer(K_gain, x_t), P_pred) # Updated uncertainty

        return (beta_new, P_new), e_t

    def scan_single_asset(y_series, x_matrix):
        """
        Function that runs Kalman Filter for one single asset across all time steps.
        Parameters:
            y_series: Returns for one asset, shape (T,) # 1D array with T elements
            x_matrix: Factor matrix, shape (T, K)
        Returns:
            residuals: Time series of residuals for this asset, shape (T,)
        """
        # Use jax.lax.scan to efficiently loop over T
        _, residuals = jax.lax.scan(
            kalman_step,
            (initial_beta, initial_P),
            (y_series, x_matrix),
        )
        return residuals

    # Vvectorisation over assets
    # VMAP automatically parallelizes computation across all N assets
    # Y has shape (T, N) so vmap processes N columns independently
    # # X has shape (T, K), same factors used for all assets
    residuals_matrix = vmap(scan_single_asset, in_axes=(1, None))(Y, X)
    
    # vmap returns shape (N, T), transpose back to (T, N)
    return residuals_matrix.T 

class KalmanFilterJAX:
    """
    Kalman Filter for adaptive factor model estimation.
    Estimates time-varying factor loadings (betas) that adapt to regime changes,
    producing residuals that account for evolving market structure.
    """
    def __init__(self, n_factors: int = 3, delta: float = 1e-3, ve: float = 1e-3):
        """
        Initialize Kalman Filter parameters.
        Parameters:
            n_factors: Number of principal components to use (default to 3)
            delta: State transition noise (higher = faster adaptation, more noise)
            ve: Observation noise variance (measurement error)
        """
        self.k = n_factors
        self.delta = delta
        self.ve = ve

    def fit_transform(self, returns: pd.DataFrame, pca_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Run Kalman filter to generate adaptive residuals.
        Parameters:
            returns: DataFrame of asset returns, shape (T, N)
            pca_factors: DataFrame of PCA factors, shape (T, K)
        Returns:
            DataFrame of residuals with same shape as returns
        """
        # Convert pandas DataFrame to JAX arrays
        Y = jnp.array(returns.values)
        X = jnp.array(pca_factors.values)
        
        # Validate input dimensions
        if X.shape[1] != self.k:
            raise ValueError(f"Expected {self.k} factors, got {X.shape[1]}")

        print(f"   [JAX] Running Filter (N={Y.shape[1]}, T={Y.shape[0]}, Delta={self.delta})...")
        
        # Call the JIT-compiled static function
        # First call compiles slowly, subsequent calls are instant
        residuals_matrix = run_kalman_static(Y, X, self.delta, self.ve)
        
        # Convert back to pandas DataFrame
        return pd.DataFrame(
            np.array(residuals_matrix),
            index=returns.index,
            columns=returns.columns,
        )

if __name__ == "__main__":
    print("KalmanFilterJAX module loaded (use via src/run_walk_forward.py).")
