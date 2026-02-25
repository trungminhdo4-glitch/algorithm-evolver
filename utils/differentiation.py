import numpy as np
from scipy.signal import savgol_filter

def compute_derivative(t, y, method='savgol', window_length=11, polyorder=3):
    """
    Computes the derivative dy/dt from noisy data.
    
    Args:
        t (array): Time steps.
        y (array): State variable values.
        method (str): 'savgol' or 'gradient'.
        window_length (int): Savitzky-Golay window size (must be odd).
        polyorder (int): Polynomial order for smoothing.
        
    Returns:
        dy_dt (array): Estimated derivatives.
    """
    if method == 'savgol':
        # Smooth the data first
        y_smoothed = savgol_filter(y, window_length, polyorder)
        # Compute gradient on smoothed data
        # Note: np.gradient uses second order accurate central differences in the interior
        # and first or second order accurate one-sides (forward or backwards) differences at the boundaries.
        dt = np.diff(t)
        if np.allclose(dt, dt[0]):
            # Uniform spacing
            dy_dt = np.gradient(y_smoothed, dt[0])
        else:
            # Non-uniform spacing
            dy_dt = np.gradient(y_smoothed, t)
    else:
        # Raw gradient (very noisy)
        dy_dt = np.gradient(y, t)
        
    return dy_dt
