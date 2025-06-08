# Usage:
#   python chap16_experiment_16C.py
#   python  chap16_experiment_16C.py --seed 123  # reproducible run

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter  # For potentially smoother derivatives
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

# Attempt to import optional libraries used for advanced fractal analysis
try:  # pragma: no cover - optional dependency
    from MFDFA import MFDFA
    MFDFA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MFDFA_AVAILABLE = False

# nolds is optional.  If unavailable we fall back to simplified Hurst
try:
    import nolds  # type: ignore
    NOLDS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NOLDS_AVAILABLE = False

# Flag returned by Hurst exponent calculators when a time series is degenerate
HURST_DEGENERATE_FLAG = -1.0
# Global debug flag for fractal calculations
DEBUG_FRACTAL_CALCS = False
# Minimum length for reliable Hurst/MFDFA calculations
MIN_HURST_LEN = 30
EPS_STD = 1e-12           # was hard-coded 1e-9 everywhere
# --- Simulation Configuration ---
# Number of Monte Carlo runs for each parameter set
# Reduced for quicker iterations during development
N_MC_RUNS = 3

# Base TD model parameters shared across scenarios
base_td_params = {
    'C': 1.0,
    'w1': 0.33,
    'w2': 0.34,
    'w3': 0.33,
    'g_initial': 1.0,
    'beta_initial': 1.0,
    'f_crit_initial': 10.0,
    'strain_initial': 0.5,
    # Simulation time step.  Smaller values generate longer time series which
    # can improve fractal estimates at the cost of slower runs.
    'dt': 0.05,
    'couple_index_window': 50,
    'phi1': 0.7,
    'f_crit_min_collapse': 0.1,
    'f_crit_min_value': 1e-3,
    'beta_min_value': 1e-3,
    'g_min_value': 1e-3,
    'g_sensitivity': 0.1,
    'beta_sensitivity': 0.05,
    'g_cost_factor': 0.01,
    'beta_cost_factor': 0.005,
    'fatigue_rate': 0.0,
    'lever_noise_std': 0.0,
    # New adaptive response parameters
    'g_lever_sensitivity_to_strain_ratio': 0.1,
    'beta_lever_sensitivity_to_strain_ratio': 0.05,
    'g_lever_cost_factor': 0.01,
    'beta_lever_cost_factor': 0.005,
    'f_crit_fatigue_rate': 0.0,
    'savgol_window': 7,
    'savgol_polyorder': 2,
    'critical_theta_factor': 0.5,
    'hurst_n_lags': 20,
    # Pre-collapse window parameters
    'precollapse_window_percent': 0.3,
    'precollapse_window_duration': 10.0,
    'min_pre_collapse_window_points': 25,
    # Lever adaptation parameters
    'g_adaptation_rate': 0.3,
    'beta_adaptation_rate': 0.3,
    'lever_value_noise_std': 1e-5,
    # Rolling window configuration for temporal fractal metrics
    'rolling_window_points': 50,
    'rolling_stride': 1,
}

# Scenario definitions with parameter sweepseps

# Large parameter sweet for final analysis
scenarios_config_5000_sims = {
    'slow_creep': {
        'base': {
            'type': 'slow_creep',
            'strain_growth_rate': 0.007,
            'max_time': 500.0,
        },
        'sweep': {
            'fcrit_decay_rate': [0.02, 0.03, 0.04],     # (3 options) - Key driver
            'f_crit_initial': [8.0, 10.0, 12.0],       # (3 options) - Initial resource
            'g_adaptation_rate': [0.2, 0.3, 0.4],      # (3 options) - Adaptation speed
            'beta_adaptation_rate': [0.2, 0.3, 0.4],   # (3 options) - Adaptation speed
            'w_config': [                              # (3 options) - Resilience strategy
                (0.33, 0.34, 0.33), # Balanced
                (0.5, 0.25, 0.25),  # w1-dominant
                (0.25, 0.25, 0.5),  # w3-dominant (Fcrit-dominant)
            ],
            'phi1': [0.5, 0.7, 0.9],                   # (3 options) - Cost of gain
            # Fixed parameters for this scenario to limit combinations:
            'g_initial': [1.0],
            'beta_initial': [1.0],
            'couple_index_window': [50],
            'g_lever_cost_factor': [0.01],
            'beta_lever_cost_factor': [0.005],
        },
    },
    'sudden_shock': {
        'base': {
            'type': 'sudden_shock',
            'shock_time': 15.0,
            'fcrit_post_shock_decay': 0.015,
            'beta_post_shock_drift': 0.007,
            'max_time': 100.0,
        },
        'sweep': {
            'fcrit_shock_factor': [0.1, 0.2, 0.3],     # (3 options) - Shock severity
            'strain_shock_factor': [3.0, 4.5, 6.0],    # (3 options) - Shock severity
            'g_adaptation_rate': [0.2, 0.4, 0.6],      # (3 options) - Post-shock adaptation
            'beta_adaptation_rate': [0.2, 0.4, 0.6],   # (3 options) - Post-shock adaptation
            'w_config': [                              # (2 options) - Resilience strategy
                (0.33, 0.34, 0.33), # Balanced
                (0.2, 0.5, 0.3),    # w2-dominant (beta-dominant)
            ],
            'phi1': [0.5, 0.7, 0.9],                   # (3 options) - Cost of gain
            # Fixed parameters for this scenario:
            'g_initial': [1.0], # Using base_td_params
            'beta_initial': [1.0],# Using base_td_params
            'f_crit_initial': [10.0],# Using base_td_params
        },
    },
    'tightening_loop': {
        'base': {
            'type': 'tightening_loop',
            'fcrit_base_decay_tightening': 0.02,
            'max_time': 500.0,
        },
        'sweep': {
            'fcrit_beta_cost_factor': [0.01, 0.015, 0.02], # (3 options) - Key driver
            'strain_tightening_feedback': [0.002, 0.003, 0.004], # (3 options) - Key driver
            'g_adaptation_rate': [0.2, 0.4, 0.6],      # (3 options) - Adaptation speed
            'beta_adaptation_rate': [0.2, 0.4, 0.6],   # (3 options) - Adaptation speed
            'w_config': [                              # (2 options) - Resilience strategy
                (0.33, 0.34, 0.33), # Balanced
                (0.4, 0.3, 0.3),    # Slightly different balance
            ],
            'phi1': [0.5, 0.7, 0.9],                   # (3 options) - Cost of gain
            # Fixed parameters for this scenario:
            'g_initial': [1.0],
            'beta_initial': [1.0],
            'f_crit_initial': [10.0],
        },
    },
}
# N_MC_RUNS is assumed to be 3 from your previous context.
# Target unique combinations: ~1000 / 3 = ~333, smaller sweep for faster iteration on the code

scenarios_config = {
    'slow_creep': {
        'base': {
            'type': 'slow_creep',
            # Smaller growth rate and much longer max_time for richer series
            'strain_growth_rate': 0.005,
            'max_time': 1500.0,
        },
        'sweep': {
            'fcrit_decay_rate': [0.01, 0.02],
            'strain_growth_rate': [0.003, 0.005],
            'f_crit_initial': [8.0, 12.0],
            'g_adaptation_rate': [0.2, 0.4],
            'beta_adaptation_rate': [0.2, 0.4],
            'w_config': [                               # (3 options) - Kept
                (0.33, 0.34, 0.33), # Balanced
                (0.5, 0.25, 0.25),  # w1-dominant
                (0.25, 0.25, 0.5),  # w3-dominant (Fcrit-dominant)
            ],
            'phi1': [0.5, 0.7],                         # (2 options) - Kept
            'lever_value_noise_std': [1e-5, 5e-5],
            # Fixed parameters (as in your _3500_sims config):
            'g_initial': [1.0], 'beta_initial': [1.0], 'couple_index_window': [50],
            'g_lever_cost_factor': [0.01], 'beta_lever_cost_factor': [0.005],
        },
    },
    'sudden_shock': {
        'base': {
            'type': 'sudden_shock',
            'shock_time': 15.0,
            'fcrit_post_shock_decay': 0.015,
            'beta_post_shock_drift': 0.007,
            # Slightly longer max_time to capture post-shock dynamics
            'max_time': 150.0,
        },
        'sweep': {
            'fcrit_shock_factor': [0.1, 0.3],          # (3 -> 2 options) Picked min/max
            'strain_shock_factor': [3.0, 6.0],         # (3 -> 2 options) Picked min/max
            'g_adaptation_rate': [0.2, 0.4, 0.6],      # (3 options) - Kept to maintain some variation
            'beta_adaptation_rate': [0.2, 0.4, 0.6],   # (3 options) - Kept
            'w_config': [                               # (2 options) - Kept
                (0.33, 0.34, 0.33), # Balanced
                (0.2, 0.5, 0.3),    # w2-dominant (beta-dominant)
            ],
            'phi1': [0.5, 0.9],                         # (2 options) - Kept
            # Fixed parameters (as in your _3500_sims config):
            'g_initial': [1.0], 'beta_initial': [1.0], 'f_crit_initial': [10.0],
        },
    },
    'tightening_loop': {
        'base': {
            'type': 'tightening_loop',
            'fcrit_base_decay_tightening': 0.015,
            'max_time': 1500.0,
        },
        'sweep': {
            'fcrit_beta_cost_factor': [0.01, 0.02],
            'strain_tightening_feedback': [0.001, 0.002],
            'fcrit_base_decay_tightening': [0.01, 0.02],
            'g_adaptation_rate': [0.2, 0.4, 0.6],
            'beta_adaptation_rate': [0.2, 0.4, 0.6],
            'w_config': [                               # (2 options) - Kept
                (0.33, 0.34, 0.33), # Balanced
                (0.4, 0.3, 0.3),    # Slightly different balance
            ],
            'phi1': [0.7, 0.9],                         # (2 options) - Kept
            'lever_value_noise_std': [1e-5, 5e-5],
            # Fixed parameters (as in your _3500_sims config):
            'g_initial': [1.0], 'beta_initial': [1.0], 'f_crit_initial': [10.0],
        },
    },
}


# ------------------------------------------------------------------
# ultra-fast sweep for testing: one value per knob, shorter runs
# ------------------------------------------------------------------
scenarios_config_fast = {
    "slow_creep": {
        "base": {
            "type": "slow_creep",
            "strain_growth_rate": 0.005,
            "max_time": 1500.0,          # was 1500
        },
        "sweep": {
            "fcrit_decay_rate":            [0.01],
            "strain_growth_rate":          [0.003],
            "f_crit_initial":              [10.0],
            "g_adaptation_rate":           [0.3],
            "beta_adaptation_rate":        [0.3],
            "w_config":                    [(0.33, 0.34, 0.33)],
            "phi1":                        [0.7],
            "lever_value_noise_std":       [1e-5],
            # “fixed” parameters still need to be lists because the generator
            # code expects iterables.
            "g_initial":                   [1.0],
            "beta_initial":                [1.0],
            "couple_index_window":         [50],
            "g_lever_cost_factor":         [0.01],
            "beta_lever_cost_factor":      [0.005],
        },
    },

    "sudden_shock": {
        "base": {
            "type": "sudden_shock",
            "shock_time": 15.0,
            "fcrit_post_shock_decay": 0.015,
            "beta_post_shock_drift": 0.007,
            "max_time": 60.0,            # was 150
        },
        "sweep": {
            "fcrit_shock_factor":          [0.3],
            "strain_shock_factor":         [6.0],
            "g_adaptation_rate":           [0.4],
            "beta_adaptation_rate":        [0.4],
            "w_config":                    [(0.33, 0.34, 0.33)],
            "phi1":                        [0.5],
            "g_initial":                   [1.0],
            "beta_initial":                [1.0],
            "f_crit_initial":              [10.0],
        },
    },

    "tightening_loop": {
        "base": {
            "type": "tightening_loop",
            "fcrit_base_decay_tightening": 0.015,
            "max_time": 150.0,           # was 1500
        },
        "sweep": {
            "fcrit_beta_cost_factor":      [0.02],
            "strain_tightening_feedback":  [0.002],
            "fcrit_base_decay_tightening": [0.02],
            "g_adaptation_rate":           [0.4],
            "beta_adaptation_rate":        [0.4],
            "w_config":                    [(0.33, 0.34, 0.33)],
            "phi1":                        [0.9],
            "lever_value_noise_std":       [1e-5],
            "g_initial":                   [1.0],
            "beta_initial":                [1.0],
            "f_crit_initial":              [10.0],
        },
    },
}


# --- Helper Functions ---

def calculate_theta_t(g_lever, beta_lever, f_crit, C, w1, w2, w3):
    """Calculates the Tolerance Sheet value."""
    if g_lever <= 0 or beta_lever <= 0 or f_crit <= 0:
        return 1e-9 # Avoid log(0) or division by zero, effectively a breached state
    return C * (g_lever**w1) * (beta_lever**w2) * (f_crit**w3)

def get_velocities(lever_series, dt):
    """Calculates velocities using simple finite difference (backward)."""
    if len(lever_series) < 2:
        return np.zeros_like(lever_series)
    # Using np.gradient for a more robust centered difference where possible
    velocities = np.gradient(lever_series, dt)
    return velocities

def calculate_speed_index(beta_dot, f_crit_dot):
    """Calculates the Speed Index."""
    return np.sqrt(beta_dot**2 + f_crit_dot**2)

def calculate_couple_index(beta_dot_series, f_crit_dot_series, window):
    """Calculates the Couple Index over a rolling window."""
    if len(beta_dot_series) < window or len(f_crit_dot_series) < window:
        return np.nan # Not enough data for correlation

    s1 = pd.Series(beta_dot_series)
    s2 = pd.Series(f_crit_dot_series)

    chunk_beta = s1.iloc[-window:]
    chunk_fcrit = s2.iloc[-window:]
    if chunk_beta.std() < 1e-7 or chunk_fcrit.std() < 1e-7:
        return np.nan

    rolling_corr = s1.rolling(
        window=window,
        min_periods=max(window // 2, window - 2),
    ).corr(s2)
    return rolling_corr.iloc[-1] if not rolling_corr.empty else np.nan


# --- Fractal Analysis Functions (Simplified) ---

def hurst_exponent_simplified(ts):
    """Simplified R/S analysis for Hurst exponent."""
    ts = np.asarray(ts)
    if len(ts) < MIN_HURST_LEN or np.std(ts) < EPS_STD:
        return np.nan
    if np.std(ts) < EPS_STD or np.all(ts == ts[0]):
        if DEBUG_FRACTAL_CALCS:
            print(f"Degenerate Hurst input (simplified), std={np.std(ts):.3e}")
        return HURST_DEGENERATE_FLAG
    
    N = len(ts)
    min_chunk_size = 10
    max_chunk_size = N // 2
    if min_chunk_size >= max_chunk_size:
        return np.nan
        
    chunk_sizes = np.logspace(np.log10(min_chunk_size), np.log10(max_chunk_size), num=10, dtype=int)
    chunk_sizes = np.unique(chunk_sizes) # Ensure unique sizes
    if len(chunk_sizes) < 2: return np.nan

    rs_values = []
    for chunk_size in chunk_sizes:
        if chunk_size == 0: continue
        num_chunks = N // chunk_size
        if num_chunks == 0: continue
        
        all_chunk_rs = []
        for i in range(num_chunks):
            chunk = ts[i*chunk_size : (i+1)*chunk_size]
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumulative_deviations = np.cumsum(deviations)
            range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            std_dev = np.std(chunk)
            if std_dev > EPS_STD: # Avoid division by zero
                all_chunk_rs.append(range_val / std_dev)
        
        if all_chunk_rs:
            rs_values.append(np.mean(all_chunk_rs))
        else: # if no valid chunks (e.g. all std_dev were zero)
             rs_values.append(np.nan)


    valid_indices = ~np.isnan(rs_values)
    if np.sum(valid_indices) < 2: return np.nan

    valid_chunk_sizes = chunk_sizes[valid_indices]
    valid_rs_values = np.array(rs_values)[valid_indices]
    
    if len(valid_chunk_sizes) < 2 or len(valid_rs_values) < 2: return np.nan

    try:
        log_lags = np.log(valid_chunk_sizes)
        log_rs = np.log(valid_rs_values)
        H, _ = np.polyfit(log_lags, log_rs, 1)
        return H
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def hurst_exponent(ts, n_lags=20):
    """Return Hurst exponent using nolds if available."""
    ts = np.asarray(ts)
    if len(ts) < MIN_HURST_LEN or np.std(ts) < EPS_STD:
        if DEBUG_FRACTAL_CALCS:
            print(f"Skipping Hurst: len={len(ts)}, std={np.std(ts):.3e}")
        return np.nan
    if np.std(ts) < EPS_STD or np.all(ts == ts[0]):
        if DEBUG_FRACTAL_CALCS:
            print(f"Degenerate Hurst input, std={np.std(ts):.3e}")
        return HURST_DEGENERATE_FLAG

    if NOLDS_AVAILABLE and len(ts) >= MIN_HURST_LEN:
        try:
            return nolds.hurst_rs(ts, fit="RANSAC", n_lags=n_lags)
        except Exception:
            # nolds failed – use the built-in R/S estimator instead
            return hurst_exponent_simplified(ts)

    # if nolds is missing or len < MIN_HURST_LEN we fall through to the same helper
    return hurst_exponent_simplified(ts)



def correlation_dimension(ts, emb_dim=2):
    """Compute correlation dimension using nolds if available."""
    ts = np.asarray(ts)
    if len(ts) < MIN_HURST_LEN or np.std(ts) < EPS_STD:
        return np.nan
    if NOLDS_AVAILABLE:
        try:
            return nolds.corr_dim(ts, emb_dim=emb_dim)
        except Exception:
            pass  # fall through
    return np.nan  # or your own simplified CD estimator if you add one


def mfdfa_metrics(ts, q=2):
    """Return MFDFA generalized H(q) and spectrum width if package available."""
    ts = np.asarray(ts)
    if len(ts) < MIN_HURST_LEN or np.std(ts) < EPS_STD:
        if DEBUG_FRACTAL_CALCS:
            print(f"Skipping MFDFA: len={len(ts)}, std={np.std(ts):.3e}")
        return np.nan, np.nan
    if not MFDFA_AVAILABLE:
        return np.nan, np.nan
    try:
        q_vals = np.arange(-5, 6)
        lag, fluct, Hq = MFDFA(ts, q=q_vals)
        hq2 = Hq[list(q_vals).index(2)] if 2 in q_vals else np.nan
        alpha, f = MFDFA.falpha(q_vals, Hq)
        width = np.max(alpha) - np.min(alpha)
        return hq2, width
    except Exception:
        return np.nan, np.nan


def box_counting_dimension_2d_simplified(trajectory_beta_dot, trajectory_f_crit_dot, num_box_sizes=10):
    """Simplified 2D box-counting dimension for a trajectory.

    Notes
    -----
    This implementation is intentionally lightweight and primarily
    illustrative.  For rigorous research applications consider using
    dedicated fractal geometry libraries or algorithms described in the
    literature which provide more stable box-size selection and bias
    correction.
    """
    if len(trajectory_beta_dot) < 2:
        return np.nan

    points = np.array([trajectory_beta_dot, trajectory_f_crit_dot]).T
    
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    if np.all(min_coords == max_coords): # Single point trajectory
        return 0.0

    # Determine range of box sizes
    # Smallest box size should be larger than smallest distance between points, roughly
    # Largest box size should cover the whole trajectory
    # For simplicity, let's use powers of 2 related to overall range
    
    max_range = np.max(max_coords - min_coords)
    if max_range == 0: return 0.0

    counts = []
    box_sizes_used = []

    # Iterate over a range of box sizes (scales)
    # Start with a box size that covers roughly 1/2 of the max_range down to a small fraction
    min_epsilon_factor = 0.01 # Smallest box relative to max_range
    max_epsilon_factor = 0.5   # Largest box relative to max_range
    
    epsilons = np.logspace(np.log10(min_epsilon_factor * max_range), 
                           np.log10(max_epsilon_factor * max_range), 
                           num_box_sizes)
    epsilons = np.unique(epsilons[epsilons > 1e-6]) # Ensure positive and unique

    if len(epsilons) < 2: return np.nan

    for epsilon in epsilons:
        if epsilon == 0: continue
        # Determine grid based on current box size
        # Normalize points to a grid to count occupied boxes
        # Shift points so min_coords is at (0,0) then scale by 1/epsilon
        normalized_points = (points - min_coords) / epsilon
        # Get integer coordinates for boxes
        box_indices = np.floor(normalized_points).astype(int)
        
        # Count unique boxes
        unique_boxes = set(map(tuple, box_indices))
        counts.append(len(unique_boxes))
        box_sizes_used.append(epsilon)

    if len(counts) < 2: return np.nan

    # Linear regression in log-log space
    log_counts = np.log(counts)
    log_box_sizes = np.log(1.0 / np.array(box_sizes_used)) # log(1/epsilon) = -log(epsilon)
    
    try:
        dimension, _ = np.polyfit(log_box_sizes, log_counts, 1)
        return dimension
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def box_counting_dimension_3d(trajectory_g_dot, trajectory_beta_dot, trajectory_f_crit_dot, num_box_sizes=10):
    """Simplified 3D box-counting dimension for a trajectory."""
    if len(trajectory_g_dot) < 2:
        return np.nan

    points = np.vstack([trajectory_g_dot, trajectory_beta_dot, trajectory_f_crit_dot]).T

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    if np.all(min_coords == max_coords):
        return 0.0

    max_range = np.max(max_coords - min_coords)
    if max_range == 0:
        return 0.0

    counts = []
    box_sizes_used = []
    min_epsilon_factor = 0.01
    max_epsilon_factor = 0.5
    epsilons = np.logspace(
        np.log10(min_epsilon_factor * max_range),
        np.log10(max_epsilon_factor * max_range),
        num_box_sizes,
    )
    epsilons = np.unique(epsilons[epsilons > 1e-6])
    if len(epsilons) < 2:
        return np.nan

    for epsilon in epsilons:
        normalized = (points - min_coords) / epsilon
        indices = np.floor(normalized).astype(int)
        unique_boxes = set(map(tuple, indices))
        counts.append(len(unique_boxes))
        box_sizes_used.append(epsilon)

    if len(counts) < 2:
        return np.nan

    log_counts = np.log(counts)
    log_box_sizes = np.log(1.0 / np.array(box_sizes_used))
    try:
        dimension, _ = np.polyfit(log_box_sizes, log_counts, 1)
        return dimension
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# --- System Dynamics Simulation Core ---
class TDSystem:
    def __init__(self, params):
        self.params = params
        self.g_lever = params['g_initial']
        self.beta_lever = params['beta_initial']
        self.f_crit = params['f_crit_initial']
        self.strain_avg = params['strain_initial']
        
        self.C = params['C']
        self.w1, self.w2, self.w3 = params['w1'], params['w2'], params['w3']
        
        self.time = 0
        self.dt = params['dt']

        # Optional smoothing parameters for velocity calculation
        self.savgol_window = params.get('savgol_window', 5)
        self.savgol_polyorder = params.get('savgol_polyorder', 2)
        self.critical_theta_factor = params.get('critical_theta_factor', 0.5)
        
        self.history = {
            'time': [], 'g_lever': [], 'beta_lever': [], 'f_crit': [],
            'g_lever_dot': [], 'beta_lever_dot': [], 'f_crit_dot': [],
            'theta_t': [], 'strain_avg': [], 'speed_index': [], 'couple_index': []
        }
        self.is_collapsed = False
        self.collapse_time = np.nan
        self.critical_dynamics_onset_time = np.nan
        self.critical_onset_defined = False
        self.initial_theta_t = calculate_theta_t(self.g_lever, self.beta_lever, self.f_crit, 
                                                 self.C, self.w1, self.w2, self.w3)


    def record_state(self):
        self.history['time'].append(self.time)
        self.history['g_lever'].append(self.g_lever)
        self.history['beta_lever'].append(self.beta_lever)
        self.history['f_crit'].append(self.f_crit)
        
        # Velocities (calculated based on last two points)
        if len(self.history['g_lever']) > 1:
            g_dot = (self.history['g_lever'][-1] - self.history['g_lever'][-2]) / self.dt
            beta_dot = (self.history['beta_lever'][-1] - self.history['beta_lever'][-2]) / self.dt
            f_crit_dot = (self.history['f_crit'][-1] - self.history['f_crit'][-2]) / self.dt
        else:
            g_dot, beta_dot, f_crit_dot = 0, 0, 0
            
        self.history['g_lever_dot'].append(g_dot)
        self.history['beta_lever_dot'].append(beta_dot)
        self.history['f_crit_dot'].append(f_crit_dot)
        
        current_theta_t = calculate_theta_t(self.g_lever, self.beta_lever, self.f_crit,
                                            self.C, self.w1, self.w2, self.w3)
        self.history['theta_t'].append(current_theta_t)
        self.history['strain_avg'].append(self.strain_avg)

        # Speed and Couple Index
        couple_window = self.params.get('couple_index_window', 20)
        if len(self.history['beta_lever_dot']) >= couple_window:
             # Ensure enough data points for couple index calculation
            speed = calculate_speed_index(beta_dot, f_crit_dot)
            # Need series for couple index
            beta_dot_series = self.history['beta_lever_dot']
            f_crit_dot_series = self.history['f_crit_dot']
            couple = calculate_couple_index(beta_dot_series, f_crit_dot_series, couple_window)
        else:
            speed = calculate_speed_index(beta_dot, f_crit_dot) # Speed can be calculated with current velocity
            couple = np.nan
        
        self.history['speed_index'].append(speed)
        self.history['couple_index'].append(couple)

        # Check for critical dynamics onset
        if not self.critical_onset_defined and current_theta_t < self.critical_theta_factor * self.initial_theta_t:
            self.critical_dynamics_onset_time = self.time
            self.critical_onset_defined = True


    def check_collapse(self):
        current_theta_t = self.history['theta_t'][-1]
        if self.strain_avg > current_theta_t or self.f_crit <= self.params.get('f_crit_min_collapse', 0.01):
            self.is_collapsed = True
            self.collapse_time = self.time
            # print(f"System collapsed at t={self.time:.2f}, Strain={self.strain_avg:.2f}, Theta_T={current_theta_t:.2f}, F_crit={self.f_crit:.2f}")
        return self.is_collapsed

    def _update_lever_adaptive_response(self):
        """Adaptive lever updates driven by strain/Theta ratio and F_crit.

        This method loosely mimics a Free Energy Principle style response where
        levers increase in proportion to perceived stress (strain relative to
        the Tolerance Sheet) while being moderated by available energetic
        resources.  Costs and optional fatigue diminish ``f_crit`` over time.
        """
        current_theta = self.history["theta_t"][-1]
        ratio = self.strain_avg / max(current_theta, 1e-6)
        gap = current_theta - self.strain_avg

        g_sens = self.params.get(
            "g_lever_sensitivity_to_strain_ratio",
            self.params.get("g_sensitivity", 0.1),
        )
        beta_sens = self.params.get(
            "beta_lever_sensitivity_to_strain_ratio",
            self.params.get("beta_sensitivity", 0.05),
        )

        g_target = self.g_lever + g_sens * (ratio - 1.0) * self.g_lever
        g_target *= 1.0 - 0.1 * max(0.0, (self.params.get("f_crit_initial", 1.0) - self.f_crit) / max(self.params.get("f_crit_initial", 1.0), 1e-6))

        beta_target = self.beta_lever + beta_sens * (gap / max(current_theta, 1e-6))

        g_rate = self.params.get("g_adaptation_rate", 0.3)
        beta_rate = self.params.get("beta_adaptation_rate", 0.3)

        dg = (g_target - self.g_lever) * g_rate
        db = (beta_target - self.beta_lever) * beta_rate

        max_change = max(self.f_crit, 0.0) / (1.0 + abs(self.g_lever) + abs(self.beta_lever))
        dg = np.clip(dg, -max_change, max_change)
        db = np.clip(db, -max_change, max_change)

        self.g_lever += dg * self.dt
        self.beta_lever += db * self.dt

        phi1 = self.params.get("phi1", 1.0)
        g_cost = self.params.get(
            "g_lever_cost_factor",
            self.params.get("g_cost_factor", 0.01),
        )
        beta_cost = self.params.get(
            "beta_lever_cost_factor",
            self.params.get("beta_cost_factor", 0.005),
        )
        cost_g = g_cost * (abs(dg) ** phi1)
        cost_beta = beta_cost * abs(db)
        self.f_crit -= (cost_g + cost_beta) * self.dt
        self.f_crit -= self.params.get(
            "f_crit_fatigue_rate",
            self.params.get("fatigue_rate", 0.0),
        ) * self.dt

        noise_std = self.params.get("lever_noise_std", 0.0)
        if noise_std > 0:
            self.g_lever += np.random.normal(0, noise_std) * np.sqrt(self.dt)
            self.beta_lever += np.random.normal(0, noise_std) * np.sqrt(self.dt)

    def update_levers_and_strain(self, scenario_type):
        # Placeholder: actual lever dynamics will be scenario-specific
        # These functions will modify self.g_lever, self.beta_lever, self.f_crit, self.strain_avg
        if scenario_type == 'slow_creep':
            self.update_slow_creep()
        elif scenario_type == 'sudden_shock':
            self.update_sudden_shock()
        elif scenario_type == 'tightening_loop':
            self.update_tightening_loop()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

    # --- Scenario-specific update methods ---
    def update_slow_creep(self):
        baseline = self.params.get('fcrit_decay_rate', 0.01)
        self.f_crit -= baseline * self.dt
        self.strain_avg += self.params.get('strain_growth_rate', 0.005) * self.dt
        self._update_lever_adaptive_response()


    def update_sudden_shock(self):
        shock_time = self.params.get('shock_time', 5.0)
        if self.time >= shock_time and self.time < shock_time + self.dt: # Apply shock once
            self.f_crit *= self.params.get('fcrit_shock_factor', 0.3)
            self.strain_avg *= self.params.get('strain_shock_factor', 3.0)
            print(f"Shock applied at t={self.time:.2f}")
        self.f_crit -= self.params.get('fcrit_post_shock_decay', 0.005) * self.dt
        self._update_lever_adaptive_response()


    def update_tightening_loop(self):
        # Forcing a "Tightening Loop" dynamic
        # Example: if strain/theta_t ratio is high, beta increases, and f_crit depletes faster with higher beta
        ratio = self.strain_avg / max(self.history['theta_t'][-1], 1e-6)
        extra_decay = self.params.get('fcrit_beta_cost_factor', 0.005) * self.beta_lever
        base_decay = self.params.get('fcrit_base_decay_tightening', 0.01)
        self.f_crit -= (base_decay + extra_decay) * self.dt
        self.strain_avg += self.params.get('strain_tightening_feedback', 0.002) * self.beta_lever * self.dt * max(0, ratio - 1.0)
        self._update_lever_adaptive_response()


    def simulate(self, scenario_type, max_time):
        self.record_state() # Record initial state
        
        while self.time < max_time and not self.is_collapsed:
            self.time += self.dt
            prev_g, prev_b, prev_f = self.g_lever, self.beta_lever, self.f_crit # Store for velocity calculation
            
            self.update_levers_and_strain(scenario_type)

            noise_val = self.params.get("lever_value_noise_std", 0.0)
            if noise_val > 0:
                self.g_lever += np.random.normal(0, noise_val)
                self.beta_lever += np.random.normal(0, noise_val)
                self.f_crit += np.random.normal(0, noise_val)
            
            # Ensure levers don't go unrealistically low (can be part of model too)
            self.f_crit = max(self.f_crit, self.params.get('f_crit_min_value', 1e-6)) # Floor for F_crit
            self.beta_lever = max(self.beta_lever, self.params.get('beta_min_value', 1e-6))
            self.g_lever = max(self.g_lever, self.params.get('g_min_value', 1e-6))

            self.record_state() # Record new state and derived metrics
            self.check_collapse()
            
        # Finalize history to DataFrame
        self.history_df = pd.DataFrame(self.history)
        # Optionally smooth lever series and compute derivatives via Savitzky-Golay
        if len(self.history_df) >= self.savgol_window:
            for lever, dot in [("g_lever", "g_lever_dot"),
                               ("beta_lever", "beta_lever_dot"),
                               ("f_crit", "f_crit_dot")]:
                try:
                    deriv = savgol_filter(
                        self.history_df[lever].values,
                        window_length=self.savgol_window,
                        polyorder=self.savgol_polyorder,
                        deriv=1,
                        delta=self.dt,
                        mode="interp",
                    )
                    self.history_df[dot] = deriv
                except Exception:
                    pass
            self.history_df["speed_index"] = calculate_speed_index(
                self.history_df["beta_lever_dot"], self.history_df["f_crit_dot"])
            couple_window = self.params.get("couple_index_window", 20)
            self.history_df["couple_index"] = (
                pd.Series(self.history_df["beta_lever_dot"]).rolling(
                    window=couple_window,
                    min_periods=max(2, int(couple_window * 0.5)),
                ).corr(pd.Series(self.history_df["f_crit_dot"]))
            )

        return self.history_df


# --- Trajectory Analysis and Plotting ---
def plot_lever_velocity_trajectory_2d(df, scenario_name, params_id, results_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(df["beta_lever_dot"], df["f_crit_dot"], alpha=0.7)
    plt.scatter(
        df["beta_lever_dot"].iloc[0],
        df["f_crit_dot"].iloc[0],
        color="green",
        s=100,
        label="Start",
        zorder=5,
    )
    if not np.isnan(df["beta_lever_dot"].iloc[-1]):
        plt.scatter(
            df["beta_lever_dot"].iloc[-1],
            df["f_crit_dot"].iloc[-1],
            color="red",
            s=100,
            label="Collapse/End",
            zorder=5,
        )

    plt.xlabel(r"$\dot{\beta}_{\mathrm{Lever}}(t)$")
    plt.ylabel(r"$\dot{F}_{\mathrm{crit}}(t)$")

    # -------------  fixed part -------------
    plt.title(
        rf"Lever Velocity Trajectory "
        r"($\dot{{\beta}}_{{\mathrm{{Lever}}}}$ vs $\dot{{F}}_{{\mathrm{{crit}}}}$) "
        f"- {scenario_name} ({params_id})"
    )
    # ---------------------------------------

    plt.grid(True)
    plt.legend()
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)
    plt.tight_layout()

    fname = os.path.join(results_dir, f"traj2d_{scenario_name}_{params_id}.png")
    plt.savefig(fname)
    plt.close()


def plot_lever_velocity_trajectory_3d(df, scenario_name, params_id, results_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['g_lever_dot'], df['beta_lever_dot'], df['f_crit_dot'], alpha=0.7)
    ax.scatter(df['g_lever_dot'].iloc[0], df['beta_lever_dot'].iloc[0], df['f_crit_dot'].iloc[0], color='green', s=100, label='Start')
    if not np.isnan(df['g_lever_dot'].iloc[-1]):
        ax.scatter(df['g_lever_dot'].iloc[-1], df['beta_lever_dot'].iloc[-1], df['f_crit_dot'].iloc[-1], color='red', s=100, label='Collapse/End')
    ax.set_xlabel(r'$\dot{g}_{Lever}(t)$')
    ax.set_ylabel(r'$\dot{\beta}_{Lever}(t)$')
    ax.set_zlabel(r'$\dot{F}_{crit}(t)$')
    ax.set_title(f'3D Lever Velocity Trajectory - {scenario_name} ({params_id})')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(results_dir, f"traj3d_{scenario_name}_{params_id}.png")
    plt.savefig(fname)
    plt.close()


def plot_diagnostic_evolution(df, scenario_name, params_id, results_dir, pre_df=None):
    """Plot evolution of Speed and Couple Index."""
    required = {"time", "speed_index", "couple_index"}
    if not required.issubset(df.columns):
        print(f"Skipping diagnostic evolution plot for {params_id} (missing columns)")
        return
    plot_df = pre_df if pre_df is not None and not pre_df.empty else df
    collapse_time = plot_df["time"].max()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(plot_df["time"], plot_df["speed_index"], label="Speed Index")
    axes[0].axvline(collapse_time, color="red", linestyle="--", label="collapse")
    axes[0].set_ylabel("Speed Index")
    axes[0].legend()

    axes[1].plot(plot_df["time"], plot_df["couple_index"], label="Couple Index", color="orange")
    axes[1].axvline(collapse_time, color="red", linestyle="--")
    axes[1].set_ylabel("Couple Index")
    axes[1].set_xlabel("time")
    axes[1].legend()
    fig.suptitle(f"Diagnostic Evolution - {scenario_name} ({params_id})")
    plt.tight_layout()
    fname = os.path.join(results_dir, f"diag_evolution_{scenario_name}_{params_id}.png")
    plt.savefig(fname)
    plt.close()


def plot_rolling_metric_evolution(df, scenario_name, params_id, results_dir, pre_df=None):
    """Plot evolution of rolling fractal metrics."""
    required = {"time", "rolling_box_dim_2d", "rolling_hurst_beta_dot", "rolling_hurst_g_dot"}
    if not required.issubset(df.columns):
        print(f"Skipping rolling metric plot for {params_id} (missing columns)")
        return
    plot_df = pre_df if pre_df is not None and not pre_df.empty else df
    collapse_time = plot_df["time"].max()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(plot_df["time"], plot_df["rolling_box_dim_2d"], label="BoxDim2D")
    axes[0].axvline(collapse_time, color="red", linestyle="--")
    axes[0].set_ylabel("BoxDim2D")

    axes[1].plot(plot_df["time"], plot_df["rolling_hurst_beta_dot"], label="Hurst beta_dot", color="green")
    axes[1].axvline(collapse_time, color="red", linestyle="--")
    axes[1].set_ylabel("Hurst β_dot")

    axes[2].plot(plot_df["time"], plot_df["rolling_hurst_g_dot"], label="Hurst g_dot", color="purple")
    axes[2].axvline(collapse_time, color="red", linestyle="--")
    axes[2].set_ylabel("Hurst g_dot")
    axes[2].set_xlabel("time")

    fig.suptitle(f"Rolling Fractal Metrics - {scenario_name} ({params_id})")
    plt.tight_layout()
    fname = os.path.join(results_dir, f"rolling_metrics_{scenario_name}_{params_id}.png")
    plt.savefig(fname)
    plt.close()


def plot_tolerance_evolution(df, scenario_name, params_id, results_dir):
    """Plot Theta_T and strain gap evolution."""
    required = {"time", "theta_t", "strain_avg"}
    if not required.issubset(df.columns):
        print(f"Skipping tolerance evolution plot for {params_id} (missing columns)")
        return
    collapse_time = df["time"].max()
    gap = df["theta_t"] - df["strain_avg"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(df["time"], df["theta_t"], label=r"$\Theta_T$")
    axes[0].plot(df["time"], df["strain_avg"], label="strain")
    axes[0].axvline(collapse_time, color="red", linestyle="--")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(df["time"], gap, label="Gap G(t)", color="brown")
    axes[1].axvline(collapse_time, color="red", linestyle="--")
    axes[1].set_ylabel("Gap")
    axes[1].set_xlabel("time")
    axes[1].legend()

    fig.suptitle(f"Tolerance Sheet & Gap - {scenario_name} ({params_id})")
    plt.tight_layout()
    fname = os.path.join(results_dir, f"tolerance_evolution_{scenario_name}_{params_id}.png")
    plt.savefig(fname)
    plt.close()


# --- Analysis Helper Functions ---
def define_pre_collapse_window(
    system,
    df,
    method="onset",
    percent=0.2,
    duration=5.0,
    min_points=10,
):
    """Return dataframe segment defining the pre-collapse window.

    Parameters
    ----------
    system : TDSystem
        The simulated system instance.
    df : pandas.DataFrame
        Full history dataframe of the run.
    method : str, optional
        Method to define the window. ``"onset"`` uses the detected
        ``critical_dynamics_onset_time`` if available. ``"percent"`` uses the
        last ``percent`` of the trajectory, and ``"fixed"`` uses a fixed
        ``duration`` before collapse.
    percent : float
        Portion of the run used when ``method='percent'``.
    duration : float
        Duration used when ``method='fixed'``.
    """

    if not system.is_collapsed:
        return pd.DataFrame()

    if method == "onset" and not np.isnan(system.critical_dynamics_onset_time):
        start = system.critical_dynamics_onset_time
        if system.collapse_time - start < max(5 * system.dt, 0.1 * system.collapse_time):
            start = system.collapse_time * (1 - percent)
            method_used = "onset_fallback_percent"
        else:
            method_used = "onset"
    elif method == "percent":
        start = system.collapse_time * (1 - percent)
        method_used = "percent"
    elif method == "fixed":
        start = max(0.0, system.collapse_time - duration)
        method_used = "fixed"
    else:  # fallback if onset not triggered
        start = system.collapse_time * (1 - percent)
        method_used = "percent_fallback"

    window_df = df[(df["time"] >= start) & (df["time"] <= system.collapse_time)].copy()
    if len(window_df) < min_points:
        return pd.DataFrame()
    window_df["pre_window_method"] = method_used
    return window_df


def compute_rolling_fractal_metrics(df, params):
    """Compute rolling fractal metrics and attach as new columns."""
    window = params.get("rolling_window_points", 50)
    stride = params.get("rolling_stride", 1)
    if window < 5:
        return df

    cols = {
        "rolling_box_dim_2d": [],
        "rolling_box_dim_3d": [],
        "rolling_corr_dim_g_dot": [],
        "rolling_corr_dim_beta_dot": [],
        "rolling_corr_dim_f_crit_dot": [],
        "rolling_mfdfa_hq2_beta_dot": [],
        "rolling_hurst_beta_dot": [],
        "rolling_hurst_g_dot": [],
    }

    for i in range(len(df)):
        if i < window - 1 or (i % stride != 0):
            for k in cols:
                cols[k].append(np.nan)
            continue

        window_df = df.iloc[i - window + 1 : i + 1]
        g_dot = window_df["g_lever_dot"].values
        beta_dot = window_df["beta_lever_dot"].values
        f_dot = window_df["f_crit_dot"].values

        cols["rolling_box_dim_2d"].append(
            box_counting_dimension_2d_simplified(beta_dot, f_dot)
        )
        cols["rolling_box_dim_3d"].append(
            box_counting_dimension_3d(g_dot, beta_dot, f_dot)
        )
        cols["rolling_corr_dim_g_dot"].append(correlation_dimension(g_dot))
        cols["rolling_corr_dim_beta_dot"].append(correlation_dimension(beta_dot))
        cols["rolling_corr_dim_f_crit_dot"].append(correlation_dimension(f_dot))
        hq2, _ = mfdfa_metrics(beta_dot)
        cols["rolling_mfdfa_hq2_beta_dot"].append(hq2)
        n_lags = params.get("hurst_n_lags", 20)
        cols["rolling_hurst_beta_dot"].append(
            hurst_exponent(beta_dot, n_lags=n_lags)
            if len(beta_dot) >= MIN_HURST_LEN
            else np.nan
        )
        cols["rolling_hurst_g_dot"].append(
            hurst_exponent(g_dot, n_lags=n_lags)
            if len(g_dot) >= MIN_HURST_LEN
            else np.nan
        )

    for k, v in cols.items():
        df[k] = v
    return df


def _temporal_stats(series, times, onset, collapse, last_n=50):
    """Return basic temporal stats for a time-series."""
    arr = np.asarray(series, dtype=float)
    t = np.asarray(times, dtype=float)
    mask = ~np.isnan(arr) & ~np.isnan(t)
    if mask.sum() < 2:
        return {
            "slope_full": np.nan,
            "slope_last": np.nan,
            "delta": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "value_at_onset": np.nan,
            "value_at_collapse": np.nan,
        }
    arr = arr[mask]
    t = t[mask]
    slope_full = np.polyfit(t, arr, 1)[0]
    n = min(last_n, len(arr))
    slope_last = np.polyfit(t[-n:], arr[-n:], 1)[0] if n >= 2 else np.nan
    delta = arr[-1] - arr[0]
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    iqr = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
    onset_val = np.interp(onset, t, arr) if not np.isnan(onset) else np.nan
    collapse_val = arr[-1]
    return {
        "slope_full": slope_full,
        "slope_last": slope_last,
        "delta": delta,
        "min": mn,
        "max": mx,
        "iqr": iqr,
        "value_at_onset": onset_val,
        "value_at_collapse": collapse_val,
    }


def _path_geometry(x, y, z=None):
    """Calculate simple path-geometry statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if z is not None:
        z = np.asarray(z, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if z is not None:
        mask &= ~np.isnan(z)
    x = x[mask]
    y = y[mask]
    if z is not None:
        z = z[mask]
    if len(x) < 2:
        return {
            "range_x": np.nan,
            "range_y": np.nan,
            "range_z": np.nan,
            "path_length": np.nan,
            "hull_area": np.nan,
            "hull_volume": np.nan,
            "direction_mean": np.nan,
            "direction_cvar": np.nan,
            "confinement_ratio": np.nan,
        }
    dx = np.diff(x)
    dy = np.diff(y)
    if z is None:
        path_len = np.sum(np.sqrt(dx ** 2 + dy ** 2))
        range_x = x.max() - x.min()
        range_y = y.max() - y.min()
        hull_area = range_x * range_y
        hull_volume = np.nan
        hull_diam = np.sqrt(range_x ** 2 + range_y ** 2)
        angles = np.arctan2(dy, dx)
        range_z = np.nan
    else:
        dz = np.diff(z)
        path_len = np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        range_x = x.max() - x.min()
        range_y = y.max() - y.min()
        range_z = z.max() - z.min()
        hull_area = range_x * range_y
        hull_volume = range_x * range_y * range_z
        hull_diam = np.sqrt(range_x ** 2 + range_y ** 2 + range_z ** 2)
        angles = np.arctan2(dy, dx)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    circ_var = 1.0 - np.sqrt(np.mean(np.sin(angles)) ** 2 + np.mean(np.cos(angles)) ** 2)
    conf_ratio = path_len / hull_diam if hull_diam > 0 else np.nan
    return {
        "range_x": range_x,
        "range_y": range_y,
        "range_z": range_z,
        "path_length": path_len,
        "hull_area": hull_area,
        "hull_volume": hull_volume,
        "direction_mean": mean_angle,
        "direction_cvar": circ_var,
        "confinement_ratio": conf_ratio,
    }


def perform_fractal_analysis(system, history_df, params):
    """Compute fractal metrics and pre-collapse diagnostics."""
    if system.is_collapsed:
        collapse_idx = history_df[history_df["time"] >= system.collapse_time].index[0]
        analysis_df = history_df.iloc[: collapse_idx + 1].copy()
    else:
        analysis_df = history_df.copy()

    valid_df = analysis_df.iloc[1:].copy()
    len_g = len(valid_df["g_lever_dot"].values)
    len_beta = len(valid_df["beta_lever_dot"].values)
    len_f = len(valid_df["f_crit_dot"].values)
    std_g = np.std(valid_df["g_lever_dot"].values) if len_g > 0 else np.nan
    std_beta = np.std(valid_df["beta_lever_dot"].values) if len_beta > 0 else np.nan
    std_f = np.std(valid_df["f_crit_dot"].values) if len_f > 0 else np.nan

    onset_time = system.critical_dynamics_onset_time
    init_theta = system.initial_theta_t
    init_strain = system.params.get("strain_initial", np.nan)
    init_gap = init_theta - init_strain

    if len(valid_df) >= 20:
        n_lags = params.get("hurst_n_lags", 20)
        hurst_g = hurst_exponent(valid_df["g_lever_dot"].values, n_lags=n_lags) if len_g >= MIN_HURST_LEN else np.nan
        hurst_beta = hurst_exponent(valid_df["beta_lever_dot"].values, n_lags=n_lags) if len_beta >= MIN_HURST_LEN else np.nan
        hurst_fcrit = hurst_exponent(valid_df["f_crit_dot"].values, n_lags=n_lags) if len_f >= MIN_HURST_LEN else np.nan
        box_dim_2d = box_counting_dimension_2d_simplified(
            valid_df["beta_lever_dot"].values, valid_df["f_crit_dot"].values
        )
        box_dim_3d = box_counting_dimension_3d(
            valid_df["g_lever_dot"].values,
            valid_df["beta_lever_dot"].values,
            valid_df["f_crit_dot"].values,
        )
        corr_g = correlation_dimension(valid_df["g_lever_dot"].values)
        corr_beta = correlation_dimension(valid_df["beta_lever_dot"].values)
        corr_f = correlation_dimension(valid_df["f_crit_dot"].values)
        mf_hq2_g, mf_w_g = (
            mfdfa_metrics(valid_df["g_lever_dot"].values)
            if len_g >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        mf_hq2_b, mf_w_b = (
            mfdfa_metrics(valid_df["beta_lever_dot"].values)
            if len_beta >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        mf_hq2_f, mf_w_f = (
            mfdfa_metrics(valid_df["f_crit_dot"].values)
            if len_f >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        geom2d = _path_geometry(
            valid_df["beta_lever_dot"].values,
            valid_df["f_crit_dot"].values,
        )
        geom3d = _path_geometry(
            valid_df["g_lever_dot"].values,
            valid_df["beta_lever_dot"].values,
            valid_df["f_crit_dot"].values,
        )
    else:
        hurst_g = hurst_beta = hurst_fcrit = box_dim_2d = np.nan
        box_dim_3d = corr_g = corr_beta = corr_f = np.nan
        mf_hq2_g = mf_w_g = mf_hq2_b = mf_w_b = mf_hq2_f = mf_w_f = np.nan
        geom2d = {k: np.nan for k in [
            "range_x", "range_y", "range_z", "path_length", "hull_area",
            "hull_volume", "direction_mean", "direction_cvar",
            "confinement_ratio",
        ]}
        geom3d = geom2d.copy()

    pre_df = define_pre_collapse_window(
        system,
        analysis_df,
        percent=params.get("precollapse_window_percent", 0.2),
        duration=params.get("precollapse_window_duration", 5.0),
        min_points=params.get("min_pre_collapse_window_points", 10),
    )

    gap_series = analysis_df["theta_t"].values - analysis_df["strain_avg"].values
    gap_stats = _temporal_stats(
        gap_series,
        analysis_df["time"].values,
        onset_time,
        system.collapse_time,
    )
    ratio_series = analysis_df["strain_avg"].values / np.maximum(
        analysis_df["theta_t"].values, 1e-6
    )
    ratio_at_onset = np.interp(onset_time, analysis_df["time"].values, ratio_series) if not np.isnan(onset_time) else np.nan
    ratio_at_collapse = ratio_series[-1] if len(ratio_series) > 0 else np.nan

    roll_stats = {}
    for m in ["rolling_box_dim_2d", "rolling_hurst_beta_dot", "rolling_hurst_g_dot"]:
        if m in analysis_df.columns:
            stats = _temporal_stats(
                analysis_df[m].values,
                analysis_df["time"].values,
                onset_time,
                system.collapse_time,
            )
            for k, v in stats.items():
                roll_stats[f"{m}_{k}"] = v
        else:
            for k in [
                "slope_full", "slope_last", "delta", "min", "max",
                "iqr", "value_at_onset", "value_at_collapse",
            ]:
                roll_stats[f"{m}_{k}"] = np.nan

    speed_stats = _temporal_stats(
        analysis_df["speed_index"].values,
        analysis_df["time"].values,
        onset_time,
        system.collapse_time,
    )
    couple_stats = _temporal_stats(
        analysis_df["couple_index"].values,
        analysis_df["time"].values,
        onset_time,
        system.collapse_time,
    )
    speed_series = analysis_df["speed_index"].values
    time_series = analysis_df["time"].values
    if len(speed_series) > 0:
        idx_max_speed = int(np.nanargmax(speed_series))
        max_speed_time = time_series[idx_max_speed]
        max_speed_val = speed_series[idx_max_speed]
    else:
        max_speed_time = np.nan
        max_speed_val = np.nan
    couple_series = analysis_df["couple_index"].values
    if len(couple_series) > 0:
        idx_min_couple = int(np.nanargmin(couple_series))
        idx_max_couple = int(np.nanargmax(couple_series))
        min_couple_val = couple_series[idx_min_couple]
        max_couple_val = couple_series[idx_max_couple]
        min_couple_time = time_series[idx_min_couple]
        max_couple_time = time_series[idx_max_couple]
    else:
        min_couple_val = max_couple_val = np.nan
        min_couple_time = max_couple_time = np.nan
    if len(pre_df) >= 20:
        n_lags = params.get("hurst_n_lags", 20)
        hurst_g_pre = hurst_exponent(pre_df["g_lever_dot"].values, n_lags=n_lags) if len(pre_df) >= MIN_HURST_LEN else np.nan
        hurst_beta_pre = hurst_exponent(pre_df["beta_lever_dot"].values, n_lags=n_lags) if len(pre_df) >= MIN_HURST_LEN else np.nan
        hurst_fcrit_pre = hurst_exponent(pre_df["f_crit_dot"].values, n_lags=n_lags) if len(pre_df) >= MIN_HURST_LEN else np.nan
        box_dim_pre = box_counting_dimension_2d_simplified(
            pre_df["beta_lever_dot"].values, pre_df["f_crit_dot"].values
        )
        box_dim_3d_pre = box_counting_dimension_3d(
            pre_df["g_lever_dot"].values,
            pre_df["beta_lever_dot"].values,
            pre_df["f_crit_dot"].values,
        )
        corr_g_pre = correlation_dimension(pre_df["g_lever_dot"].values)
        corr_beta_pre = correlation_dimension(pre_df["beta_lever_dot"].values)
        corr_f_pre = correlation_dimension(pre_df["f_crit_dot"].values)
        mf_hq2_g_pre, mf_w_g_pre = (
            mfdfa_metrics(pre_df["g_lever_dot"].values)
            if len(pre_df) >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        mf_hq2_b_pre, mf_w_b_pre = (
            mfdfa_metrics(pre_df["beta_lever_dot"].values)
            if len(pre_df) >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        mf_hq2_f_pre, mf_w_f_pre = (
            mfdfa_metrics(pre_df["f_crit_dot"].values)
            if len(pre_df) >= MIN_HURST_LEN
            else (np.nan, np.nan)
        )
        roll_means = {
            "rolling_box_dim_2d_pre_mean": pre_df["rolling_box_dim_2d"].mean(),
            "rolling_box_dim_3d_pre_mean": pre_df["rolling_box_dim_3d"].mean(),
            "rolling_corr_dim_beta_dot_pre_mean": pre_df["rolling_corr_dim_beta_dot"].mean(),
        }
    else:
        hurst_g_pre = hurst_beta_pre = hurst_fcrit_pre = box_dim_pre = np.nan
        box_dim_3d_pre = corr_g_pre = corr_beta_pre = corr_f_pre = np.nan
        mf_hq2_g_pre = mf_w_g_pre = mf_hq2_b_pre = mf_w_b_pre = mf_hq2_f_pre = mf_w_f_pre = np.nan
        roll_means = {
            "rolling_box_dim_2d_pre_mean": np.nan,
            "rolling_box_dim_3d_pre_mean": np.nan,
            "rolling_corr_dim_beta_dot_pre_mean": np.nan,
        }

    avg_speed_pre = pre_df["speed_index"].mean() if not pre_df.empty else np.nan
    # ``couple_index`` in ``pre_df`` contains the rolling CoupleIndex values
    # calculated during the simulation via ``TDSystem.record_state``.  These
    # are computed using ``calculate_couple_index`` for each time step within
    # the specified window.  The ``avg_couple_pre_collapse`` metric below is
    # therefore the average of these rolling window CoupleIndex values over
    # the pre-collapse period.  Interpretation of whether high positive or
    # strong negative coupling is "detrimental" depends on the scenario.
    avg_couple_pre = pre_df["couple_index"].mean(skipna=True) if not pre_df.empty else np.nan

    # Aggregate Speed and Couple Index statistics over the full trajectory to
    # assist systematic investigation of their relationship to fractal metrics.
    speed_full_mean = analysis_df["speed_index"].mean()
    speed_full_std = analysis_df["speed_index"].std()
    couple_full_mean = analysis_df["couple_index"].mean(skipna=True)
    couple_full_std = analysis_df["couple_index"].std(skipna=True)
    time_to_failure = (
        system.collapse_time - system.critical_dynamics_onset_time
        if system.is_collapsed and not np.isnan(system.critical_dynamics_onset_time)
        else np.nan
    )

    metrics = {
        "hurst_g_dot": hurst_g,
        "hurst_beta_dot": hurst_beta,
        "hurst_f_crit_dot": hurst_fcrit,
        "box_dim_2d": box_dim_2d,
        "box_dim_3d": box_dim_3d,
        "corr_dim_g_dot": corr_g,
        "corr_dim_beta_dot": corr_beta,
        "corr_dim_f_crit_dot": corr_f,
        "mfdfa_hq2_g_dot": mf_hq2_g,
        "mfdfa_width_g_dot": mf_w_g,
        "mfdfa_hq2_beta_dot": mf_hq2_b,
        "mfdfa_width_beta_dot": mf_w_b,
        "mfdfa_hq2_f_crit_dot": mf_hq2_f,
        "mfdfa_width_f_crit_dot": mf_w_f,
        "hurst_g_dot_precrisis": hurst_g_pre,
        "hurst_beta_dot_precrisis": hurst_beta_pre,
        "hurst_f_crit_dot_precrisis": hurst_fcrit_pre,
        "box_dim_2d_precrisis": box_dim_pre,
        "box_dim_3d_precrisis": box_dim_3d_pre,
        "corr_dim_g_dot_precrisis": corr_g_pre,
        "corr_dim_beta_dot_precrisis": corr_beta_pre,
        "corr_dim_f_crit_dot_precrisis": corr_f_pre,
        "mfdfa_hq2_g_dot_precrisis": mf_hq2_g_pre,
        "mfdfa_width_g_dot_precrisis": mf_w_g_pre,
        "mfdfa_hq2_beta_dot_precrisis": mf_hq2_b_pre,
        "mfdfa_width_beta_dot_precrisis": mf_w_b_pre,
        "mfdfa_hq2_f_crit_dot_precrisis": mf_hq2_f_pre,
        "mfdfa_width_f_crit_dot_precrisis": mf_w_f_pre,
        "rolling_box_dim_2d_pre_mean": roll_means["rolling_box_dim_2d_pre_mean"],
        "rolling_box_dim_3d_pre_mean": roll_means["rolling_box_dim_3d_pre_mean"],
        "rolling_corr_dim_beta_dot_pre_mean": roll_means["rolling_corr_dim_beta_dot_pre_mean"],
        "avg_speed_pre_collapse": avg_speed_pre,
        "avg_couple_pre_collapse": avg_couple_pre,
        "speed_index_mean": speed_full_mean,
        "speed_index_std": speed_full_std,
        "couple_index_mean": couple_full_mean,
        "couple_index_std": couple_full_std,
        "time_to_failure_from_onset": time_to_failure,
        "collapse_time": system.collapse_time if system.is_collapsed else params.get("max_time", np.nan),
        "is_collapsed": system.is_collapsed,
        "pre_window_method": pre_df["pre_window_method"].iloc[0] if not pre_df.empty else "none",
        "len_g_dot_series": len_g,
        "len_beta_dot_series": len_beta,
        "len_f_crit_dot_series": len_f,
        "std_g_dot_series": std_g,
        "std_beta_dot_series": std_beta,
        "std_f_crit_dot_series": std_f,
        "onset_time": onset_time,
        "initial_theta_t": init_theta,
        "initial_strain": init_strain,
        "initial_gap": init_gap,
    }
    metrics.update(roll_stats)
    metrics.update({
        "speed_slope_full": speed_stats["slope_full"],
        "speed_slope_last50": speed_stats["slope_last"],
        "speed_delta": speed_stats["delta"],
        "max_speed_index": max_speed_val,
        "time_max_speed_index": max_speed_time,
        "couple_slope_full": couple_stats["slope_full"],
        "couple_slope_last50": couple_stats["slope_last"],
        "couple_delta": couple_stats["delta"],
        "min_couple_index": min_couple_val,
        "time_min_couple_index": min_couple_time,
        "max_couple_index": max_couple_val,
        "time_max_couple_index": max_couple_time,
    })
    metrics.update({f"gap_{k}": v for k, v in gap_stats.items()})
    metrics.update({
        "gap_ratio_at_onset": ratio_at_onset,
        "gap_ratio_at_collapse": ratio_at_collapse,
    })
    metrics.update({f"traj2d_{k}": v for k, v in geom2d.items()})
    metrics.update({f"traj3d_{k}": v for k, v in geom3d.items()})
    return metrics


def run_single_simulation_condition(condition_params):
    """Run simulation for a single parameter configuration."""
    np.random.seed(np.random.randint(0, np.iinfo(np.int32).max))
    scenario_type = condition_params["type"]
    max_time = condition_params.get("max_time", 300.0)
    system = TDSystem(condition_params)
    history_df = system.simulate(scenario_type=scenario_type, max_time=max_time)
    history_df = compute_rolling_fractal_metrics(history_df, condition_params)
    pre_df = define_pre_collapse_window(
        system,
        history_df,
        percent=condition_params.get("precollapse_window_percent", 0.2),
        duration=condition_params.get("precollapse_window_duration", 5.0),
        min_points=condition_params.get("min_pre_collapse_window_points", 10),
    )
    metrics = perform_fractal_analysis(system, history_df, condition_params)
    metrics.update({
        "w1": condition_params["w1"],
        "w2": condition_params["w2"],
        "w3": condition_params["w3"],
        "phi1": condition_params.get("phi1", base_td_params["phi1"]),
        "scenario_type": scenario_type,
    })
    return metrics, history_df, pre_df


def generate_parameter_combinations():
    """Yield parameter dictionaries for all scenario sweeps."""
    for scen_name, conf in scenarios_config.items():
        base = conf.get("base", {}).copy()
        sweep = conf.get("sweep", {})
        keys = list(sweep.keys())
        values = list(sweep.values())
        for combo in itertools.product(*values):
            params = base_td_params.copy()
            params.update(base)
            params.update(dict(zip(keys, combo)))
            if "w_config" in params:
                params["w1"], params["w2"], params["w3"] = params["w_config"]
            params["scenario_name"] = scen_name
            yield params
def correlation_report(df, essential_cols=None):
    df = df.copy()
    df["scenario_type_code"] = pd.factorize(df["scenario_type"])[0]

    corr_cols = [
        "hurst_g_dot", "hurst_beta_dot", "hurst_f_crit_dot", "box_dim_2d",
        "avg_speed_pre_collapse", "avg_couple_pre_collapse",
        "speed_index_mean", "couple_index_mean",
        "speed_index_std",  "couple_index_std",
        "time_to_failure_from_onset", "collapse_time",
        "scenario_type_code", "w1", "w2", "w3", "phi1",
    ] + [c for c in df.columns if c.startswith("param_")]

    corr_cols = [c for c in corr_cols if c in df.columns]
    essential_cols = [
        c for c in (essential_cols or ["hurst_beta_dot", "box_dim_2d"])
        if c in corr_cols
    ]

    # --- build the working DataFrame ---
    corr_df = df[corr_cols]
    corr_df = (
        corr_df.dropna(subset=essential_cols, how="any")
        if essential_cols else
        corr_df.dropna(how="all")
    )

    # keep only columns with ≥20 valid points
    valid_col_counts = corr_df.notna().sum()
    sufficient = valid_col_counts[valid_col_counts >= 20].index.tolist()
    corr_df = corr_df[sufficient]

    # keep essential list in sync *after* the pruning
    essential_cols = [c for c in essential_cols if c in corr_df.columns]

    # ------------------------------------------------------------------ #
    # everything below is unchanged except that it is now guaranteed
    # never to touch a missing column
    # ------------------------------------------------------------------ #
    if len(corr_df) > 1 and len(corr_df.columns) >= 2:
        print("\n--- Correlation Matrix ---")
        corr_matrix = corr_df.corr()
        print(corr_matrix)

        pval_matrix = pd.DataFrame(
            np.ones_like(corr_matrix),
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                r, p = pearsonr(corr_df[col1], corr_df[col2])
                pval_matrix.loc[col1, col2] = p
                pval_matrix.loc[col2, col1] = p
        if pval_matrix.isnull().values.any():
            print(
                "# Note on P-value Matrix:\n"
                "# NaN values can occur if one or both variables have zero variance\n"
                "# within the data subset used for correlation. In such cases,\n"
                "# Pearson correlation and its p-value are undefined."
            )
        print("\n--- P-value Matrix ---")
        print(pval_matrix)
    else:
        # every essential column that *survived* is all-NaN
        missing = [c for c in essential_cols if corr_df[c].isna().all()]
        print(
            "\nInsufficient valid data for correlation matrix "
            f"due to NaNs in key metrics: {missing}"
        )

    # Quick validity report
    key_metrics = ["hurst_g_dot", "hurst_beta_dot", "hurst_f_crit_dot", "box_dim_2d"]
    print("\n--- Valid Data Percentages ---")
    for col in key_metrics:
        if col in df.columns:
            pct = 100 * df[col].notna().mean()
            print(f"{col}: {pct:.1f}% valid")



def grouped_summary(df):
    """Print mean and median metrics grouped by scenario name and type."""
    metrics = [
        "box_dim_2d",
        "hurst_beta_dot",
        "collapse_time",
        "box_dim_2d_precrisis",
        "hurst_beta_dot_precrisis",
    ]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return
    print("\n--- Grouped Statistics by scenario_name ---")
    print(df.groupby("scenario_name")[available].agg(["mean", "median"]))
    print("\n--- Grouped Statistics by scenario_type ---")
    print(df.groupby("scenario_type")[available].agg(["mean", "median"]))


def distribution_plots(df, results_dir):
    """Generate distribution plots for key metrics."""
    metrics = [
        "box_dim_2d",
        "hurst_beta_dot",
        "len_beta_dot_series",
        "std_beta_dot_series",
        "hurst_g_dot",
        "speed_index_mean",
        "couple_index_mean",
        "avg_speed_pre_collapse",
        "avg_couple_pre_collapse",
    ]
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure()
        sns.histplot(df[metric].dropna(), kde=True)
        plt.title(f"Distribution of {metric} - overall")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"dist_{metric}_overall.png"))
        plt.close()

        for scen_type, group in df.groupby("scenario_type"):
            plt.figure()
            sns.histplot(group[metric].dropna(), kde=True)
            plt.title(f"{metric} distribution - {scen_type}")
            plt.tight_layout()
            fname = f"dist_{metric}_{scen_type}.png"
            plt.savefig(os.path.join(results_dir, fname))
            plt.close()

        param_cols = [
            c for c in [
                "param_phi1",
                "param_fcrit_decay_rate",
                "param_w1",
                "param_w2",
                "param_w3",
            ]
            if c in df.columns
        ]

        for pcol in param_cols:
            uniques = df[pcol].dropna().unique()
            if 1 < len(uniques) <= 5:
                for val in sorted(uniques):
                    subset = df[df[pcol] == val]
                    if subset.empty:
                        continue
                    plt.figure()
                    sns.histplot(subset[metric].dropna(), kde=True)
                    plt.title(f"{metric} distribution - {pcol} = {val}")
                    plt.tight_layout()
                    fname = f"dist_{metric}_{pcol}_{val}.png"
                    plt.savefig(os.path.join(results_dir, fname))
                    plt.close()
            elif len(uniques) > 5 or pcol.startswith("param_w"):
                high = df[df[pcol] > 0.4]
                low = df[df[pcol] < 0.3]
                for label, subset in zip(["high", "low"], [high, low]):
                    if subset.empty:
                        continue
                    plt.figure()
                    sns.histplot(subset[metric].dropna(), kde=True)
                    plt.title(f"{metric} distribution - {pcol} {label}")
                    plt.tight_layout()
                    fname = f"dist_{metric}_{pcol}_{label}.png"
                    plt.savefig(os.path.join(results_dir, fname))
                    plt.close()


def exploratory_non_linear_analysis(df, results_dir):
    """Basic exploration of non-linear relationships."""
    pairs = [
        ("w1", "hurst_g_dot"),
        ("avg_couple_pre_collapse", "box_dim_2d"),
    ]
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        plt.figure()
        sns.scatterplot(data=df, x=x, y=y)
        if len(df.dropna(subset=[x, y])) > 2:
            rho, p = spearmanr(df[x], df[y], nan_policy="omit")
            plt.title(f"{y} vs {x} (Spearman r={rho:.2f}, p={p:.3f})")
        else:
            plt.title(f"{y} vs {x}")
        plt.tight_layout()
        fname = f"scatter_{y}_vs_{x}.png"
        plt.savefig(os.path.join(results_dir, fname))
        plt.close()


# ===== Parallel processing start
def _task_wrapper(args):
    """Run a single simulation task with optional seeding."""
    params_template, index, mc, seed = args
    if seed is not None:
        np.random.seed(seed)
    params_copy = params_template.copy()
    params_copy["run_id"] = f"set{index}_mc{mc}"
    params_copy["max_time"] = params_copy.get("max_time", 300.0)
    metrics, hist_df, pre_df = run_single_simulation_condition(params_copy)
    if mc != 0:
        hist_df = None
        pre_df = None
    return mc, params_template, metrics, hist_df, pre_df


def run_parallel(tasks, max_workers):
    """Execute tasks in parallel; override worker count via TD_NUM_WORKERS."""
    env = os.getenv("TD_NUM_WORKERS")
    if env:
        try:
            max_workers = int(env)
        except ValueError:
            pass
    if not max_workers:
        max_workers = min(10, os.cpu_count() or 1)
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(_task_wrapper, tasks), total=len(tasks)):
            results.append(res)
    print(f"Parallel run with {max_workers} workers finished in {time.time() - start:.2f}s")
    return results
# ===== Parallel processing end


def main(args=None):
    args = args or argparse.Namespace(seed=None)
    start_wall = time.time()
    all_results = []
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    parameter_combinations_list = list(generate_parameter_combinations())
    total_unique_param_sets = len(parameter_combinations_list)
    tasks = []
    for i, params_template in enumerate(parameter_combinations_list):
        for mc in range(N_MC_RUNS):
            seed = args.seed + len(tasks) if args.seed is not None else None
            tasks.append((params_template, i, mc, seed))

    total_simulations = len(tasks)
    print(f"Total simulations to run: {total_simulations}")

    max_workers = min(10, os.cpu_count() or 1)
    results = run_parallel(tasks, max_workers)
    max_plots=30
    current_plots=0
    for mc, params_template, result, hist_df, pre_df in results:
        if mc == 0 and hist_df is not None and not np.isnan(result.get("box_dim_2d", np.nan)):
            pid = f"{params_template['scenario_name']}_mc{mc}".replace(" ", "_")
            if result.get("is_collapsed", False) and mc == 0:
                current_plots+=1
                if current_plots <= max_plots:
                    plot_lever_velocity_trajectory_2d(hist_df, params_template['scenario_name'], pid, results_dir)
                    plot_lever_velocity_trajectory_3d(hist_df, params_template['scenario_name'], pid, results_dir)
            plot_diagnostic_evolution(hist_df, params_template['scenario_name'], pid, results_dir, pre_df)
            plot_rolling_metric_evolution(hist_df, params_template['scenario_name'], pid, results_dir, pre_df)
            plot_tolerance_evolution(hist_df, params_template['scenario_name'], pid, results_dir)
        for k in params_template:
            if k not in base_td_params and k not in ["scenario_name", "type", "w_config"]:
                result[f"param_{k}"] = params_template[k]
        result["scenario_name"] = params_template["scenario_name"]
        all_results.append(result)

    elapsed = time.time() - start_wall
    print(f"Completed all simulations in {elapsed:.2f}s")

    df = pd.DataFrame(all_results)
    print("\n--- Simulation Results Summary ---")
    print(df)
    csv_path = os.path.join(results_dir, "td_fractal_experiment_1_results.csv")
    df.to_csv(csv_path, index=False)
    collapse_rates = df.groupby("scenario_name")[["is_collapsed"]].mean() * 100
    print("\n--- Collapse Rates by Scenario ---")
    print(collapse_rates.rename(columns={"is_collapsed": "collapse_%"}))
    correlation_report(df)
    grouped_summary(df)
    distribution_plots(df, results_dir)
    exploratory_non_linear_analysis(df, results_dir)

    # Future Work Suggestion (Interpretation):
    # The fractal dimension metrics (box_dim_2d) could inform intervention strategies.
    # For instance, if low box_dim_2d consistently characterizes 'brittle' collapses,
    # interventions might aim to increase the dimensionality/complexity of the system's
    # adaptive response space (e.g., by fostering exploration or decoupling constraints).


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Base RNG seed")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    main(args)

""" 
Example output:



--- Simulation Results Summary ---
      hurst_g_dot  hurst_beta_dot  ...  param_fcrit_beta_cost_factor  param_strain_tightening_feedback
0        1.024095        1.029303  ...                           NaN                               NaN
1        1.026335        1.029332  ...                           NaN                               NaN
2        1.026147        1.029466  ...                           NaN                               NaN
3        0.987108        0.971211  ...                           NaN                               NaN
4        0.986693        0.969455  ...                           NaN                               NaN
...           ...             ...  ...                           ...                               ...
3307     1.012838        1.017325  ...                          0.02                             0.002
3308     1.009966        1.020692  ...                          0.02                             0.002
3309     1.014423        1.019383  ...                          0.02                             0.002
3310     1.019792        1.026397  ...                          0.02                             0.002
3311     1.005572        1.019368  ...                          0.02                             0.002

[3312 rows x 132 columns]

--- Collapse Rates by Scenario ---
                 collapse_%
scenario_name
slow_creep            100.0
sudden_shock          100.0
tightening_loop       100.0

--- Correlation Matrix ---
                                   hurst_g_dot  ...  param_strain_tightening_feedback
hurst_g_dot                           1.000000  ...                     -6.217302e-03
hurst_beta_dot                        0.223619  ...                     -6.455733e-04
hurst_f_crit_dot                      0.344349  ...                      5.557120e-03
box_dim_2d                           -0.126374  ...                     -2.250707e-03
avg_speed_pre_collapse               -0.112302  ...                     -1.041938e-05
avg_couple_pre_collapse               0.352673  ...                      2.578491e-03
speed_index_mean                     -0.101449  ...                     -4.211174e-05
couple_index_mean                     0.324412  ...                      9.056152e-03
speed_index_std                      -0.119519  ...                     -1.504056e-04
couple_index_std                      0.104982  ...                     -8.515154e-03
time_to_failure_from_onset           -0.172232  ...                     -2.485334e-03
collapse_time                        -0.074361  ...                     -9.242719e-04
scenario_type_code                    0.115330  ...                               NaN
w1                                    0.315117  ...                     -2.834647e-16
w2                                   -0.033490  ...                      2.675408e-16
w3                                   -0.340503  ...                     -2.324570e-15
phi1                                  0.068082  ...                      2.972458e-17
param_strain_growth_rate              0.144133  ...                               NaN
param_max_time                        0.119344  ...                               NaN
param_fcrit_decay_rate                0.140732  ...                               NaN
param_shock_time                           NaN  ...                               NaN
param_fcrit_post_shock_decay               NaN  ...                               NaN
param_beta_post_shock_drift                NaN  ...                               NaN
param_fcrit_shock_factor             -0.065510  ...                               NaN
param_strain_shock_factor            -0.574877  ...                               NaN
param_fcrit_base_decay_tightening     0.155332  ...                     -1.863669e-16
param_fcrit_beta_cost_factor          0.278033  ...                      2.290910e-15
param_strain_tightening_feedback     -0.006217  ...                      1.000000e+00

[28 rows x 28 columns]
# Note on P-value Matrix:
# NaN values can occur if one or both variables have zero variance
# within the data subset used for correlation. In such cases,
# Pearson correlation and its p-value are undefined.

--- P-value Matrix ---
                                    hurst_g_dot  ...  param_strain_tightening_feedback
hurst_g_dot                        1.000000e+00  ...                               NaN
hurst_beta_dot                     8.290999e-39  ...                               NaN
hurst_f_crit_dot                   7.667466e-93  ...                               NaN
box_dim_2d                         2.891730e-13  ...                               NaN
avg_speed_pre_collapse             9.108920e-11  ...                               NaN
avg_couple_pre_collapse            1.340985e-97  ...                               NaN
speed_index_mean                   4.877995e-09  ...                               NaN
couple_index_mean                  5.031386e-82  ...                               NaN
speed_index_std                    5.182384e-12  ...                               NaN
couple_index_std                   1.393784e-09  ...                               NaN
time_to_failure_from_onset                  NaN  ...                               NaN
collapse_time                      1.837208e-05  ...                               NaN
scenario_type_code                 2.795189e-11  ...                               NaN
w1                                 2.974558e-77  ...                               NaN
w2                                 5.396211e-02  ...                               NaN
w3                                 1.082170e-90  ...                               NaN
phi1                               8.811871e-05  ...                               NaN
param_strain_growth_rate                    NaN  ...                               NaN
param_max_time                     5.567323e-12  ...                               NaN
param_fcrit_decay_rate                      NaN  ...                               NaN
param_shock_time                            NaN  ...                               NaN
param_fcrit_post_shock_decay                NaN  ...                               NaN
param_beta_post_shock_drift                 NaN  ...                               NaN
param_fcrit_shock_factor                    NaN  ...                               NaN
param_strain_shock_factor                   NaN  ...                               NaN
param_fcrit_base_decay_tightening           NaN  ...                               NaN
param_fcrit_beta_cost_factor                NaN  ...                               NaN
param_strain_tightening_feedback            NaN  ...                               1.0

[28 rows x 28 columns]

--- Valid Data Percentages ---
hurst_g_dot: 100.0% valid
hurst_beta_dot: 100.0% valid
hurst_f_crit_dot: 100.0% valid
box_dim_2d: 100.0% valid

--- Grouped Statistics by scenario_name ---
                box_dim_2d           hurst_beta_dot  ... box_dim_2d_precrisis hurst_beta_dot_precrisis
                      mean    median           mean  ...               median                     mean    median
scenario_name                                        ...
slow_creep        1.431307  1.438885       1.010945  ...             1.401845                 0.965465  0.992667
sudden_shock      0.125095  0.108760       0.754717  ...             0.108760                 0.774134  0.780119
tightening_loop   1.287247  1.292995       1.005368  ...             1.355785                 1.003925  1.016294

[3 rows x 10 columns]

--- Grouped Statistics by scenario_type ---
                box_dim_2d           hurst_beta_dot  ... box_dim_2d_precrisis hurst_beta_dot_precrisis
                      mean    median           mean  ...               median                     mean    median
scenario_type                                        ...
slow_creep        1.431307  1.438885       1.010945  ...             1.401845                 0.965465  0.992667
sudden_shock      0.125095  0.108760       0.754717  ...             0.108760                 0.774134  0.780119
tightening_loop   1.287247  1.292995       1.005368  ...             1.355785                 1.003925  1.016294

[3 rows x 10 columns]
"""