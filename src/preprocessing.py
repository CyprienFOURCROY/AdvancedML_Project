import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def fill_null_pd(data, method='mean'):
    """
    Fill null values in a pandas DataFrame with column-wise statistics.
    
    Args
    -----
    data (pd.DataFrame): DataFrame with null values
    method (str): Method to use for filling nulls. Options: 'mean', 'median', 'mode', 'zero'
    
    Returns
    --------
    pd.DataFrame: DataFrame with filled null values
    """
    if method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'mode':
        return data.apply(lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else col))
    elif method == 'zero':
        return data.fillna(0)
    else:
        raise ValueError("Method must be one of: 'mean', 'median', 'mode', 'zero'")

def fill_null_np(data, method='mean'):
    """
    Fill null values (NaN) in a numpy array.
    
    Args
    -----
    data (np.ndarray): Numpy array with null values
    method (str): Method to use for filling nulls. Options: 'mean', 'median', 'mode', 'zero'
    
    Returns
    --------
    np.ndarray: Array with filled null values
    """
    feature_names = [f"column_{i}" for i in range(data.shape[1])] if data.ndim > 1 else ["column_0"]
    
    if data.ndim == 1:
        df = pd.DataFrame(data.reshape(-1, 1), columns=feature_names)
    else:
        df = pd.DataFrame(data, columns=feature_names)
    
    df_filled = fill_null_pd(df, method)
    
    return df_filled.values.reshape(data.shape)

def fill_null(data, method='mean'):
    """
    Fill null values in either a pandas DataFrame or numpy array.
    
    Args
    -----
    data: Either pandas DataFrame or numpy array with null values
    method (str): Method to use for filling nulls. Options: 'mean', 'median', 'mode', 'zero'
    
    Returns
    --------
    Same type as input: Data structure with filled null values
    """
    if isinstance(data, pd.DataFrame):
        return fill_null_pd(data, method)
    elif isinstance(data, np.ndarray):
        return fill_null_np(data, method)
    else:
        raise TypeError("Input must be either a pandas DataFrame or numpy array")

def remove_collinear_pd(data, threshold=5.0, verbose=True):
    """
    Remove collinear features from a pandas DataFrame based on VIF.
    
    Args
    -----
    data (pd.DataFrame): DataFrame with potentially collinear features
    threshold (float): VIF threshold above which features are considered collinear
    verbose (bool): Whether to print information about removed features
    
    Returns
    --------
    pd.DataFrame: DataFrame with collinear features removed
    """
    X = data.copy()
    dropped_info = []
    dropped_cols = []
    
    while True:
        vif_data = pd.DataFrame()
        X_with_const = add_constant(X)
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(X.shape[1])]
        
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            dropped_info.append({
                'feature': feature_to_drop,
                'vif': max_vif
            })
            dropped_cols.append(feature_to_drop)
            X = X.drop(feature_to_drop, axis=1)
        else:
            break
    
    if verbose and dropped_info:
        print(f"Removed {len(dropped_cols)} collinear features with VIF threshold {threshold}:")
        for info in dropped_info:
            print(f"  Dropped '{info['feature']}', VIF: {info['vif']:.4f}")
    
    return data.drop(dropped_cols, axis=1)

def remove_collinear_np(data, threshold=5.0, verbose=True, feature_names=None):
    """
    Remove collinear features from a numpy array based on VIF.
    
    Args
    -----
    data (np.ndarray): 2D array with potentially collinear features
    threshold (float): VIF threshold above which features are considered collinear
    verbose (bool): Whether to print information about removed features
    feature_names (list): Optional list of feature names for better reporting
    
    Returns
    --------
    np.ndarray: Array with collinear features removed
    columns_kept (list): Indices of columns that were kept
    """
    if feature_names is None:
        feature_names = [f"column_{i}" for i in range(data.shape[1])]
    
    df = pd.DataFrame(data, columns=feature_names)
    df_cleaned = remove_collinear_pd(df, threshold, verbose)
    
    columns_kept = [i for i, col in enumerate(feature_names) if col in df_cleaned.columns]
    return df_cleaned.values, columns_kept

def remove_collinear(data, threshold=5.0, verbose=True):
    """
    Remove collinear features using Variance Inflation Factor (VIF).
    
    Args
    -----
    data: Either pandas DataFrame or numpy array with potentially collinear features
    threshold (float): VIF threshold above which features are considered collinear
    verbose (bool): Whether to print information about removed features
    
    Returns
    --------
    Same type as input: Data structure with collinear features removed
    """
    if isinstance(data, pd.DataFrame):
        return remove_collinear_pd(data, threshold, verbose)
    elif isinstance(data, np.ndarray):
        feature_names = [f"column_{i}" for i in range(data.shape[1])]
        result, _ = remove_collinear_np(data, threshold, verbose, feature_names)
        return result
    else:
        raise TypeError("Input must be either a pandas DataFrame or numpy array")

def scale_features_pd(data, feature_range=(0, 1), copy=True):
    """
    Scale features in a pandas DataFrame to a specified range.
    
    Args
    -----
    data (pd.DataFrame): DataFrame with features to scale
    feature_range (tuple): Target range for scaling (min, max)
    copy (bool): Whether to return a new DataFrame or modify in place
    
    Returns
    --------
    pd.DataFrame: DataFrame with scaled features
    """
    min_val, max_val = feature_range
    X = data.copy() if copy else data
    
    for column in X.columns:
        col_min = X[column].min()
        col_max = X[column].max()
        
        if col_max > col_min:
            X[column] = min_val + (X[column] - col_min) * (max_val - min_val) / (col_max - col_min)
        else:
            X[column] = min_val
            
    return X

def scale_features_np(data, feature_range=(0, 1), copy=True):
    """
    Scale features in a numpy array to a specified range.
    
    Args
    -----
    data (np.ndarray): 2D array with features to scale
    feature_range (tuple): Target range for scaling (min, max)
    copy (bool): Whether to return a new array or modify in place
    
    Returns
    --------
    np.ndarray: Array with scaled features
    """
    feature_names = [f"column_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_names)
    df_scaled = scale_features_pd(df, feature_range, copy)
    
    return df_scaled.values

def scale_features(data, feature_range=(0, 1), copy=True):
    """
    Scale features to a specified range.
    
    Args
    -----
    data: Either pandas DataFrame or numpy array with features to scale
    feature_range (tuple): Target range for scaling (min, max)
    copy (bool): Whether to return a new data structure or modify in place
    
    Returns
    --------
    Same type as input: Data structure with scaled features
    """
    if isinstance(data, pd.DataFrame):
        return scale_features_pd(data, feature_range, copy)
    elif isinstance(data, np.ndarray):
        return scale_features_np(data, feature_range, copy)
    else:
        raise TypeError("Input must be either a pandas DataFrame or numpy array")