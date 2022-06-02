"""
Functions for determining if a time series is stationary and for making it stationary
in case it is not.
"""
# Import standard library modules
import math
from typing import Tuple, List
import warnings

# Import third party modules
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

def perform_kpss_test(df: pd.DataFrame, col: str, debug: bool=False) -> Tuple[bool, float]:
    """Perform  the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null 
    hypothesis that x is level or trend stationary.

    Arguments:
        - df (pd.DataFrame): Dataframe for which to check for stationarity.
        - col (str): Name of column within dataframe to check stationarity for.
        - debug (bool): Whether or not to print intermediate results.

    Returns:
        - bool: Whether or not the column of the dataframe is stationary.
        - float: Significance with which conclusion is made.
    """
    # Select `col` column from argument `df` dataframe
    df_col = df[[col]]
        
    # Perform KPSS test (hyp: stationary) while catching InterpolationWarning messages
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        kpss_test = kpss(df_col, nlags='legacy') # regression='c'|'ct'

        if len(w) == 1 and issubclass(w[-1].category, InterpolationWarning):
            p_value_oob = True
        else:
            p_value_oob = False

    kpss_output = pd.Series(kpss_test[0:3], 
                            index=['test_statistic', 'p_value', 'lags'])
    for key, value in kpss_test[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    
    p_value = kpss_output['p_value']
    stationary = p_value >= 0.05 # Stationary if null-hyp. cannot be rejected.

    if debug or not stationary:
        print(f'\t(KPSS) Time-series IS {"" if stationary else "NOT "}trend-stationary (p{">" if p_value_oob else "="}{p_value})!')
    return stationary, p_value

    
def perform_adf_test(df: pd.DataFrame, col: str, debug: bool=False) -> Tuple[bool, float]:
    """Perform Augmented Dickey-Fuller (ADF) unit root test for a unit root in a
     univariate process in the presence of serial correlation.

    Arguments:
        - df (pd.DataFrame): Dataframe for which to check for stationarity.
        - col (str): Name of column within dataframe to check stationarity for.
        - debug (bool): Whether or not to print intermediate results.

    Returns:
        - bool: Whether or not the column of the dataframe is stationary.
        - float: Significance with which conclusion is made.
    """
    # Select `col` column from argument `df` dataframe
    df_col = df[[col]]
        
    # Difference column values
    df_col = df_col[col].diff()
    df_col = df_col.fillna(0) # Remove first month of differenced data
    
    # Perform ADF unit root test
    adf_test = adfuller(df_col, autolag='AIC')
    adf_output = pd.Series(adf_test[0:4], index=['test_statistic','p_value','lags','observations'])
    for key,value in adf_test[4].items():
        adf_output['Critical Value (%s)'%key] = value
    
    p_value = adf_output['p_value']
    stationary = p_value < 0.05 # Stationary if null-hyp. is rejected!

    if debug or not stationary:
        print(f'\t(ADF)  Time-series IS {"" if stationary else "NOT "}difference stationary (p={p_value})!')
    
    return stationary, p_value


def remove_trend_and_diff(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    """Perform Seasonal-Trend decomposition using LOESS (STL) to remove trend 
    and seasonality and difference residuals as much as necessary to make 
    time-series stationary.

    Arguments:
        - df (pd.DataFrame): Dataframe of which stationarity must be checked and
            guaranteed.
        - debug (bool): Whether or not to print intermediate results (for
            debugging purposed).
    
    Result:
        - pd.DataFrame: Stationary dataframe.
    """
    # Keep track of number of differencing operations to omit NaN values at start of dataframe
    max_diff = 1

    # Initialize differenced dataframe
    df_diff = df.copy()
    

    # Make every column of dataframe stationary by...
    for col in df_diff.columns:
        print("tackling new col", col)
        periods = 0
        kpss_stat, kpss_p = perform_kpss_test(df_diff[periods:], col, debug=debug)
        adf_stat, adf_p = perform_adf_test(df_diff[periods:], col, debug=debug)
        
        while not (kpss_stat and adf_stat):
            print(f"  iteration {periods}")

            # Log number of differencing operations
            periods += 1
            print(f'\tDifferencing results over {periods} period{"s" if periods - 1 else ""}...')
            
            # Difference signal
            df_diff[col] = df_diff[col].diff()
            df_diff = df_diff.fillna(0)

            # Check for stationarity
            kpss_stat, kpss_p = perform_kpss_test(df_diff[periods:], col, debug=debug)
            adf_stat, adf_p = perform_adf_test(df_diff[periods:], col, debug=debug)

        # Print if stationarity is obtained
        print(f' --> (KPSS & ADF) Time-series IS stationary for {col} (after {periods} differencing operations)!')

        # Break up print statements between columns
        print('')
    
    print(f'(Maximum number of differencing operations performed was {max_diff})')
    # Return detrended (and possibly differenced) dataframe
    return df_diff[max_diff:]