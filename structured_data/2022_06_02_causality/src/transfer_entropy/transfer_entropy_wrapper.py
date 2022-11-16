"""
Functions for identifying causality.
"""
# Import standard library modules
from typing import List, Tuple, Dict
import warnings

# Import third party modules
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from transfer_entropy.pycausality.src import TransferEntropy

warnings.filterwarnings("ignore")

# Function definitions
def grangers_causation_matrix(df: pd.DataFrame, test: str='ssr_chi2test', max_lag: int=7, verbose=False) -> Tuple[pd.DataFrame, pd.DataFrame]:    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    Arguments:
        - df (pd.DataFrame):
        - test (str):
        - max_lag (int):
        - verbose: Whether or not to display intermediate results.

    Results:
        - Tuple[pd.DataFrame, pd.DataFrame]: Dataframes containing the minimum p_value (i.e., largest
          significance) and corresponding lag for each of the columns of the argument dataframe.
    """
    df_gc = pd.DataFrame(np.zeros((1, len(df.columns[1:]))), columns=df.columns[1:], index=[df.columns[0]])
    df_gc_lags = pd.DataFrame(np.zeros((1, len(df.columns[1:]))), columns=df.columns[1:], index=[df.columns[0]])

    col_res = df.columns[0]
    for col_orig in df.columns[1:]:
        test_result = grangercausalitytests(df[[col_res, col_orig]], maxlag=max_lag, verbose=False)
        p_values = [round(test_result[i+1][0][test][1],4) for i in range(max_lag)]
        if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
        min_p_value = np.min(p_values)
        min_lags = np.argmin(p_values)
        df_gc.loc[col_res, col_orig] = min_p_value
        df_gc_lags.loc[col_res, col_orig] = min_lags

    df_gc.columns = [col + '_x' for col in df.columns[1:]]
    df_gc.index = [df.columns[0]]

    df_gc_lags.columns = [col + '_x' for col in df.columns[1:]]
    df_gc_lags.index = [df.columns[0]]
    return df_gc, df_gc_lags


def calculate_transfer_entropy(df: pd.DataFrame, lag: int, linear: bool=False, effective: bool=False, window_size: Dict={'MS': 6}, window_stride: Dict={'MS': 1}, n_shuffles=100, debug=False) -> List:
    """Perform Seasonal-Trend decomposition using LOESS (STL) to remove trend 
    and seasonality and difference residuals as much as necessary to make 
    time-series stationary.

    Arguments:
        - df (pd.DataFrame): Dataframe for which transfer entropies must be
            calculated.
        - linear (bool): Whether the required transfer entropies should be linear
            (True) or non-linear (False).
        - effective (bool): Whether or not to calculate the effective transfer
            entropy. Can only be done for `n_shuffles>0`, but has proven to not
            give the most reliable results given the size of the dataset.
        - window_size (Dict): Dictionary indicating the size of a window, either in 'MS'
            (Month Start) or 'D' (Days; to express weeks), e.g., {'MS': 6}.
        - window_stride (Dict): Dictionary indicating the stride of a window, either in 'MS'
            (Month Start) or 'D' (Days; to express weeks), e.g., {'MS': 1}.
        - n_shuffles (int): Number of shuffling operations to do when calculating
            the average transfer entropy. Only relevant if the results should be
            either the effective entropy or if p-values should be included for 
            significance.
        - debug (bool): Whether or not to print intermediate results (for
            debugging purposed).
    
    Result:
        - List[List[str, pd.DataFrame]]: List containing nested lists (pairs) of
            the column names and the resulting Pandas dataframe containing the
            transfer entropy for each window in the respective column.
    """
    
    te_results = []
    
    col_res = df.columns[0]
    col_origs = df.columns[1:]
    for col_orig in col_origs:
        print(f'{col_orig} -> {col_res}')
        
        # Initialise Object to Calculate Transfer Entropy
        TE = TransferEntropy(DF=df,
                            endog=col_res,
                            exog=col_orig,
                            lag=lag,
                            window_size=window_size,
                            window_stride=window_stride
                            )

        # Calculate TE using KDE
        if linear:
            TE.linear_TE(n_shuffles=n_shuffles)
        else:
            TE.nonlinear_TE(pdf_estimator='kernel', n_shuffles=n_shuffles)

        # Standardize column naming
        if (linear):
            TE.results = TE.results.rename(mapper=(lambda col: col.replace('linear_', '')), axis=1)

        # Display TE_XY, TE_YX and significance values
        if debug:
            if n_shuffles and effective:
                #print('\t', TE.results[[f'TE_XY', f'Ave_TE_XY', f'p_value_XY']])
                print('\t', f"TE_XY_Eff=({TE.results['TE_XY'].values[0] - TE.results['Ave_TE_XY'].values[0]}), p=({TE.results['p_value_YX'].values[0]})", '\n')
            elif n_shuffles:
                print('\t', f"TE_XY=({TE.results['TE_XY'].values[0]}), p=({TE.results['p_value_YX'].values[0]})", '\n')
            else:
                print('\t', f"TE_XY=({TE.results[['TE_XY']]})", '\n')
        
        # Track results of current link
        te_results.append([col_orig, TE.results])
    return te_results

def average_transfer_entropy(df: pd.DataFrame, linear: bool, effective: bool, tau_min: int=0, tau_max: int=4, n_shuffles=None, debug: bool=False) -> List:
    """Wrapper function around `calculate_transfer_entropy` for calculating the 
    average (non-)linear transfer entropy.

    Arguments:
        - df (pd.DataFrame): Dataframe for which transfer entropies must be
            calculated.
        - linear (bool): Whether the required transfer entropies should be linear
            (True) or non-linear (False).
        - effective (bool): Whether or not to calculate the effective transfer
            entropy. Can only be done for `n_shuffles>0`, but has proven to not
            give the most reliable results given the size of the dataset.
        - tau_min (int): Minimal lag to calculate transfer entropy for.
        - tau_max (int): Maximal lag to calculate transfer entropy for.
        - n_shuffles (int): Number of shuffling operations to do when calculating
            the average transfer entropy. Only relevant if the results should be
            either the effective entropy or if p-values should be included for 
            significance.
        - debug (bool): Whether or not to print intermediate results (for
            debugging purposed).
    
    Result:
        - List[List[str, pd.DataFrame]]: List containing nested lists (pairs) of
            the column names and the resulting Pandas dataframe containing the
            transfer entropy for each window in the respective column.
    """
    te_results_arr = []
    for lag in range(tau_min, tau_max+1):
        print(f'\nlag({lag})')
        import time
        t= time.time()

        # Call over-arching Transfer Entropy function
        te_results = calculate_transfer_entropy(df, lag=lag, linear=linear, window_size=None, window_stride=None, n_shuffles=n_shuffles, debug=debug)

        # Construct dataframe from results
        te_results_df = pd.DataFrame(data=pd.concat(np.array(te_results)[:, 1]))
        te_results_df.index = np.array(te_results)[:, 0]
        
        # Keep track of results
        te_results_arr.append(te_results)
        print("took", time.time() - t, "seconds")
        
    return te_results_arr