import matplotlib.pyplot as plt
import pandas as pd

def export_as_df(te_output):
    """Transform the output of transfer entropy to a pandas dataframe.

    Args:
        te_output (list): output of a transfer entropy analysis.

    Returns:
        pd.DataFrame: dataframe including the p values.
    """
    df = pd.DataFrame()
    for index, info in enumerate(te_output[0]):
        col_name, *_ = info
        data = []
        for lag in te_output:
            data.append(lag[index][1]["p_value_XY"].iloc[0])
        df[col_name] = pd.Series(data)
    return df


def viz_df_raw(df, booldf, threshold):
    """Vizualize results of a Transfer Entropy analysis.

    Args:
        df (pd.DataFrame): input data with raw p values.
        booldf (pd.DataFrame): input data after thresholding containing booleans.
        threshold (float): threshold used.
    """
    fs, fs_ax = plt.subplots(len(df.columns), 1, figsize=(10,len(df.columns)*2))

    for ind, col in enumerate(df.columns):
        print(col)
        df[col].astype(float).plot(kind='line', ax=fs_ax[ind], legend = col)
        booldf[col].astype(float).plot(kind='bar', ax=fs_ax[ind], stacked=False, alpha=0.3)
        fs_ax[ind].set_ylim([0,1])
        if ind==0:
            fs_ax[ind].set_title(f"Causal relationships found - Transfer Entropy with significance level = {threshold}")
        if ind == len(df.columns)-1:
            fs_ax[ind].set_xlabel("lags")
    fs.tight_layout()
    fs.subplots_adjust(hspace=0.4, wspace=0)
