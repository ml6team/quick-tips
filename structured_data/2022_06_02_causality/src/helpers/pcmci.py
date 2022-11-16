import numpy as np
from tigramite import plotting as tp

def get_selected_links(df, tau_min=0, tau_max=3, selected_columns_indices = None):
    """
    Initialize dictionary with every possible link (i.e., combination of 
    columns) to be tested by PCMCI. Note that only causality of marketing 
    channels on sales are considered and NOT between channels.
    
    Arguments:
        - df (pd.DataFrame): input data
        - tau_min (int): timelag to start from
        - tau_max (int): timelag to end with
        - selected_columns_indices (list[int]): column indices to exclude columns
    
    Retruns:
        - list[Tuple]: links
    """
    selected_links = {}
    n_cols = list(range(len(df.columns)))

    for col in n_cols:
        selected_links[col] = [(link_col, -lag) for link_col in n_cols
                            for lag in range(tau_min, tau_max + 1)
                            if link_col>0 and lag>0]
        
        if col not in selected_columns_indices:
            # Do not consider causality between channels
            selected_links[col] = [] # only need first col as ref

    return selected_links

def process_and_visualize_results(results, pcmci, cols, target_indices, controlFDR = False):
    """
    Process and visualize the results of PCMCI.

    Arguments:
        - results (list): Output of PCMCI run.
        - pcmci (tigramite.pcmci.PCMCI): PCMCI object
        - cols (list): column names
        - target_indices (list): indices of target columns
        - controlFDR (bool): whether to use the q_matrix, which involves a transformation
                of the p_values to account for amount of statistical tests done. 
                Recommended if you checked many links using PCMCI.
                See the following link for more information:
                https://github.com/jakobrunge/tigramite/blob/master/tutorials/tigramite_tutorial_basics.ipynb
    """


    if not controlFDR:
        pcmci.print_significant_links(
            p_matrix = results['p_matrix'], 
            val_matrix = results['val_matrix'],
            alpha_level = 0.01)

    else:
        q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], 
                                                       fdr_method='fdr_bh',
                                                       exclude_contemporaneous = False)

        pcmci.print_significant_links(
            p_matrix = results['p_matrix'], 
            q_matrix = q_matrix,
            val_matrix = results['val_matrix'],
            alpha_level = 0.01)

    column_indices = set(target_indices)

    for ind, i in enumerate(results['graph']):
        if not set(i.flatten()) == set(['']):
            column_indices.add(ind)


    tmp_results_val_matrix = np.array([i[list(column_indices)] for ind, i in enumerate(results['val_matrix']) if ind in list(column_indices)])

    graph_small = np.array([i[list(column_indices)] for ind, i in enumerate(results['graph']) if ind in list(column_indices)])
    
    var_names_small = []
    for i in column_indices:
        print(cols[i])
    for i in column_indices:
        var_names_small.append(cols[i])

    tp.plot_graph(
        val_matrix=tmp_results_val_matrix,
        graph=graph_small,
        var_names=var_names_small,
        )

    # Plot time series graph    
    tp.plot_time_series_graph(
        figsize=(6, 4),
        val_matrix=tmp_results_val_matrix,
        graph=graph_small,
        var_names=var_names_small,
        link_colorbar_label='MCI',
        )