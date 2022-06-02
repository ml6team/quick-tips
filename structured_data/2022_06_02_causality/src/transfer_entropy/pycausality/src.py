import pandas as pd
import statsmodels.api as sm
import numpy as np

from numpy import ma, atleast_2d, pi, sqrt, sum
from scipy import stats, linalg
from scipy.special import gammaln
from six import string_types
from scipy.stats.mstats import mquantiles

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dateutil.relativedelta import relativedelta

import warnings
import sys


class LaggedTimeSeries():
    """
        Custom wrapper class for pandas DataFrames for performing predictive analysis.
        Generates lagged time series and performs custom windowing over datetime indexes
    """

    def __init__(self, df, endog, lag=None, max_lag_only=True, window_size=None, window_stride=None):
        """
        Args:
            df              -   Pandas DataFrame object of N columns. Must be indexed as an increasing 
                                time series (i.e. past-to-future), with equal timesteps between each row
            lags            -   The number of steps to be included. Each increase in Lags will result 
                                in N additional columns, where N is the number of columns in the original 
                                dataframe. It will also remove the first N rows.
            max_lag_only    -   Defines whether the returned dataframe contains all lagged timeseries up to 
                                and including the defined lag, or only the time series equal to this lag value
            window_size     -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride   -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases

        Returns:    -   n/a
        """
        self.df = sanitise(df)
        self.endog = endog
        self.axes = list(self.df.columns.values)  # Variable names

        self.max_lag_only = max_lag_only
        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()

        if window_size is not None and window_stride is not None:
            self.has_windows = True
            self. __apply_windows__(window_size, window_stride)
        else:
            self.has_windows = False

    def __apply_lags__(self):
        """
        Args:
            n/a
        Returns:
            new_df.iloc[self.t:]    -   This is a new dataframe containing the original columns and
                                        all lagged columns. Note that the first few rows (equal to self.lag) will
                                        be removed from the top, since lagged values are of coursenot available
                                        for these indexes.
        """
        # Create a new dataframe to maintain the new data, dropping rows with NaN
        new_df = self.df.copy(deep=True).dropna()

        # Create new column with lagged timeseries for each variable
        col_names = self.df.columns.values.tolist()

        # If the user wants to only consider the time series lagged by the
        # maximum number specified or by every series up to an including the maximum lag:
        if self.max_lag_only == True:
            for col_name in col_names:
                new_df[col_name + '_lag' +
                       str(self.t)] = self.df[col_name].shift(self.t)

        elif self.max_lag_only == False:
            for col_name in col_names:
                for t in range(1, self.t+1):
                    new_df[col_name + '_lag' +
                           str(t)] = self.df[col_name].shift(t)
        else:
            raise ValueError('Error')

        # Drop the first t rows, which now contain NaN
        return new_df.iloc[self.t:]

    def __apply_windows__(self, window_size, window_stride):
        """
        Args:
            window_size      -   Dict passed from self.__init__
            window_stride    -   Dict passed from self.__init__
        Returns:    
            n/a              -   Sets the daterange for the self.windows property to iterate along
        """
        self.window_size = {'YS': 0, 'MS': 0, 'D': 0,
                            'H': 0, 'min': 0, 'S': 0, 'ms': 0}
        self.window_stride = {'YS': 0, 'MS': 0,
                              'D': 0, 'H': 0, 'min': 0, 'S': 0, 'ms': 0}

        self.window_stride.update(window_stride)
        self.window_size.update(window_size)
        freq = ''
        daterangefreq = freq.join(
            [str(v)+str(k) for (k, v) in self.window_stride.items() if v != 0])
        self.daterange = pd.date_range(
            self.df.index.min(), self.df.index.max(), freq=daterangefreq)

    def date_diff(self, window_size):
        """
        Args: 
            window_size     -    Dict passed from self.windows function
        Returns:
            start_date      -    The start date of the proposed window
            end_date        -    The end date of the proposed window    

        This function is TBC - proposed due to possible duplication of the relativedelta usage in self.windows and self.headstart
        """
        pass

    @property
    def windows(self):
        """
        Args: 
            n/a
        Returns:
            windows         -   Generator defining a pandas DataFrame for each window of the data. 
                                Usage like:   [window for window in LaggedTimeSeries.windows]
        """
        if self.has_windows == False:
            return self.df
        # Loop Over TimeSeries Range
        for i, dt in enumerate(self.daterange):

            # Ensure Each Division Contains Required Number of Months
            if dt-relativedelta(years=self.window_size['YS'],
                                months=self.window_size['MS'],
                                days=self.window_size['D'],
                                hours=self.window_size['H'],
                                minutes=self.window_size['min'],
                                seconds=self.window_size['S'],
                                microseconds=self.window_size['ms']
                                ) >= self.df.index.min():

                # Create Window
                yield self.df.loc[(dt-relativedelta(years=self.window_size['YS'],
                                                    months=self.window_size['MS'],
                                                    days=self.window_size['D'],
                                                    hours=self.window_size['H'],
                                                    minutes=self.window_size['min'],
                                                    seconds=self.window_size['S'],
                                                    microseconds=self.window_size['ms']
                                                    )): dt]

    @property
    def headstart(self):
        """
        Args: 
            n/a
        Returns:
            len(windows)    -   The number of windows which would have start dates before the desired date range. 
                                Used in TransferEntropy class to slice off incomplete windows.

        """
        windows = [i for i, dt in enumerate(self.daterange)
                   if dt-relativedelta(years=self.window_size['YS'],
                                       months=self.window_size['MS'],
                                       days=self.window_size['D'],
                                       hours=self.window_size['H'],
                                       minutes=self.window_size['min'],
                                       seconds=self.window_size['S'],
                                       microseconds=self.window_size['ms']
                                       ) < self.df.index.min()]
        # i.e. count from the first window which falls entirely after the earliest date
        return len(windows)


class TransferEntropy():
    """
        Functional class to calculate Transfer Entropy between time series, to detect causal signals.
        Currently accepts two series: X(t) and Y(t). Future extensions planned to accept additional endogenous 
        series: X1(t), X2(t), X3(t) etc. 
    """

    def __init__(self, DF, endog, exog, lag=None, window_size=None, window_stride=None):
        """
        Args:
            DF            -   (DataFrame) Time series data for X and Y (NOT including lagged variables)
            endog         -   (string)    Fieldname for endogenous (dependent) variable Y
            exog          -   (string)    Fieldname for exogenous (independent) variable X
            lag           -   (integer)   Number of periods (rows) by which to lag timeseries data
            window_size   -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        Returns:
            n/a
        """
        self.lts = LaggedTimeSeries(df=sanitise(DF),
                                    endog=endog,
                                    lag=lag,
                                    window_size=window_size,
                                    window_stride=window_stride)

        if self.lts.has_windows is True:
            self.df = self.lts.windows
            self.date_index = self.lts.daterange[self.lts.headstart:]
            self.results = pd.DataFrame(index=self.date_index)
            self.results.index.name = "windows_ending_on"
        else:
            self.df = [self.lts.df]
            self.results = pd.DataFrame(index=[0])
        self.max_lag_only = True
        self.endog = endog                             # Dependent Variable Y
        self.exog = exog                               # Independent Variable X
        self.lag = lag

        """ If using KDE, this ensures the covariance matrices are calculated once over all data, rather
            than for each window. This saves computational time and provides a fair point for comparison."""
        self.covars = [[], []]

        for i, (X, Y) in enumerate({self.exog: self.endog, self.endog: self.exog}.items()):
            X_lagged = X+'_lag'+str(self.lag)
            Y_lagged = Y+'_lag'+str(self.lag)

            self.covars[i] = [np.cov(self.lts.df[[Y, Y_lagged, X_lagged]].values.T),
                              np.cov(
                self.lts.df[[X_lagged, Y_lagged]].values.T),
                np.cov(self.lts.df[[Y, Y_lagged]].values.T),
                np.ones(shape=(1, 1)) * self.lts.df[Y_lagged].std()**2]

            # Account for equal signals in case of lag 0 by adding identity matrix to covariance matrices
            if lag == 0:
                for j, c_j in enumerate(self.covars[i]):
                    if j % 2 == 0:
                        self.covars[i][j] += 1e-10 * np.eye(*c_j.shape)

    def linear_TE(self, df=None, n_shuffles=0):
        """
        Linear Transfer Entropy for directional causal inference

        Defined:            G-causality * 0.5, where G-causality described by the reduction in variance of the residuals
                            when considering side information.
        Calculated using:   log(var(e_joint)) - log(var(e_independent)) where e_joint and e_independent
                            represent the residuals from OLS fitting in the joint (X(t),Y(t)) and reduced (Y(t)) cases

        Arguments:
            n_shuffles  -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to 
                                        perform significance testing.
        Returns:
            transfer_entropies  -  (list) Directional Linear Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively
        """
        # Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        # Loop over all windows
        for i, df in enumerate(self.df):
            df = deepcopy(df)

            # Shows user that something is happening
            # if self.lts.has_windows is True:
            #    print("Window ending: ", self.date_index[i])

            # Initialise list to return TEs
            transfer_entropies = [0, 0]

            # Require us to compare information transfer bidirectionally
            for i, (X, Y) in enumerate({self.exog: self.endog, self.endog: self.exog}.items()):

                # Note X-t, Y-t
                X_lagged = X+'_lag'+str(self.lag)
                Y_lagged = Y+'_lag'+str(self.lag)

                # Calculate Residuals after OLS Fitting, for both Independent and Joint Cases
                joint_residuals = sm.OLS(df[Y], sm.add_constant(
                    df[[Y_lagged, X_lagged]])).fit().resid
                independent_residuals = sm.OLS(
                    df[Y], sm.add_constant(df[Y_lagged])).fit().resid

                # Use Geweke's formula for Granger Causality
                if np.var(joint_residuals) == 0:
                    granger_causality = 0
                else:
                    granger_causality = np.log(np.var(independent_residuals) /
                                               np.var(joint_residuals))

                # Calculate Linear Transfer Entropy from Granger Causality
                transfer_entropies[i] = granger_causality/2

            TEs.append(transfer_entropies)

            # Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(df=df,
                                             TE=transfer_entropies,
                                             endog=self.endog,
                                             exog=self.exog,
                                             lag=self.lag,
                                             n_shuffles=n_shuffles,
                                             method='linear')

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        # Store Linear Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results({'TE_linear_XY': np.array(TEs)[:, 0],
                          'TE_linear_YX': np.array(TEs)[:, 1],
                          'p_value_linear_XY': None,
                          'p_value_linear_YX': None,
                          'z_score_linear_XY': 0,
                          'z_score_linear_YX': 0
                          })

        if n_shuffles > 0:
            # Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)

            self.add_results({'p_value_linear_XY': np.array(p_values)[:, 0],
                              'p_value_linear_YX': np.array(p_values)[:, 1],
                              'z_score_linear_XY': np.array(z_scores)[:, 0],
                              'z_score_linear_YX': np.array(z_scores)[:, 1],
                              'Ave_TE_linear_XY': np.array(shuffled_TEs)[:, 0],
                              'Ave_TE_linear_YX': np.array(shuffled_TEs)[:, 1]
                              })

        return transfer_entropies

    def nonlinear_TE(self, df=None, pdf_estimator='histogram', bins=None, bandwidth=None, gridpoints=20, n_shuffles=0):
        """
        NonLinear Transfer Entropy for directional causal inference

        Defined:            TE = TE_XY - TE_YX      where TE_XY = H(Y|Y-t) - H(Y|Y-t,X-t)
        Calculated using:   H(Y|Y-t,X-t) = H(Y,Y-t,X-t) - H(Y,Y-t)  and finding joint entropy through density estimation

        Arguments:
            pdf_estimator   -   (string)    'Histogram' or 'kernel' Used to define which method is preferred for density estimation
                                            of the distribution - either histogram or KDE
            bins            -   (dict of lists) Optional parameter to provide hard-coded bin-edges. Dict keys 
                                            must contain names of variables - including lagged columns! Dict values must be lists
                                            containing bin-edge numerical values. 
            bandwidth       -   (float)     Optional parameter for custom bandwidth in KDE. This is a scalar multiplier to the covariance
                                            matrix used (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.covariance_factor.html)
            gridpoints      -   (integer)   Number of gridpoints (in each dimension) to discretise the probablity space when performing
                                            integration of the kernel density estimate. Increasing this gives more precision, but significantly
                                            increases execution time
            n_shuffles      -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to 
                                            perform significance testing.

        Returns:
            transfer_entropies  -  (list) Directional Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively

        (Also stores TE, Z-score and p-values in self.results - for each window if windows defined.)
        """
        # Retrieve user-defined bins
        self.bins = bins
        if self.bins is None:
            self.bins = {self.endog: None}

        # Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        # Loop over all windows
        for j, df in enumerate(self.df):
            df = deepcopy(df)

            # Shows user that something is happening
            # if self.lts.has_windows is True and debug:
            #    print("Window ending: ", self.date_index[j])

            # Initialise list to return TEs
            transfer_entropies = [0, 0]

            # Require us to compare information transfer bidirectionally
            for i, (X, Y) in enumerate({self.exog: self.endog, self.endog: self.exog}.items()):
                # Entropy calculated using Probability Density Estimation:
                # Following: https://stat.ethz.ch/education/semesters/SS_2006/CompStat/sk-ch2.pdf
                # Also: https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec5.pdf

                # Note Lagged Terms
                X_lagged = X+'_lag'+str(self.lag)
                Y_lagged = Y+'_lag'+str(self.lag)

                # Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)
                # 1. H(Y,Y-t,X-t)
                H1 = get_entropy(df=df[[Y, Y_lagged, X_lagged]],
                                 gridpoints=gridpoints,
                                 bandwidth=bandwidth,
                                 estimator=pdf_estimator,
                                 bins={k: v for (k, v) in self.bins.items()
                                       if k in [Y, Y_lagged, X_lagged]},
                                 covar=self.covars[i][0])

                # 2. H(Y-t,X-t)
                H2 = get_entropy(df=df[[X_lagged, Y_lagged]],
                                 gridpoints=gridpoints,
                                 bandwidth=bandwidth,
                                 estimator=pdf_estimator,
                                 bins={k: v for (k, v) in self.bins.items()
                                       if k in [X_lagged, Y_lagged]},
                                 covar=self.covars[i][1])
                #print('\t', H2)
                # 3. H(Y,Y-t)
                H3 = get_entropy(df=df[[Y, Y_lagged]],
                                 gridpoints=gridpoints,
                                 bandwidth=bandwidth,
                                 estimator=pdf_estimator,
                                 bins={k: v for (k, v) in self.bins.items()
                                       if k in [Y, Y_lagged]},
                                 covar=self.covars[i][2])
                #print('\t', H3)
                # 4. H(Y-t)
                H4 = get_entropy(df=df[[Y_lagged]],
                                 gridpoints=gridpoints,
                                 bandwidth=bandwidth,
                                 estimator=pdf_estimator,
                                 bins={k: v for (k, v) in self.bins.items()
                                       if k in [Y_lagged]},
                                 covar=self.covars[i][3])

                # Calculate Conditonal Entropy using: H(Y|X-t,Y-t) = H(Y,X-t,Y-t) - H(X-t,Y-t)
                conditional_entropy_joint = H1 - H2

                # And Conditional Entropy independent of X(t) H(Y|Y-t) = H(Y,Y-t) - H(Y-t)
                conditional_entropy_independent = H3 - H4

                # Directional Transfer Entropy is the difference between the conditional entropies
                transfer_entropies[i] = conditional_entropy_independent - \
                    conditional_entropy_joint

            TEs.append(transfer_entropies)

            # Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(df=df,
                                             TE=transfer_entropies,
                                             endog=self.endog,
                                             exog=self.exog,
                                             lag=self.lag,
                                             n_shuffles=n_shuffles,
                                             pdf_estimator=pdf_estimator,
                                             bins=self.bins,
                                             bandwidth=bandwidth,
                                             method='nonlinear')

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        # Store Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results({'TE_XY': np.array(TEs)[:, 0],
                          'TE_YX': np.array(TEs)[:, 1],
                          'p_value_XY': None,
                          'p_value_YX': None,
                          'z_score_XY': 0,
                          'z_score_YX': 0
                          })
        if n_shuffles > 0:
            # Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)

            self.add_results({'p_value_XY': np.array(p_values)[:, 0],
                              'p_value_YX': np.array(p_values)[:, 1],
                              'z_score_XY': np.array(z_scores)[:, 0],
                              'z_score_YX': np.array(z_scores)[:, 1],
                              'Ave_TE_XY': np.array(shuffled_TEs)[:, 0],
                              'Ave_TE_YX': np.array(shuffled_TEs)[:, 1]
                              })
        return transfer_entropies

    def add_results(self, dict):
        """
        Args:
            dict    -   JSON-style data to store in existing self.results DataFrame
        Returns:
            n/a
        """
        for (k, v) in dict.items():
            self.results[str(k)] = v


def significance(df, TE, endog, exog, lag, n_shuffles, method, pdf_estimator=None, bins=None, bandwidth=None,  both=True):
    """
    Perform significance analysis on the hypothesis test of statistical causality, for both X(t)->Y(t)
    and Y(t)->X(t) directions

    Calculated using:  Assuming stationarity, we shuffle the time series to provide the null hypothesis. 
                       The proportion of tests where TE > TE_shuffled gives the p-value significance level.
                       The amount by which the calculated TE is greater than the average shuffled TE, divided
                       by the standard deviation of the results, is the z-score significance level.

    Arguments:
        TE              -      (list)    Contains the transfer entropy in each direction, i.e. [TE_XY, TE_YX]
        endog           -      (string)  The endogenous variable in the TE analysis being significance tested (i.e. X or Y) 
        exog            -      (string)  The exogenous variable in the TE analysis being significance tested (i.e. X or Y) 
        pdf_estimator   -      (string)  The pdf_estimator used in the original TE analysis
        bins            -      (Dict of lists)  The bins used in the original TE analysis

        n_shuffles      -      (float) Number of times to shuffle the dataframe, destroyig temporality
        both            -      (Bool) Whether to shuffle both endog and exog variables (z-score) or just exog                                  variables (giving z*-score)  
    Returns:
        p_value         -      Probablity of observing the result given the null hypothesis
        z_score         -      Number of Standard Deviations result is from mean (normalised)
    """

    # Prepare array for Transfer Entropy of each Shuffle
    shuffled_TEs = np.zeros(shape=(2, n_shuffles))

    ##
    if both is True:
        pass  # TBC

    for i in range(n_shuffles):
        # Perform Shuffle
        df = shuffle_series(df)

        # Calculate New TE
        shuffled_causality = TransferEntropy(DF=df,
                                             endog=endog,
                                             exog=exog,
                                             lag=lag
                                             )
        if method == 'linear':
            TE_shuffled = shuffled_causality.linear_TE(df, n_shuffles=0)
        else:
            TE_shuffled = shuffled_causality.nonlinear_TE(
                df, pdf_estimator, bins, bandwidth, n_shuffles=0)
        shuffled_TEs[:, i] = TE_shuffled

    # Calculate p-values for each direction
    p_values = (np.count_nonzero(TE[0] < shuffled_TEs[0, :]) / n_shuffles,
                np.count_nonzero(TE[1] < shuffled_TEs[1, :]) / n_shuffles)

    # Calculate z-scores for each direction
    z_scores = ((TE[0] - np.mean(shuffled_TEs[0, :])) / np.std(shuffled_TEs[0, :]),
                (TE[1] - np.mean(shuffled_TEs[1, :])) / np.std(shuffled_TEs[1, :]))

    TE_mean = (np.mean(shuffled_TEs[0, :]),
               np.mean(shuffled_TEs[1, :]))

    # Return the self.DF value to the unshuffled case
    return p_values, z_scores, TE_mean

##############################################################################################################
# U T I L I T Y    C L A S S E S
##############################################################################################################


class NDHistogram():
    """
        Custom histogram class wrapping the default numpy implementations (np.histogram, np.histogramdd). 
        This allows for dimension-agnostic histogram calculations, custom auto-binning and 
        associated data and methods to be stored for each object (e.g. Probability Density etc.)
    """

    def __init__(self, df, bins=None, max_bins=15):
        """
        Arguments:
            df          -   DataFrame passed through from the TransferEntropy class
            bins        -   Bin edges passed through from the TransferEntropy class
            max_bins    -   Number of bins per each dimension passed through from the TransferEntropy class
        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.
        """
        df = sanitise(df)
        self.df = df.reindex(columns=sorted(df.columns))   # Sort axes by name
        self.max_bins = max_bins
        self.axes = list(self.df.columns.values)
        self.bins = bins
        self.n_dims = len(self.axes)

        # Bins must match number and order of dimensions
        if self.bins is None:
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
        elif set(self.bins.keys()) != set(self.axes):
            warnings.warn(
                'Incompatible bins provided - defaulting to sigma bins')
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)

        ordered_bins = [sorted(self.bins[key])
                        for key in sorted(self.bins.keys())]

        # Create ND histogram (np.histogramdd doesn't scale down to 1D)
        if self.n_dims == 1:
            self.Hist, self.Dedges = np.histogram(
                self.df.values, bins=ordered_bins[0], normed=False)
        elif self.n_dims > 1:
            self.Hist, self.Dedges = np.histogramdd(
                self.df.values, bins=ordered_bins, normed=False)

        # Empirical Probability Density Function
        if self.Hist.sum() == 0:
            print(self.Hist.shape)

            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(self.df.tail(40))

            sys.exit(
                "User-defined histogram is empty. Check bins or increase data points")
        else:
            self.pdf = self.Hist/self.Hist.sum()
            self._set_entropy_(self.pdf)

    def _set_entropy_(self, pdf):
        """
        Arguments:
            pdf   -   Probabiiity Density Function; this is calculated using the N-dimensional histogram above.
        Returns:
            n/a   
        Sets entropy for marginal distributions: H(X), H(Y) etc. as well as joint entropy H(X,Y)
        """
        # Prepare empty dict for marginal entropies along each dimension
        self.H = {}

        if self.n_dims > 1:

            # Joint entropy H(X,Y) = -sum(pdf(x,y) * log(pdf(x,y)))
            # Use masking to replace log(0) with 0
            self.H_joint = -np.sum(pdf * ma.log2(pdf).filled(0))

            # Single entropy for each dimension H(X) = -sum(pdf(x) * log(pdf(x)))
            for a, axis_name in enumerate(self.axes):
                # Use masking to replace log(0) with 0
                self.H[axis_name] = - \
                    np.sum(pdf.sum(axis=a) * ma.log2(pdf.sum(axis=a)).filled(0))
        else:
            # Joint entropy and single entropy are the same
            self.H_joint = -np.sum(pdf * ma.log2(pdf).filled(0))
            self.H[self.df.columns[0]] = self.H_joint


class AutoBins():
    """
        Prototyping class for generating data-driven binning.
        Handles lagged time series, so only DF[X(t), Y(t)] required.
    """

    def __init__(self, df, lag=None):
        """
        Args:
            df      -   (DateFrame) Time series data to classify into bins
            lag     -   (float)     Lag for data to provided bins for lagged columns also
        Returns:
            n/a
        """
        # Ensure data is in DataFrame form
        self.df = sanitise(df)
        self.axes = self.df.columns.values
        self.ndims = len(self.axes)
        self.N = len(self.df)
        self.lag = lag

    def __extend_bins__(self, bins):
        """
           Function to generate bins for lagged time series not present in self.df
        Args:   
            bins    -   (Dict of List)  Bins edges calculated by some AutoBins.method()
        Returns:
            bins    -   (Dict of lists) Bin edges keyed by column name
        """
        self.max_lag_only = True  # still temporary until we kill this

        # Handle lagging for bins, and calculate default bins where edges are not provided
        if self.max_lag_only == True:
            bins.update({fieldname + '_lag' + str(self.lag): edges
                         for (fieldname, edges) in bins.items()})
        else:
            bins.update({fieldname + '_lag' + str(t): edges
                         for (fieldname, edges) in bins.items() for t in range(self.lag)})

        return bins

    def MIC_bins(self, max_bins=15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the mutual information divided by number of bins. Only accepts data
        with two dimensions [X(t),Y(t)].
        We increase the n_bins parameter in each dimension, and take the bins which
        result in the greatest Maximum Information Coefficient (MIC)

        (Note that this is restricted to equal-width bins only.)
        Defined:            MIC = I(X,Y)/ max(n_bins)
                            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]}, 
                            n_bins = [bx,by]
        Calculated using:   argmax { I(X,Y)/ max(n_bins) }
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
        """
        if len(self.df.columns.values) > 2:
            raise ValueError(
                'Too many columns provided in DataFrame. MIC_bins only accepts 2 columns (no lagged columns)')

        min_bins = 3

        # Initialise array to store MIC values
        MICs = np.zeros(shape=[1+max_bins-min_bins, 1+max_bins-min_bins])

        # Loop over each dimension
        for b_x in range(min_bins, max_bins+1):

            for b_y in range(min_bins, max_bins+1):

                # Update parameters
                n_bins = [b_x, b_y]

                # Update dict of bin edges
                edges = {dim:  list(np.linspace(self.df[dim].min(),
                                                self.df[dim].max(),
                                                n_bins[i]+1))
                         for i, dim in enumerate(self.df.columns.values)}

                # Calculate Maximum Information Coefficient
                HDE = NDHistogram(self.df, edges)

                I_xy = sum([H for H in HDE.H.values()]) - HDE.H_joint

                MIC = I_xy / np.log2(np.min(n_bins))

                MICs[b_x-min_bins][b_y-min_bins] = MIC

        # Get Optimal b_x, b_y values
        n_bins[0] = np.where(MICs == np.max(MICs))[0] + min_bins
        n_bins[1] = np.where(MICs == np.max(MICs))[1] + min_bins

        bins = {dim:  list(np.linspace(self.df[dim].min(),
                                       self.df[dim].max(),
                                       n_bins[i]+1))
                for i, dim in enumerate(self.df.columns.values)}

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        # Return the optimal bin-edges
        return bins

    def knuth_bins(self, max_bins=15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the log-likelihood given data. Only accepts data
        with two dimensions [X(t),Y(t)]. 
        Derived from Matlab code provided in Knuth (2013):  https://arxiv.org/pdf/physics/0605197.pdf

        (Note that this is restricted to equal-width bins only.)
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
        """
        if len(self.df.columns.values) > 2:
            raise ValueError(
                'Too many columns provided in DataFrame. knuth_bins only accepts 2 columns (no lagged columns)')

        min_bins = 3

        # Initialise array to store MIC values
        log_probabilities = np.zeros(
            shape=[1+max_bins-min_bins, 1+max_bins-min_bins])

        # Loop over each dimension
        for b_x in range(min_bins, max_bins+1):

            for b_y in range(min_bins, max_bins+1):

                # Update parameters
                Ms = [b_x, b_y]

                # Update dict of bin edges
                bins = {dim:  list(np.linspace(self.df[dim].min(),
                                               self.df[dim].max(),
                                               Ms[i]+1))
                        for i, dim in enumerate(self.df.columns.values)}

                # Calculate Maximum log Posterior

                # Create N-d histogram to count number per bin
                HDE = NDHistogram(self.df, bins)
                nk = HDE.Hist

                # M = number of bins in total =  Mx * My * Mz ... etc.
                M = np.prod(Ms)

                log_prob = (self.N * np.log(M)
                            + gammaln(0.5 * M)
                            - M * gammaln(0.5)
                            - gammaln(self.N + 0.5 * M)
                            + np.sum(gammaln(nk.ravel() + 0.5)))

                log_probabilities[b_x-min_bins][b_y-min_bins] = log_prob

        # Get Optimal b_x, b_y values
        Ms[0] = np.where(log_probabilities == np.max(
            log_probabilities))[0] + min_bins
        Ms[1] = np.where(log_probabilities == np.max(
            log_probabilities))[1] + min_bins

        bins = {dim:  list(np.linspace(self.df[dim].min(),
                                       self.df[dim].max(),
                                       Ms[i]+1))
                for i, dim in enumerate(self.df.columns.values)}

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        # Return the optimal bin-edges
        return bins

    def sigma_bins(self, max_bins=15):
        """ 
        Returns bins for N-dimensional data, using standard deviation binning: each 
        bin is one S.D in width, with bins centered on the mean. Where outliers exist 
        beyond the maximum number of SDs dictated by the max_bins parameter, the
        bins are extended to minimum/maximum values to ensure all data points are
        captured. This may mean larger bins in the tails, and up to two bins 
        greater than the max_bins parameter suggests in total (in the unlikely case of huge
        outliers on both sides). 
        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """

        bins = {k: [np.mean(v)-int(max_bins/2)*np.std(v) + i * np.std(v) for i in range(max_bins+1)]
                for (k, v) in self.df.iteritems()}   # Note: same as:  self.df.to_dict('list').items()}

        # Since some outliers can be missed, extend bins if any points are not yet captured
        [bins[k].append(self.df[k].min())
         for k in self.df.keys() if self.df[k].min() < min(bins[k])]
        [bins[k].append(self.df[k].max())
         for k in self.df.keys() if self.df[k].max() > max(bins[k])]

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins

    def equiprobable_bins(self, max_bins=15):
        """ 
        Returns bins for N-dimensional data, such that each bin should contain equal numbers of
        samples. 
        *** Note that due to SciPy's mquantiles() functional design, the equipartion is not strictly true - 
        it operates independently on the marginals, and so with large bin numbers there are usually 
        significant discrepancies from desired behaviour. Fortunately, for TE we find equipartioning is
        extremely beneficial, so we find good accuracy with small bin counts ***
        Args:
            max_bins        -   (int)       The number of bins in each dimension
        Returns:
            bins            -   (dict)      The calculated bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """
        quantiles = np.array([i/max_bins for i in range(0, max_bins+1)])
        bins = dict(zip(self.axes, mquantiles(
            a=self.df, prob=quantiles, axis=0).T.tolist()))

        # Remove_duplicates
        bins = {k: sorted(set(bins[k])) for (k, v) in bins.items()}

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins


class _kde_(stats.gaussian_kde):
    """
    Subclass of scipy.stats.gaussian_kde. This is to enable the passage of a pre-defined covariance matrix, via the
    `covar` parameter. This is handled internally within TransferEntropy class.
    The matrix is calculated on the overall dataset, before windowing, which allows for consistency between windows,
    and avoiding duplicative computational operations, compared with calculating the covariance each window.
    Functions left as much as possible identical to scipi.stats.gaussian_kde; docs available:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    """

    def __init__(self, dataset, bw_method=None, df=None, covar=None):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bw_method=bw_method, covar=covar)

    def set_bandwidth(self, bw_method=None, covar=None):

        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance(covar)

    def _compute_covariance(self, covar):

        if covar is not None:
            try:
                self._data_covariance = covar
                self._data_inv_cov = linalg.inv(self._data_covariance)
            except Exception as e:
                print('\tSingular matrix encountered...')
                covar += 10e-6 * np.eye(*covar.shape)
                self._data_covariance = covar
                self._data_inv_cov = linalg.inv(self._data_covariance)

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = atleast_2d(np.cov(self.dataset, rowvar=1,
                                               bias=False))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2*pi*self.covariance)) * self.n


##############################################################################################################
# U T I L I T Y    F U N C T I O N S
##############################################################################################################


def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None, covar=None):
    """
        Function for non-parametric density estimation
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        pdf         -       (Numpy ndarray) Probability of a sample being in a specific 
                                        bin (technically a probability mass)
    """
    DF = sanitise(df)

    if estimator == 'histogram':
        pdf = pdf_histogram(DF, bins)
    else:
        pdf = pdf_kde(DF, gridpoints, bandwidth, covar)
    return pdf


def pdf_kde(df, gridpoints=None, bandwidth=1, covar=None):
    """
        Function for non-parametric density estimation using Kernel Density Estimation
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix).
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        If None, these are calculated from df during the 
                                        KDE analysis
    Returns:
        Z/Z.sum()   -       (Numpy ndarray) Probability of a sample being between
                                        specific gridpoints (technically a probability mass)
    """
    # Create Meshgrid to capture data
    if gridpoints is None:
        gridpoints = 20

    N = complex(gridpoints)

    slices = [slice(dim.min(), dim.max(), N)
              for dimname, dim in df.iteritems()]
    grids = np.mgrid[slices]

    # Pass Meshgrid to Scipy Gaussian KDE to Estimate PDF
    positions = np.vstack([X.ravel() for X in grids])
    values = df.values.T
    kernel = _kde_(values, bw_method=bandwidth, covar=covar)
    Z = np.reshape(kernel(positions).T, grids[0].shape)

    # Normalise
    return Z/Z.sum()


def pdf_histogram(df, bins):
    """
        Function for non-parametric density estimation using N-Dimensional Histograms
    Args:
        df            -       (DataFrame) Samples over which to estimate density
        bins          -       (Dict of lists) Bin edges for NDHistogram. 
    Returns:
        histogram.pdf -       (Numpy ndarray) Probability of a sample being in a specific 
                                    bin (technically a probability mass)
    """
    histogram = NDHistogram(df=df, bins=bins)
    return histogram.pdf


def get_entropy(df, gridpoints=15, bandwidth=None, estimator='kernel', bins=None, covar=None):
    """
        Function for calculating entropy from a probability mass 

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                        = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        entropy     -       (float)     Shannon entropy in bits
    """
    pdf = get_pdf(df, gridpoints, bandwidth, estimator, bins, covar)
    # log base 2 returns H(X) in bits
    return -np.sum(pdf * ma.log2(pdf).filled(0))


def shuffle_series(DF, only=None):
    """
    Function to return time series shuffled rowwise along each desired column. 
    Each column is shuffled independently, removing the temporal relationship.
    This is to calculate Z-score and Z*-score. See P. Boba et al (2015)
    Calculated using:       df.apply(np.random.permutation)
    Arguments:
        df              -   (DataFrame) Time series data 
        only            -   (list)      Fieldnames to shuffle. If none, all columns shuffled 
    Returns:
        df_shuffled     -   (DataFrame) Time series shuffled along desired columns    
    """
    if not only == None:
        shuffled_DF = DF.copy()
        for col in only:
            series = DF.loc[:, col].to_frame()
            shuffled_DF[col] = series.apply(np.random.permutation)
    else:
        shuffled_DF = DF.apply(np.random.permutation)

    return shuffled_DF


def plot_pdf(df, estimator='kernel', gridpoints=None, bandwidth=None, covar=None, bins=None, show=False,
             cmap='inferno', label_fontsize=7):
    """
    Wrapper function to plot the pdf of a pandas dataframe

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        estimator   -       (string)    'kernel' or 'histogram'
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        show        -       (Boolean)   whether or not to plot direclty, or simply return axes for later use
        cmap        -       (string)    Colour map (see: https://matplotlib.org/examples/color/colormaps_reference.html)
        label_fontsize -    (float)     Defines the fontsize for the axes labels
    Returns:
        ax          -       AxesSubplot object. Can be added to figures to allow multiple plots.
    """

    DF = sanitise(df)
    if len(DF.columns) != 2:
        print("DataFrame has " + str(len(DF.columns)) +
              " dimensions. Only 2D or less can be plotted")
        axes = None
    else:
        # Plot data in Histogram or Kernel form
        if estimator == 'histogram':

            if bins is None:
                bins = {axis: np.linspace(DF[axis].min(),
                                          DF[axis].max(),
                                          9) for axis in DF.columns.values}
            fig, axes = plot_pdf_histogram(df, bins, cmap)
        else:
            fig, axes = plot_pdf_kernel(df, gridpoints, bandwidth, covar, cmap)

        # Format plot
        axes.set_xlabel(DF.columns.values[0], labelpad=20)
        axes.set_ylabel(DF.columns.values[1], labelpad=20)
        for label in axes.xaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        for label in axes.yaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        for label in axes.zaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        axes.view_init(10, 45)
        if show == True:
            plt.show()
        plt.close(fig)

        axes.remove()

    return axes


def plot_pdf_histogram(df, bins, cmap='inferno'):
    """
    Function to plot the pdf of a dataset, estimated via histogram.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function
    """
    DF = sanitise(df)  # in case function called directly

    # Calculate PDF
    PDF = get_pdf(df=DF, estimator='histogram', bins=bins)

    # Get x-coords, y-coords for each bar
    (x_edges, y_edges) = bins.values()
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    # Get dx, dy for each bar
    dxs, dys = np.meshgrid(np.diff(x_edges), np.diff(y_edges))

    # Colourmap
    cmap = cm.get_cmap(cmap)
    rgba = [cmap((p-PDF.flatten().min())/PDF.flatten().max())
            for p in PDF.flatten()]

    # Create subplots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(x=X.flatten(),  # x coordinates of each bar
             y=Y.flatten(),  # y coordinates of each bar
             z=0,  # z coordinates of each bar
             dx=dxs.flatten(),  # width of each bar
             dy=dys.flatten(),  # depth of each bar
             dz=PDF.flatten(),  # height of each bar
             alpha=1,  # transparency
             color=rgba
             )
    ax.set_title("Histogram Probability Distribution", fontsize=10)

    return fig, ax


def plot_pdf_kernel(df, gridpoints=None, bandwidth=None, covar=None, cmap='inferno'):
    """
        Function to plot the pdf, calculated by KDE, of a dataset

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 

    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function
    """
    DF = sanitise(df)
    # Estimate the PDF from the data
    if gridpoints is None:
        gridpoints = 20

    pdf = get_pdf(DF, gridpoints=gridpoints, bandwidth=bandwidth)
    N = complex(gridpoints)
    slices = [slice(dim.min(), dim.max(), N)
              for dimname, dim in DF.iteritems()]
    X, Y = np.mgrid[slices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, pdf, cmap=cmap)

    ax.set_title("KDE Probability Distribution", fontsize=10)

    return fig, ax


def sanitise(df):
    """
        Function to convert DataFrame-like objects into pandas DataFrames

    Args:
        df          -        Data in pd.Series or pd.DataFrame format
    Returns:
        df          -        Data as pandas DataFrame
    """
    # Ensure data is in DataFrame form
    if isinstance(df, pd.DataFrame):
        df = df
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise ValueError(
            'Data passed as %s Please ensure your data is stored as a Pandas DataFrame' % (str(type(df))))
    return df
