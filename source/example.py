
"""
An example run of SARIMAX trained with SGD, Kalman filter and that of
statsmodels for comparison. M3 Competition monthly data is used. *pmdarima* is
used to determine an order for SARIMAX models to fit with.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima
import statsmodels.api as sm
from kalman_sarimax import SARIMAX_KF
from sgd_sarimax import SARIMAX_SGD
from utils import plot_ts


def get_m3_data(n=1, seed=None):
    """
    Prepares a list of series from M3 monthly dataset.

    Parameters
    ----------
    n : int
        `n` series in the M3 monthly dataset are returned in a list.

    seed : int, optional
        Used as a random state in sampling.

    Returns
    -------
    (list of strings [IDs in M3], list of np.ndarrays [series themselves])
    """
    subset = m3data.sample(n=n, random_state=seed)
    ids = subset.loc[:, "Series"].to_list()
    series_subset = subset.loc[:, 1:].to_numpy()
    series_list = [arr[~np.isnan(arr)] for arr in series_subset]
    return ids, series_list


# Load data and sample some series
m3data = pd.read_excel("../sample_data/M3_data.xls", sheet_name="M3Month")
n_series = 5
seed = 1284
ids, series_list = get_m3_data(n=n_series, seed=seed)

# for each series...
for id_, m_data in zip(ids, series_list):
    print(f"Series {id_}")
    # Last 18 obs are for test
    h = 18
    endog = m_data[:-h]

    # Pmd for auto detection of orders
    pmd = pmdarima.auto_arima(endog, m=12, with_intercept=False,
                              suppress_warnings=True, error_action="ignore")
    preds_pmd = pmd.predict_in_sample()
    fors_pmd = pmd.predict(h)

    p, d, q = pmd.order
    P, D, Q, m = pmd.seasonal_order
    print(f"\tOrder: {p, d, q}{P, D, Q, m}")

    # SGD way
    sgd = SARIMAX_SGD(endog, order=(p, d, q), seas_order=(P, D, Q, m))
    sfit = sgd.fit()
    preds_sgd = sfit.predict_in_sample()
    fors_sgd = sfit.forecast(h)

    # Our Kalman Filter Impl
    tkf = SARIMAX_KF(endog, order=(p, d, q), seas_order=(P, D, Q, m))
    tkfit = tkf.fit()
    preds_kf = tkf.predict_in_sample()
    fors_kf = tkf.forecast(h)

    # Statsmodels' (MLE with KF)
    sta = sm.tsa.SARIMAX(endog, order=tkf.order, seasonal_order=tkf.seas_order)
    stafit = sta.fit(disp=0)
    preds_sta = stafit.predict()
    fors_sta = stafit.forecast(h)

    # Plotting
    plot_ts(m_data[m * D + d:],
            np.r_[preds_sta[m * D + d:], fors_sta],
            np.r_[preds_kf, fors_kf],
            np.r_[preds_sgd, fors_sgd],
            legend=["Truth",
                    "Statsmodels",
                    "With Kalman filter",
                    "With SGD"],
            title=f"#{id_}: SARIMAX{p, d, q}{P, D, Q}[{m}]",
            xlabel="Month",
            legend_kwargs={"fontsize": 13, "markerscale": 3})
    test_boundary = m_data.size - (m * D + d) - 1 - h
    plt.axvline(test_boundary, c="k", ls="--")
    plt.show()
