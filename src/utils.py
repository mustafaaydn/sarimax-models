import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.validation.validation import array_like

# Make use of imprecise float arithmetic :)
EPS = 7.0 / 3 - 4.0 / 3 - 1


def _check_endog(endog):
    """
    Validates endog and returns a numpy array if possible.

    Parameters
    ----------
    endog : array_like
        Expected to be an array-like.

    Returns
    -------
        The numpy array-casted data.

    Raises
    ------
        ValueError if endog is not suitable i.e. not one-dimension'able.

    Notes
    -----
        If endog is None, returns None.
    """
    if endog is None:
        return None
    endog = np.asanyarray(endog)
    endog = np.squeeze(endog)
    endog = array_like(endog, "endog", ndim=1)
    return endog


def _check_exog(exog):
    """
    Validates exog and returns a numpy array if possible.

    Parameters
    ----------
    exog : array_like

    Returns
    -------
        The numpy array-casted data.

    Raises
    ------
        ValueError if exog is not suitable i.e. not two-dimension'able.

    Notes
    -----
        If endog is None, returns None.
    """
    if exog is None:
        return None
    exog = np.asanyarray(exog)
    exog = np.squeeze(exog)
    exog = array_like(exog, "exog", ndim=2)
    return exog


def _get_ordinal_design_mat_part(endog_or_resid, order):
    """
    Prepares `p, q` parts of the design matrix for SGD, which is p- (or q-)
    lagged matrix of endogenous array (or residuals).

    Parameters
    ----------
    endog_or_resid : array_like
        Either the endogenous array for p's part in the design matrix, or the
        (estimated) residuals for q's part.

    order : int
        Either p or q of SARIMAX.

    Returns
    -------
        A lagged matrix shaped (N, p) or (N, q) that will correspond to the
        multiplying terms in SARIMAX's ordinal parameters.
    """
    part = lagmat(endog_or_resid, maxlag=order)
    return part


def _get_seasonal_design_mat_part(endog_or_resid, order, m):
    """
    Prepares `P, Q` parts of the design matrix for SGD.

    Parameters
    ----------
    endog_or_resid : array_like
        Either the endogenous array for P's part in the design matrix, or the
        (estimated) residuals for Q's part.

    order : int
        Either P or Q of SARIMAX.

    m : int
        The seasonal period of SARIMAX.

    Returns
    -------
        A matrix shaped (N, P) or (N, Q) that will correspond to the
        multiplying terms in SARIMAX's seasonal-only parameters.
    """
    N = endog_or_resid.shape[0]
    part = np.empty((N, order))
    for i in range(order):
        _i = i + 1
        part[:, i] = np.r_[np.zeros(_i * m), endog_or_resid[: N - _i * m]]
    return part


def _get_mixed_design_mat_part(seas_part, order, seas_order, m):
    """
    Prepares `pP, qQ` parts of the design matrix for SGD.

    Parameters
    ----------
    seas_part : array_like
        Either the seasonal part that was previosuly calculated with
        `_get_seasonal_design_mat_part` for P or for Q.

    order : int
        Either p or q of SARIMAX.

    seas_order : int
        Either P or Q of SARIMAX.

    m : int
        The seasonal period of SARIMAX.

    Returns
    -------
        A matrix shaped (N, P) or (N, Q) that will correspond to the
        multiplying terms in SARIMAX's "mixed" parameters i.e. containing both
        ordinals and seasonals as a result of multiplicative formulation.
    """
    N = seas_part.shape[0]
    part = np.empty((N, order * seas_order))
    for i in range(seas_order):
        part[:, i * order: (i + 1) * order] = lagmat(
            seas_part[:, i], maxlag=order
        )
    return part


def generate_arma_process(ar_params=None, ma_params=None, sigma=1, n_obs=250):
    r"""
    Given AR and MA coefficients and number of observations, produces and
    returns a time series of length n_obs that is an ARMA(p, q) process.

    The process is prescribed as:

    Y_t = \phi_1 * Y_(t-1) + ... + \phi_p * Y_(t-p) +
            \theta_1 * Z_(t-1) + ... + \theta_q * Z(t-q) + Z_t

        where Z_t ~ N(0, sigma^2).

    Parameters
    ----------
    ar_params: array_like, optional, default: None
        The p-length array of \phi_1, ... \phi_p. Default is no AR component.

    ma_params: array_like, optional, default: None
        The q-length array of \theta_1, ... \theta_q. Default is no MA
        component.

    sigma: float, optional, default: 1
        The standard deviation of noise (disturbance) terms.

    n_obs: int, optional, default: 250
        Number of observations i.e. length of desired time series.

    Returns
    -------
        Numpy array of shape (n_obs, 1); representing ARMA(p, q) as prescribed.
    """
    # If no AR and/or MA params passed, make them empty list
    if ar_params is None:
        ar_params = []
    if ma_params is None:
        ma_params = []

    # Statsmodels expects params in IIR filter form
    ar_coeffs = np.r_[1, -np.array(ar_params)]
    ma_coeffs = np.r_[1, +np.array(ma_params)]

    arma_process = sm.tsa.ArmaProcess(ar_coeffs, ma_coeffs)
    arma_data = arma_process.generate_sample(nsample=n_obs, scale=sigma)
    return arma_data.reshape(-1, 1)


def plot_ts(
    *ts,
    plotting_args=None,
    fig_size=(16, 9),
    style="fivethirtyeight",
    title="Time Series",
    xlabel="t (u.t)",
    ylabel="$y_t$",
    legend=None,
    fig_ax=None,
    seconds=None,
):
    """
    Plotting utility function to plot one or more series.

    Parameters
    ----------
    ts : array_like(s)
        One or more series to plot.

    plotting_args : tuple or iterable of tuples, optional
        Either a 2-tuple of (str, dict) if one series, or a sequence of n
        2-tuples of the same form.

    fig_size : tuple, optional, default: (16, 9)
        the figure size in inches. Used iff fig_ax is None.

    style : str, optional, default: "fivethirtyeight"
        plt.style uses this.

    title : str, optional, default: "Time Series"
        title of graph.

    xlabel : str, optional, default: "t (u.t)"
        xlabel of graph.

    ylabel : str, optional, default: $y_t$"
        ylabel of graph.

    legend : iterable of str, optional, default: None
        legend of the graph.

    fig_ax : iterable, optional, default: None
        2-tuple of a figure and an axis.

    seconds: float, optional, default: None
        How many seconds should the graph pend and block the process. Defaults
        to None i.e. blocks indefinitely.

    Returns
    -------
        The figure where the plot lies.

    Notes:
    ------
        Example usage:
        plot_ts(np.arange(10), np.arange(5, 15),
                plotting_args=[("r+", {"lw":2}), ("g--", {"ms":3})],
                legend=["till 10", "5 to 15"]
        )
    """
    plt.style.use(style)
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=fig_size)

    if seconds is not None:
        plt.ion()
        plt.show()

    n_series = len(ts)
    if not isinstance(plotting_args, (tuple, list)):
        if plotting_args is None:
            plotting_args = [("", {})] * n_series
        else:
            plotting_args = [plotting_args]

    n_plot_args = len(plotting_args)
    if n_plot_args < n_series:
        plotting_args.extend([("", {})] * (n_series - n_plot_args))
    elif n_plot_args > n_series:
        plotting_args = plotting_args[:n_series]

    for series, (str_arg, kwargs) in zip(ts, plotting_args):
        ax.plot(series, str_arg, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend is not None:
        ax.legend(legend)

    if seconds is not None:
        plt.draw()
        plt.pause(seconds)
        plt.show()
    return fig


def plot_cf(data, which="both", fig_size=None, **sm_kwargs):
    """
    Plots Autocorrelation and Partial Autocorrelation of the given data in a
    subplot. If which is not `both` but `acf` or `pacf`, only one is drawn.

    Parameters
    ----------
    data : array_like
        The input data to take the acf and / or pacf.

    which : str, optional, default: "both"
        The correleation function to plot. Options are `acf`, `pacf` or `both`.

    fig_size : sequence, optional
        The figure size. If both acf and pacf to be plotted, defaults to
        (16, 9); if only one is plotted, defaults to (16, 4.5) [inches].

    sm_kwargs : dict, optional
        Passed to statsmodels' acf and pacf plotting functions. Default `lags`
        is set to 25 (so that it's > 2*12).
        (TODO: might seperate this into 3 arguments: acf_kwargs=None,
        pacf_kwargs=None, **both_kwargs; and act accordingly.)

    Returns
    -------
        The figure on which the plots lie.
    """
    # Validate which
    if which not in ("both", "acf", "pacf"):
        raise ValueError(
            "The correlation function(s) to plot not understood. Must be one"
            f" of `both`, `acf`, `pacf`; got {which}."
        )

    # Prepare the fig and ax(s)
    nrows = 2 if which == "both" else 1
    if fig_size is None:
        fig_size = (16, 9*nrows/2)
    fig, ax_s = plt.subplots(nrows=nrows, ncols=1,
                             figsize=fig_size)

    # Show maximum of 25 lags to cover at least 2 seasonal periods of monthlies
    sm_kwargs.setdefault("lags", 25)

    # Plot the correlations
    if which != "both":
        # First make the ax_s a list for convenience even if it's one item
        ax_s = [ax_s]
        # Then choose the suitable function and plot
        corr_fun = getattr(sm.tsa.graphics, f"plot_{which}")
        corr_fun(data, ax=ax_s[0], **sm_kwargs)

    else:
        sm.tsa.graphics.plot_acf(data, ax=ax_s[0], **sm_kwargs)
        sm.tsa.graphics.plot_pacf(data, ax=ax_s[1], **sm_kwargs)

    # Get the maxlag to adjust xticks nicely
    maxlag = int(ax_s[0].get_xlim()[1])
    for ax in ax_s:
        ax.set_xticks(np.arange(maxlag + 1))

    plt.show()
    return fig


def is_causal(ar_coeffs):
    r"""
    Checks if the given \phi_1 .. \phi_p vector begets
    a causal i.e. stationary ARMA process.

    Parameters
    ----------
    ar_params: array_like
        The p-length array of \phi_1, ... \phi_p.

    Returns
    -------
        True if polynomial (1 - \phi_1*z - ... - \phi_p*z^p) has no root
        outside the unit circle. False otherwise.
    """
    # Stationary iff all roots of 1 - \phi_1*z - .. -\phi_p*z^p lie
    # outside the unit circle
    coeffs = np.r_[1, -np.array(ar_coeffs)]
    poly = np.polynomial.Polynomial(coeffs)
    roots = poly.roots()
    return np.all(np.abs(roots) - 1 > EPS)


def is_invertible(ma_coeffs):
    r"""
    Checks if the given \theta_1 .. \theta_q vector begets
    an invertible ARMA process.

    Parameters
    ----------
    ma_params: array_like
        The q-length array of \theta_1, ... \theta_q.

    Returns
    -------
        True if polynomial (1 + \theta_1*z - ... - \theta_q*z^q) has no root
        outside the unit circle. False otherwise.
    """
    # Invertible iff all roots of 1 + \theta_1*z - .. + \theta_p*z^q lie
    # outside the unit circle
    coeffs = np.r_[1, np.array(ma_coeffs)]
    poly = np.polynomial.Polynomial(coeffs)
    roots = poly.roots()
    return np.all(np.abs(roots) - 1 > EPS)


def is_stationary(data, test="kpss"):
    # TODO: first test seasonality? then stationarity and suggest d, D?
    raise NotImplementedError


def ordinal_diff(data, d=1):
    """
    Takes ordinal differences of the given data. `d` elements from the
    beginning are cropped i.e. resultant array's numel is N-d.

    Parameters
    ----------
    data : array_like
        The data to take the ordinal difference(s) on. Remains unaltered.

    d : int, optional, default: 1
        The number of differences. Ought to be a non-negative integer.

    Returns
    -------
        The differenced data.
    """
    return np.diff(data, n=d, axis=0)


def seasonal_diff(data, D, m):
    """
    Wrapper around statsmodels.tsa.statespace.tools.diff for seasonal diff.
    The `m*D` elements from the beginning are cropped i.e. resultant array's
    numel is N-m*D.

    Parameters
    ----------
    data : array_like
        The input data to take the seasonal difference on.

    D : int
        The seasonal differencing order. Nonnegative int.

    m : int
        The seasonal period e.g. 4 for quarterly data.

    Returns
    -------
        D-seasonally differenced data.
    """
    return sm.tsa.statespace.tools.diff(
                data, k_diff=0, k_seasonal_diff=D, seasonal_periods=m
            )


def shift(data, periods=1, to="right", circular=False, fill_val=None,
          crop=False):
    """
    Shift array_like data.

    Parameters
    ----------
    data : array_like
        The input data to shift. Remains unmodified.

    periods : int, optional, default: 1
        Number of periods to shift. Should be non-negative. For negative
        periods, change the direction of shift. See `to`.

    to : string, optional, default: "right"
        The direction of shift. Either "right" or "left"

    circular : bool, optional, default: False
        Whether the shifting is circular.

    fill_val : float, optional, default: np.nan
        The value to fill the gaps when shifted. Ignored if circular is True.

    crop : bool, optional, default: False
        If True, `periods` elements will be thrown away i.e. the resultant
        array's numel would be `N-periods`. Circular and fill_val will be
        ignored if this is in effect.

    Returns
    -------
    The shifted array like that has the same shape as the input.
    """
    data = np.atleast_1d(np.asanyarray(data))
    if data.ndim >= 2:
        raise ValueError("Array-likes of 2D or more can't be shifted (yet).")

    if crop and (circular or fill_val is not None):
        warnings.warn(
            "`crop` is True but you also supplied a `fill_val` or opted for "
            "`circular` operation. The latter(s) will be ignored."
        )

    if circular and fill_val is not None:
        warnings.warn(
            "`circular` is True but you also supplied a `fill_val`; fill value"
            " will be ignored."
        )

    if not circular and fill_val is None:
        fill_val = np.nan

    if to not in ("left", "right"):
        raise ValueError(f"`to` must be either `left` or `right`; got {to}.")

    numel = data.size
    periods = periods % numel

    # The return value
    rv = None
    if periods == 0:
        rv = data

    if not circular:
        filler = np.full((periods,), fill_val, dtype=data.dtype)

    if to == "right":
        if circular:
            filler = data[-periods:]
        rv = np.r_[filler, data[:numel - periods]]
    else:
        if circular:
            filler = data[:periods]
        rv = np.r_[data[periods:], filler]

    if crop:
        rv = rv[periods:]
    return rv
