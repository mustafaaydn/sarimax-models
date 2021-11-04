# Estimates SARIMAX coefficients (and error variance) via putting it into
# a state space form and maximizing the likelihood obtained with Kalman Filter.
# Mayd, 08 / 2020

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.statespace.tools import diff

from base_sarimax import SARIMAX_Base
from utils import ordinal_diff, seasonal_diff


class SARIMAX_KF(SARIMAX_Base):
    """
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    model. Solved with Kalman Filter.
    """
    def __init__(self, endog, exog=None, order=(1, 0, 0),
                 seas_order=(0, 0, 0, 0)):
        """
        Parameters
        ----------
        endog: array_like
            The time series array.

        exog: array_like, optional -> defaults to None
            The exogenous regressors i.e. side information. Shape is n_obs x k.

        order: sequence of three ints, optional -> defaults to (1, 0, 0)
            The (p, d, q) of non-seasonal ARIMA part's order. All items must be
            nonnegative integers. Default is AR(1) process.

        seas_order: sequence of four ints, optional -> defaults to (0, 0, 0, 0)
            The (P, D, Q, m) of seasonal ARIMA part. All items must be
            nonnegative integers. Default is no seasonal components.
        """
        super(SARIMAX_KF, self).__init__(endog, exog, order, seas_order)

        # Demean the endog, save original mean and revise `the_endog`
        self.endog = self.endog - self.endog.mean()
        self._orig_endog_mean = self.the_endog.mean()
        self.the_endog = self.endog.copy()

        # scipy's MLE Result
        self.mle_result = None

        self.start_params = None

    def fit(self, start_params=None, ensure_causality=True,
            ensure_invertibility=True, **minimize_kwargs):
        R"""
        Estimates the \vec{phi}, \vec{theta}, sigma^2 and \vec{beta} via
        fitting an SARIMA-X(p, d, q)(P, D, Q, m) to `y` via MLE with Kalman
        filter.

        Parameters
        ----------
        start_params : array_like, optional
            Includes \vec{phi}, \vec{PHI}, \vec{theta}, \vec{THETA}, \sigma^2
            and \vec{beta}:

            [\phi_1,... \phi_p, \Phi_1, ... \Phi_P, \theta_1, ... \theta_q,
            \Theta_1, ... \Theta_Q, \sigma^2, \beta_1, ... \beta_k].

            Used to kick-off the MLE. Default is to use Hannan-Rissanen
            for \phi, \PHI, \theta, \THETA and \sigma^2; \beta are initialized
            to zeros.

        enforce_causality : bool, optional, default: True
            Whether constrain \vec{phi} s.t. \phi(B) has all its roots inside
            unit circle i.e. process is stationary.

        enforce_invertibility : bool, optional, default: True
            Whether constrain \vec{theta} s.t. \theta(B) has all its roots
            inside unit circle i.e. process is invertible.

        minimize_kwargs :  dict, optional
            Passed to scipy.optimize.minimize.

        Returns
        -------
            self
        """
        # Unpack orders (to prevent over-attribute acccess)
        p, d, q = self.order
        P, D, Q, m = self.seas_order

        # Difference endog and exog if needed
        if d != 0 or D != 0:
            self.endog = diff(self.endog, k_diff=d, k_seasonal_diff=D,
                              seasonal_periods=m)
            if self.exog is not None:
                self.exog = diff(self.exog, k_diff=d, k_seasonal_diff=D,
                                 seasonal_periods=m)

        # Get number of X-regressors
        k = self.exog.shape[1] if self.exog is not None else 0

        # If no initial params supplied, get it from Hannan-Rissanen
        if start_params is None:
            # TODO: Do this large AR fitting in one shot with reduced polys
            # as in sgd_sarimax's _get_design_mat. Until then, we `try`.
            try:
                # First for the non-seasonal part
                hr, hr_results = hannan_rissanen(self.endog, ar_order=p,
                                                 ma_order=q, demean=False)

                # Then the seasonal
                seas_hr_ar_order = m * np.arange(1, P + 1)
                seas_hr_ma_order = m * np.arange(1, Q + 1)
                seas_hr, _ = hannan_rissanen(
                                hr_results.resid, ar_order=seas_hr_ar_order,
                                ma_order=seas_hr_ma_order, demean=False,
                            )
            except ValueError:
                print("series too short for large AR(p) of hannan-risanen.")
                start_params = np.r_[
                                    np.zeros(p + P + q + Q + k),
                                    self.endog.var()
                                    ]
            else:
                # Stack them all
                start_params = np.hstack((hr.ar_params, seas_hr.ar_params,
                                          hr.ma_params, seas_hr.ma_params,
                                          seas_hr.sigma2, np.zeros(k)))

        # sigma^2 estimate is to be nonnegative so put a bound on it
        # bounds = ([(None, None) for _ in range(p + q + P + Q)] +
        #           [(0, None)] +
        #           [(None, None)] * k)
        bounds = None

        # Check if start_params satisfy stationarity and invertibility requests
        self.params.ar_params = start_params[:p]
        self.params.seasonal_ar_params = start_params[p:p + P]
        self.params.ma_params = start_params[p + P:p + P + q]
        self.params.seasonal_ma_params = start_params[p + P + q:p + P + q + Q]
        self.params.sigma2 = start_params[p + P + q + Q]

        if ensure_causality and not self.params.is_stationary:
            start_params[:p + P] = 0.
        if ensure_invertibility and not self.params.is_invertible:
            start_params[p + P:p + P + q + Q] = 0
        self.start_params = start_params

        # Maximize likelihood
        def _kalman_sarimax_loglike(params):
            return self.filter(params)[0]
        minimize_kwargs.setdefault("method", "BFGS")
        res = minimize(_kalman_sarimax_loglike, start_params, bounds=bounds,
                       **minimize_kwargs)
        self.mle_result = res

        # Put the estimated parameters to self.params
        self._set_params()

        return self

    def predict_in_sample(self, start=None, end=None):
        """
        Generates in-sample i.e. training predictions.

        Parameters
        ----------
        start : int, optional  TODO: add datetime-like support.
            The starting time of is-predictions. Default is the beginning.

        end : int, optional  TODO: add datetime-like support.
            The end time of is-predictions. Default is the last observation.

        Returns
        -------
            ndarray of length (end-start+1).
        """
        if self.mle_result is None:
            raise RuntimeError("Can't predict in sample prior to fit.")

        # Start defaults to beginning
        # don't check `end is None` as None gives the end in slicing anyway
        if start is None:
            start = 0

        # Get the "raw" predictions
        preds = self.filter(self.mle_result.x)[1]

        # Undo ordinary and seasonal differences
        preds = self.undo_differences(preds)

        # As we fitted to demeaned endog, undo that
        preds = preds + self._orig_endog_mean

        return preds[start:end]

    def forecast(self, steps=1, exog=None, return_vars=False):
        """
        Get out-of-sample forecasts.

        Parameters
        ----------
        steps : int, optional -> defaults to 1
            The step size into future till which we forecast.

        exog : array_like, optional
            The side information variables for the forecasting period. Should
            be shaped (steps, k).

        return_vars : bool, optional -> defaults to False
            Whether the forecast variances are also returned.

        Returns
        -------
            The h step forecasts. If `return_vars` is True, corresponding
            variances are also returned.
        """
        # Check if already fitted
        if self.mle_result is None:
            raise RuntimeError("Can't forecast before fit")

        if exog is None:
            exog = np.zeros((steps, 1))

        # Get the estimated parameters, last filtered state & its cov, and
        # the state space matrices
        params = self.mle_result.x
        *_, filtered_states, filtered_state_vars, ss_mats = self.filter(params)
        posterior_mean = filtered_states[-1]
        posterior_cov = filtered_state_vars[-1]
        A, B, Q, H = ss_mats

        # Allocate for forecasts
        forecasts = np.empty(steps)
        forecast_vars = np.empty(steps)

        # In case of differencing, we need to extended `the_endog`
        xtended_endog = self.the_endog[-(self.m * self.D + self.d):].tolist()

        # Forecast loop
        for h in range(steps):
            # Predictor equations
            prior_mean = A @ posterior_mean + B @ exog[h][:, np.newaxis]
            prior_cov = A @ posterior_cov @ A.T + Q

            # Save forecasts
            forecasts[h] = H @ prior_mean
            forecast_vars[h] = H @ prior_cov @ H.T

            # Corrector equations: No update happens here
            posterior_mean = prior_mean
            posterior_cov = prior_cov

            # Undo ordinary differences
            for i in range(self.d):
                forecasts[h] += ordinal_diff(xtended_endog, i)[-1]

            ord_diffed_endog = ordinal_diff(xtended_endog, self.d)

            # Undo seasonal differences
            for j in range(self.D):
                forecasts[h] += seasonal_diff(
                                    ord_diffed_endog, j, self.m
                                )[-self.m]

            # For recursive forecasting, record this
            xtended_endog.append(forecasts[h])

        # Add back the mean
        forecasts += self._orig_endog_mean

        return (forecasts, forecast_vars) if return_vars else forecasts

    def filter(self, params):
        R"""
        Runs Kalman recursions and computes S-AR(I)MA-X(p, q)(P, Q, m)
        loglikelihood for the given parameters. Hamilton's SS is used:

            y_t = (\theta(B) * \Theta(B^m)) @ [x_t, x_(t-1), ... x_(t-r+1)]'

            \vec{x_t} = A @ \vec{x_(t-1)} + B @ \vec{u_(t-1)} +
            [\eps_t, 0, ... 0]

        where A has \vec{phi} and \vec{Phi} in it, r := max(m*P+p, m*Q+q+1),
        \eps_t ~ N(0, \sigma^2) and \vec{u} (shape: (k, )) represent the
        exogenous i.e. side information variables - the respective coefficents
        lie in the first row of B. Note that y_t is expected to be already both
        seasonally and ordinally differenced (and de-meaned).

        Parameters
        ----------
        params : numpy.ndarray, dtype=object
            The parameter array that includes \vec{phi}, \vec{Phi},
            \vec{theta}, \vec{Theta},\sigma^2 and \vec{beta}.
            Shaped (p+P+q+Q+1+k,):

            [\phi_1,... \phi_p, \Phi_1, ... \Phi_P, \theta_1, ... \theta_q,
            \Theta_1, ... \Theta_Q, \sigma^2, \beta_1, ... \beta_k]

        Returns
        -------
            6-tuple: the negative log likelihood of the measurements with these
            parameters, measurement predictions & their variances, filtered
            states & their variances, and state space matrices i.e. A, B, Q, H.

        Notes
        -----
            The initial state mean is set to zeros and the initial covariance
            matrix is solution to `P = A @ P @ A.T + Q`, since process is
            stationary. (Even though it's SARIMAX, the endog and exog are
            already appropriately differenced and those will be undone when
            making predictions & forecasts.)

            The measurement error is assumed to be zero.
        """
        # Unpack orders to prevent over-attribute accesss
        p, _, q = self.order
        P, _, Q, m = self.seas_order

        # Unpack parameters
        phi = np.array(params[:p])
        PHI = np.array(params[p:p + P])
        theta = np.array(params[p + P: p + P + q])
        THETA = np.array(params[p + P + q: p + P + q + Q])
        sigma2 = params[p + P + q + Q]
        beta = np.array(params[p + P + q + Q + 1:])

        # We don't store k but it's available as:
        k = beta.size

        # Identify numel of observations
        T = self.endog.size

        # r is state dimension
        r = max(m * P + p, m * Q + q + 1)

        # Form phi(B) * PHI(B^m) polynomial
        seas_ar_coeffs = np.zeros(P * m + 1)
        seas_ar_coeffs[0] = 1
        if m != 0:
            seas_ar_coeffs[m:P * m + 1:m] = -PHI
        seas_ar_poly = np.polynomial.Polynomial(seas_ar_coeffs)
        ord_ar_poly = np.polynomial.Polynomial(np.r_[1, -phi])
        the_ar_coeffs = (seas_ar_poly * ord_ar_poly).coef
        temp_size_ar = the_ar_coeffs.size
        the_ar_coeffs = np.r_[the_ar_coeffs, np.zeros((P*m+1+p)-temp_size_ar)]

        # Form theta(B) * THETA(B^m) polynomial
        seas_ma_coeffs = np.zeros(Q * m + 1)
        seas_ma_coeffs[0] = 1
        if m != 0:
            seas_ma_coeffs[m:Q * m + 1:m] = THETA
        seas_ma_poly = np.polynomial.Polynomial(seas_ma_coeffs)
        ord_ma_poly = np.polynomial.Polynomial(np.r_[1, theta])
        the_ma_coeffs = (seas_ma_poly * ord_ma_poly).coef
        temp_size_ma = the_ma_coeffs.size
        the_ma_coeffs = np.r_[the_ma_coeffs, np.zeros((Q*m+1+q)-temp_size_ma)]

        # Refine "the" ar and ma coeffs by appending zeros to fit to state dim
        if r == m*P+p:
            the_ma_coeffs = np.r_[the_ma_coeffs, np.zeros(r - (m*Q+q + 1))]
        else:
            the_ar_coeffs = np.r_[the_ar_coeffs, np.zeros(r - (m * P + p))]

        # Define signal model matrices
        # Transition mat
        ar_vec = -the_ar_coeffs[1:]
        A = np.vstack(
            (
             ar_vec,
             np.hstack((np.eye(r - 1), np.zeros(r - 1)[:, np.newaxis]))
            )
        )

        # Design mat
        H = the_ma_coeffs[np.newaxis, :]

        # Controller mat (x-regressor mat)
        if k > 0:
            B = np.zeros((r, k))
            B[0] = beta
        else:
            # No external var: make B, u zero
            B = np.zeros((r, 1))
            self.exog = np.zeros((T, 1))

        # Process error covariance
        Q = np.zeros((r, r))
        Q[0, 0] = sigma2

        # Initialize mean and covariance
        x0 = np.zeros((r, 1))
        P0 = solve_discrete_lyapunov(A, Q)

        # For the sake of getting rid of `if t != 0`, predefine following
        posterior_mean = x0
        posterior_cov = P0

        # Also pre-define I_(rxr)
        EYE = np.eye(r)

        # Likelihood
        neg_logl = 0

        # Store predictions and their variances
        predictions = np.empty(T)
        prediction_vars = np.empty(T)

        # Store filtered states and their variances
        filtered_states = np.empty((T, r, 1))
        filtered_state_vars = np.empty((T, r, r))

        # Kalman loop
        for t, (y_t, u_t) in enumerate(zip(self.endog, self.exog)):
            # Predictor equations
            prior_mean = A @ posterior_mean + B @ u_t[:, np.newaxis]
            prior_cov = A @ posterior_cov @ A.T + Q

            # Corrector equations
            K = (prior_cov @ H.T) / (H @ prior_cov @ H.T)
            posterior_mean = prior_mean + K @ (y_t - H @ prior_mean)
            posterior_cov = (EYE - K @ H) @ prior_cov

            # To calculate likelihood, we need H @ P_k @ H.T (~var(z_k))
            # and H @ x_k^- (~E[z_k] i.e. the prediction)
            mean_prediction = H @ prior_mean
            var_prediction = H @ prior_cov @ H.T

            neg_logl += 0.5 * (np.log(np.abs(var_prediction)) +
                               (y_t - mean_prediction) ** 2 / var_prediction)

            # Save predictions and filtered states
            predictions[t] = mean_prediction
            prediction_vars[t] = var_prediction

            filtered_states[t] = posterior_mean
            filtered_state_vars[t] = posterior_cov

        # Add the constant for the sake of completeness
        neg_logl += 0.5 * T * np.log(2 * np.pi)

        # LLF, predicted state & its cov, filtered state & its cov
        return (neg_logl.item(),
                predictions, prediction_vars,
                filtered_states, filtered_state_vars,
                (A, B, Q, H))

    def _set_params(self, *args):
        p, _, q = self.order
        P, _, Q, __ = self.seas_order

        estimates = self.mle_result.x

        self.params.ar_params = estimates[:p]
        self.params.seasonal_ar_params = estimates[p:p + P]
        self.params.ma_params = estimates[p + P:p + P + q]
        self.params.seasonal_ma_params = estimates[p + P + q:p + P + q + Q]
        self.params.sigma2 = estimates[p + P + q + Q]
        if self.the_exog is not None:
            self.params.exog_params = estimates[p + P + q + Q + 1:]
