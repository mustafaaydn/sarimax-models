import warnings
from collections import deque
from itertools import chain

import numpy as np
from sklearn.linear_model import SGDRegressor
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen

from base_sarimax import SARIMAX_Base
from utils import (_get_mixed_design_mat_part, _get_ordinal_design_mat_part,
                   _get_seasonal_design_mat_part, ordinal_diff, seasonal_diff)


class SARIMAX_SGD(SARIMAX_Base):
    """
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    model. Solved with Stochastic Gradient Descent.
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
            nonnegative integers.
        """
        super(SARIMAX_SGD, self).__init__(endog, exog, order, seas_order)

        # Standardize the endog
        self.endog = (self.endog - self.endog.mean()) / self.endog.std()

        # Save original mean and std
        self._orig_endog_mean = self.the_endog.mean()
        self._orig_endog_std = self.the_endog.std()

        # Revise the_endog
        self.the_endog = self.endog.copy()

        # The SGD Regressor-related
        self.sgd_reg = None
        self.design_matrix = None

        # Store in sample preds i.e. "cache"
        self.preds_in_sample = None
        self.raw_preds_in_sample = None

    def fit(self, loss="squared_loss", *, penalty="l2", alpha=0.0001,
            l1_ratio=0.15, trend="n", max_iter=20_000, tol=None, shuffle=False,
            verbose=0, epsilon=0.1, random_state=None,
            learning_rate="invscaling", eta0=0.01, power_t=0.25,
            early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
            warm_start=False, average=False):
        """
        Fits the SARIMAX model via Stochastic Gradient Descent Regressor. Uses
        sklearn.linear_model.SGDRegressor. The design matrix is composed of
        lagged endog, lagged (estimated) residuals and possibly trend related
        components (e.g. constant or linear trend).

        Parameters
        ----------
        loss : str, default='squared_loss'
            The loss function to be used. The possible values are
            'squared_loss', 'huber', 'epsilon_insensitive', or
            'squared_epsilon_insensitive'

            The 'squared_loss' refers to the ordinary least squares fit.
            'huber' modifies 'squared_loss' to focus less on getting outliers
            correct by switching from squared to linear loss past a distance
            of epsilon. 'epsilon_insensitive' ignores errors less than epsilon
            and is linear past that. 'squared_epsilon_insensitive' is the same
            but becomes squared loss past a tolerance of epsilon.

        penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
            The penalty (aka regularization term) to be used. Defaults to 'l2'
            which is the standard regularizer for linear SVM models. 'l1' and
            'elasticnet' might bring sparsity to the model (feature selection)
            not achievable with 'l2'.

        alpha : float, default=0.0001
            Constant that multiplies the regularization term. The higher the
            value, the stronger the regularization.
            Also used to compute the learning rate when set to `learning_rate`
            is set to 'optimal'.

        l1_ratio : float, default=0.15
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
            Only used if `penalty` is 'elasticnet'.

        trend: string {"n", "c", "t", "ct"}, optional -> defaults to "n"
            The deterministic trend component. "n" for no such trend, "c" for
            constant trend i.e. bias, "t" for linear trend and "ct" for both
            constant and linear.

        max_iter : int, default=1000
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the ``fit`` method, and not the
            :meth:`partial_fit` method.

        tol : float, default=1e-3
            The stopping criterion. If it is not None, training will stop
            when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
            epochs.

        shuffle : bool, default=False
            Whether or not the training data should be shuffled after each
            epoch.

        verbose : int, default=0
            The verbosity level.

        epsilon : float, default=0.1
            Epsilon in the epsilon-insensitive loss functions; only if `loss`is
            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
            For 'huber', determines the threshold at which it becomes less
            important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current
            prediction and the correct label are ignored if they are less than
            this threshold.

        random_state : int, RandomState instance, default=None
            Used for shuffling the data, when ``shuffle`` is set to ``True``.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        learning_rate : string, default='invscaling'
            The learning rate schedule:

            - 'constant': `eta = eta0`
            - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
            where t0 is chosen by a heuristic proposed by Leon Bottou.
            - 'invscaling': `eta = eta0 / pow(t, power_t)`
            - 'adaptive': eta = eta0, as long as the training keeps decreasing.
            Each time n_iter_no_change consecutive epochs fail to decrease the
            training loss by tol or fail to increase validation score by tol if
            early_stopping is True, the current learning rate is divided by 5.

        eta0 : double, default=0.01
            The initial learning rate for the 'constant', 'invscaling' or
            'adaptive' schedules. The default value is 0.01.

        power_t : double, default=0.25
            The exponent for inverse scaling learning rate.

        early_stopping : bool, default=False
            Whether to use early stopping to terminate training when validation
            score is not improving. If set to True, it will automatically set
            asidea fraction of training data as validation and terminate
            training when validation score returned by the `score` method is
            not improving by at least `tol` for `n_iter_no_change` consecutive
            epochs.

        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set for
            early stopping. Must be between 0 and 1.
            Only used if `early_stopping` is True.

        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early
            stopping.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            See :term:`the Glossary <warm_start>`.

            Repeatedly calling fit or partial_fit when warm_start is True can
            result in a different solution than when calling fit a single time
            because of the way the data is shuffled.
            If a dynamic learning rate is used, the learning rate is adapted
            depending on the number of samples already seen. Calling ``fit``
            resets this counter, while ``partial_fit``  will result in
            increasing the existing counter.

        average : bool or int, default=False
            When set to True, computes the averaged SGD weights accross all
            updates and stores the result in the ``coef_`` attribute. If set to
            an int greater than 1, averaging will begin once the total number
            of samples seen reaches `average`. So ``average=10`` will begin
            averaging after seeing 10 samples.

        Returns
        -------
            self
        """

        # Validate trend
        if trend not in ("n", "c", "t", "ct"):
            raise ValueError(
                f"Trend must be one of `n`, `c`, `t`, `ct`; got {trend}"
            )
        self.trend = trend

        # Should include bias?
        if trend.startswith("c"):
            fit_intercept = True
        else:
            fit_intercept = False

        # Prepare X for SGDRegressor; y is self.endog
        if self.design_matrix is None:
            self.design_matrix = self._get_design_mat()

        # Make the SGDRegressor
        self.sgd_reg = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    fit_intercept=fit_intercept,
                                    max_iter=max_iter, tol=tol,
                                    shuffle=shuffle,
                                    verbose=verbose, epsilon=epsilon,
                                    random_state=random_state,
                                    learning_rate=learning_rate, eta0=eta0,
                                    power_t=power_t,
                                    early_stopping=early_stopping,
                                    validation_fraction=validation_fraction,
                                    n_iter_no_change=n_iter_no_change,
                                    warm_start=warm_start, average=average)

        # Fit it
        self.sgd_reg = self.sgd_reg.fit(self.design_matrix, self.endog)

        # Set params
        self._set_params()
        return self

    def predict_in_sample(self, start=None, end=None, raw=False):
        """
        Generates in-sample i.e. training predictions.

        Parameters
        ----------
        start : int, optional  TODO: add datetime-like support
            The starting time of is-predictions. Default is the beginning.

        end : int, optional  TODO: add datetime-like support
            The end time of is-predictions. Default is the last observation.

        raw : bool, optional, default: False
            If True, then the predictions that are about the differenced data
            are returned. If d == D == 0, this has no effect. If not, then
            the integrated predictions are returned.

        Returns
        -------
            ndarray of length (end-start+1).
        """
        if self.sgd_reg is None:
            raise RuntimeError("Can't predict in sample prior to fit.")

        # Start defaults to beginning
        # don't check `end is None` as None gives the end in slicing anyway
        if start is None:
            start = 0

        # If already calculated, return early
        if self.preds_in_sample is not None:
            if raw:
                return self.raw_preds_in_sample[start:end]
            else:
                return self.preds_in_sample[start:end]

        # Get the "raw" predictions
        preds = self.sgd_reg.predict(self.design_matrix)
        if self.raw_preds_in_sample is None:
            self.raw_preds_in_sample = preds.copy()
        if raw:
            return preds[start:end]

        # Undo ordinary and seasonal differences
        preds = self.undo_differences(preds)

        # As we fitted to normalized endog, undo that
        preds = preds * self._orig_endog_std + self._orig_endog_mean

        # Store so that if asked again, give from "cache"
        self.preds_in_sample = preds

        return preds[start:end]

    def forecast(self, steps=1, method=None):
        """
        Get out-of-sample forecasts.

        Parameters
        ----------
        steps : int, optional -> defaults to 1
            The step size into future till which we forecast.
        method: str, optional
            The forecast method. "rec" for recursive (or plug-in), "direct" for
            multi-step direct, "rectify" for combination. Defaults to "rec".
        """
        # Check if already fitted
        if self.sgd_reg is None:
            raise RuntimeError("Can't forecast before fit")

        # Unpack orders for ease
        p, d, q = self.order
        P, D, Q, m = self.seas_order

        # Get the reduced i.e. multiplied AR and MA parameters' estimates
        tmp_ar_coeffs = -self.params.reduced_ar_poly.coef[1:]
        tmp_ar_size = tmp_ar_coeffs.size
        ar_coeffs = np.r_[tmp_ar_coeffs, np.zeros((m*P+p)-tmp_ar_size)]

        tmp_ma_coeffs = self.params.reduced_ma_poly.coef
        tmp_ma_size = tmp_ma_coeffs.size
        ma_coeffs = np.r_[tmp_ma_coeffs, np.zeros((m*Q+q+1)-tmp_ma_size)]

        # Get last mP+p observations and mQ+q residuals (as deques)
        last_obs = deque(self.endog[-1: - (m * P + p + 1):-1],
                         maxlen=m * P + p)

        preds_in_sample = self.predict_in_sample(raw=True)
        resids = self.endog - preds_in_sample
        last_resids = deque(np.r_[0, resids[-1:(m * Q + q + 1):-1]],
                            maxlen=m * Q + q + 1)

        # Forecast loop
        forecasts = np.empty(steps)
        xtended_endog = self.the_endog[-(m * D + d):].tolist()
        for h in range(steps):
            # Get the h'th "raw" forecast
            forecasts[h] = ar_coeffs.dot(last_obs) + ma_coeffs.dot(last_resids)

            # Undo ordinary differences
            for i in range(d):
                forecasts[h] += ordinal_diff(xtended_endog, i)[-1]

            ord_diffed_endog = ordinal_diff(xtended_endog, d)

            # Undo seasonal differences
            for j in range(D):
                forecasts[h] += seasonal_diff(
                                    ord_diffed_endog, j, m
                                )[-m]

            # Prepend predictions to `y` and zeros to future errors
            last_obs.appendleft(forecasts[h])
            last_resids.appendleft(0)

        # Undo standardization
        forecasts = forecasts * self._orig_endog_std + self._orig_endog_mean

        return forecasts

    @property
    def ar_params(self):
        empty = np.array([])
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            params = empty
        else:
            p = self.order[0]
            if p > 0:
                params = self.sgd_reg.coef_[:p]
            else:
                params = empty
        return params

    @property
    def ma_params(self):
        empty = np.array([])
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            params = empty
        else:
            if self.q > 0:
                lower = self.p + self.P + self.p*self.P
                params = self.sgd_reg.coef_[lower:lower+self.q]
            else:
                params = empty
        return params

    @property
    def seas_ar_params(self):
        empty = np.array([])
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            params = empty
        else:
            if self.P > 0:
                lower = self.p
                params = self.sgd_reg.coef_[lower:lower+self.P]
            else:
                params = empty
        return params

    @property
    def seas_ma_params(self):
        empty = np.array([])
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            params = empty
        else:
            if self.Q > 0:
                lower = self.p + self.P + self.p * self.P + self.q
                params = self.sgd_reg.coef_[lower:lower + self.Q]
            else:
                params = empty
        return params

    @property
    def bias(self):
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            return np.array([])
        if not self.trend.startswith("c"):
            warnings.warn("No intercept specified.")
            return np.array([])
        return self.sgd_reg.intercept_

    @property
    def drift(self):
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            return np.array([])
        if self.trend not in ("t", "ct"):
            warnings.warn("No linear trend specified.")
            return np.array([])
        return self.sgd_reg.coef_[-1]

    @property
    def exog_params(self):
        """
        Notes
        -----
            Doesn't include bias or trend components.
        """
        if self.sgd_reg is None:
            warnings.warn("You should query params after fitting.")
            return np.array([])
        if self.exog is None:
            warnings.warn("No exogenous variable provided.")
            return np.array([])

        # Extract exogenous coeffs
        num_side_info = self.exog.shape[1]
        lower = (self.p + self.P + self.p * self.P + self.q + self.Q +
                 self.q * self.Q)

        return self.sgd_reg.coef_[lower:lower + num_side_info]

    def _set_params(self):
        """
        Sets SARIMAXParams of the object.
        .exog_params *does* include bias and trend.
        """
        params = self.params
        params.ar_params = self.ar_params
        params.ma_params = self.ma_params
        params.seasonal_ar_params = self.seas_ar_params
        params.seasonal_ma_params = self.seas_ma_params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params.exog_params = np.r_[self.bias, self.drift, self.exog_params]

    def _get_design_mat(self):
        """
        Generates a design matrix of dimension N x (p+P+pP+q+Q+qQ+k+1). The
        endog-related entries are that of differenced data, and the residual-
        related entries are that of estimated ones (e.g. via HR).

        Returns
        -------
        The design matrix which has 8 parts (max): p part, P part, pP
                part, q part, Q part, qQ part, k part and trend part.
                If any of these is zero, dimension diminishes.
        """
        # Apply differences and get the number of observations after
        if self.d != 0:
            self.endog = ordinal_diff(self.endog, d=self.d)

            if self.exog is not None:
                self.exog = ordinal_diff(self.exog, d=self.d)

        if self.D != 0:
            self.endog = seasonal_diff(self.endog, D=self.D,
                                       m=self.seas_period)

            if self.exog is not None:
                self.exog = seasonal_diff(self.exog, D=self.D,
                                          m=self.seas_period)

        N = self.endog.size

        # Get estimates for residuals if we have MA terms
        if self.q > 0 or self.Q > 0:
            # via hannan-risanen
            reduced_ar_poly = list(chain.from_iterable([[1] * self.p,
                                   *[[0]*(self.seas_period - self.p - 1) +
                                   [1]*(self.p + 1)
                                   for _ in range(self.P)]]))

            reduced_ma_poly = list(chain.from_iterable([[1] * self.q,
                                   *[[0]*(self.seas_period - self.q - 1) +
                                   [1]*(self.q + 1)
                                   for _ in range(self.Q)]]))

            hr, hr_results = hannan_rissanen(self.endog,
                                             ar_order=reduced_ar_poly,
                                             ma_order=reduced_ma_poly,
                                             demean=False)
            hr_resid = hr_results.resid
            resid_numel = hr_resid.size
            self.resid = np.r_[np.zeros(N-resid_numel), hr_resid]

            # via statsmodels' sarimax
            # resid = sm.tsa.SARIMAX(
            #                        self.the_endog, self.the_exog,
            #                        order=self.order,
            #                        seasonal_order=self.seas_order
            #                       ).fit(disp=False).resid
            # self.resid = resid[self.d + self.seas_period * self.D:]

        # The 8 parts begin
        # Account for non-existent parts
        empty_part = np.array([]).reshape(N, 0)

        # AR terms
        # p part: Ordinal AR order part that accounts for \phi_i
        p_part = (
            _get_ordinal_design_mat_part(self.endog, self.p)
            if self.p > 0
            else empty_part
        )

        # P part: Seasonal AR order part that accounts for \PHI_i
        P_part = (
            _get_seasonal_design_mat_part(self.endog, self.P, self.seas_period)
            if self.P > 0
            else empty_part
        )

        # pP part: Mixed AR order part, result of multiplication
        pP_part = (
            _get_mixed_design_mat_part(
                P_part, self.p, self.P, self.seas_period
            )
            if self.p * self.P > 0
            else empty_part
        )

        # MA terms
        # q part: Ordinal MA order part that accounts for \theta_j
        q_part = (
            _get_ordinal_design_mat_part(self.resid, self.q)
            if self.q > 0
            else empty_part
        )

        # Q part: Seasonal MA order part that accounts for \THETA_j
        Q_part = (
            _get_seasonal_design_mat_part(self.resid, self.Q, self.seas_period)
            if self.Q > 0
            else empty_part
        )

        # qQ part: Mixed MA order part, result of multiplication
        qQ_part = (
            _get_mixed_design_mat_part(
                Q_part, self.q, self.Q, self.seas_period
            )
            if self.q * self.Q > 0
            else empty_part
        )

        # Exog
        exog_part = (
            self.exog
            if self.exog is not None
            else empty_part
        )

        # Deterministic liner trend
        trend_part = (
            np.arange(1, N + 1).reshape(N, 1)
            if self.trend in ("t", "ct")
            else empty_part
        )

        # Combine them all
        return np.hstack(
            (p_part, P_part, pP_part, q_part, Q_part, qQ_part,
                exog_part, trend_part)
        )
