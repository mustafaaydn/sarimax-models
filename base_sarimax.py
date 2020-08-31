from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification

from utils import _check_endog, _check_exog, ordinal_diff, seasonal_diff, shift


class SARIMAX_Base:
    """
    Base class for SARIMAX models.
    """
    def __init__(self, endog, exog=None, order=(1, 0, 0),
                 seas_order=(0, 0, 0, 0)):
        # Validate endog & exog
        self.endog = _check_endog(endog)
        self.exog = _check_exog(exog)

        # Save them to use in case of differencing
        self.the_endog = self.endog.copy()
        self.the_exog = self.exog.copy() if exog is not None else None

        # Orders are attributes too
        self.order = order
        self.seas_order = seas_order

        # "has" a specification and params (helps validate orders, also)
        self.spec = SARIMAXSpecification(self.the_endog, self.the_exog,
                                         self.order, self.seas_order)
        self.params = SARIMAXParams(self.spec)

        # If P == D == Q == 0, m stays the same; but should be 0, too
        if self.seas_order[:3] == (0, 0, 0):
            self.seas_order = (0, 0, 0, 0)

        # After validation, unpack order
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.seas_period = self.seas_order

        # For convenience
        self.m = self.seas_period

    def fit(self, *args, **kwargs):
        """
        Fits the model.
        """
        raise NotImplementedError

    def predict_in_sample(self, start=None, end=None):
        """
        Predicts in-sample with start and end points.
        """
        raise NotImplementedError

    def forecast(self, steps=1, exog=None, method=None):
        """
        Forecasts for out-of-sample.
        """
        raise NotImplementedError

    def undo_differences(self, preds):
        """
        Undoes the differences' effect on the predictions. Model that predicts
        over the differenced series gets back the integrated predictions.

        Parameters
        ----------
        preds : array_like
            The "raw" predictions made on stationary data.

        Returns
        -------
            The integrated predictions or `preds` itself if d = D = 0.
        """
        numel_preds = preds.size

        # Handle ordinary differences' undo
        if self.d != 0:
            preds += sum(
                        shift(
                            data=ordinal_diff(self.the_endog, i),
                            crop=True)[-numel_preds:]
                        for i in range(self.d))

        # Handle seasonal differences' undo
        if self.D != 0:
            ordi_diffed_endog = ordinal_diff(self.the_endog, self.d)
            preds += sum(
                        shift(
                            data=seasonal_diff(
                                     ordi_diffed_endog, i, self.seas_period
                                ),
                            periods=self.seas_period, crop=True
                        )[-numel_preds:]
                        for i in range(self.D)
                    )
        return preds

    def _set_params(self, *args):
        """
        Set the estimated parameters of the model.
        """
        raise NotImplementedError
