from pmdarima import arima, auto_arima, pipeline
from pmdarima.preprocessing import FourierFeaturizer

from tsaugur.utils import data_utils
from tsaugur.models import base_model


class FourierSarima(base_model.BaseModel):
    """
    Auto-Regression Integrated Moving Average with seasonality captured by Fourier terms.
    """

    def _tune(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        """
        Tune hyperparameters of the model.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: not used for TBATS model
        :param metric: Str, the metric used for model selection. One of "mse" (mean squared error), "mae" (mean absolute
        error).
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        if verbose:
            print("Tuning FourierSARIMA parameters...")
        self.period = data_utils._period_to_int(period) if type(period) == str else period
        val_size = int(len(y) * .1) if val_size is None else val_size
        pipe = pipeline.Pipeline([
            ("fourier", FourierFeaturizer(self.period, self.period / 2)),  # TODO: Tune no. of Fourier terms as well?
            ("arima", auto_arima(y, m=self.period, seasonal=False, d=None, information_criterion='oob', maxiter=100,
                                 error_action='ignore', suppress_warnings=True, stepwise=True, max_order=None,
                                 out_of_sample_size=val_size, scoring=metric, exogenous=x))
        ])
        self.params.update(pipe.steps[1][1].get_params())
        self.params["tuned"] = True

    def fit(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        """
        Build the model using best-tuned hyperparameter values.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :param metric: Str, the metric used for model selection. One of "mse" (mean squared error), "mae" (mean absolute
        error).
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        self._tune(y=y, period=period, x=x, metric=metric, val_size=val_size, verbose=verbose)
        pipe = pipeline.Pipeline([
            ("fourier", FourierFeaturizer(self.period, self.period / 2)),
            ("arima", arima.ARIMA(maxiter=100, order=self.params["order"], seasonal_order=self.params["seasonal_order"],
                                  suppress_warnings=True))
        ])
        self.model = pipe.fit(y, exogenous=x)

    def predict(self, horizon, x=None):
        """
        Predict future values of the time series using the fitted model.
        :param horizon: Int, the number of observations in the future to predict
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        :return: 1-D np.array with predictions
        """
        return self.model.predict(n_periods=horizon, exogenous=x)

