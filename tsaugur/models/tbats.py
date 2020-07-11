from tbats import TBATS

from tsaugur.utils import data_utils
from tsaugur.models import base_model


class Tbats(base_model.BaseModel):
    """
    Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.
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
        :param metric: not used for TBATS model; model selection is based on the AIC.
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        self.model = TBATS(seasonal_periods=[period], show_warnings=False)
        self.params["tuned"] = True

    def fit(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        """
        Build the model using best-tuned hyperparameter values.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: not used for TBATS model
        :param metric: not used for TBATS model; model selection is based on the AIC.
        :param val_size: not used for TBATS model; model selection is based on the AIC.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        self.y = y
        self.name = "TBATS"
        self.key = "tbats"
        self._tune(y=y, period=period, x=x, metric=metric, val_size=val_size, verbose=verbose)
        self.model = self.model.fit(y)

    def predict(self, horizon, x=None):
        """
        Predict future values of the time series using the fitted model.
        :param horizon: Int, the number of observations in the future to predict
        :param x: not used for TBATS model
        :return: 1-D np.array with predictions
        """
        return self.model.forecast(steps=horizon)

