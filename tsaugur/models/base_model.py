from tsaugur.metrics import _get_metric


class BaseModel:
    """
    Base model class. All tsaugur models inherit after this class.
    """

    def __init__(self):
        self.model = None
        self.period = None
        self.params = {"tuned": False}
        self.last_fitting_date = None
        self.time_delta = None

    def _tune(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        """
        Tune hyperparameters of the model.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :param metric: Str, the metric used for model selection.
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        raise NotImplementedError

    def fit(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        """
        Build the model using best-tuned hyperparameter values.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :param metric: Str, the metric used for model selection.
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        raise NotImplementedError

    def predict(self, horizon, x=None):
        """
        Predict future values of the time series using the fitted model.
        :param horizon: Int, the number of observations in the future to predict
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        :return: 1-D np.array with predictions
        """
        raise NotImplementedError

    def score(self, y_true, x=None, metric="smape"):
        """
        Score the model performance on a separate test set.
        :param y_true: pd.Series or 1-D np.array, ground-truth values.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        :param metric: Str, the metric to produce.
        :return: Float, model performance metric.
        """
        y_pred = self.predict(horizon=len(y_true), x=x)
        metric_fun = _get_metric(metric)
        return metric_fun(y_true, y_pred)
