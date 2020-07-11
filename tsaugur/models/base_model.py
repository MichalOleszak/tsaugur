import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from tsaugur.metrics import get_metric


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
        self.y = None
        self.name = None
        self.key = None

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

    def fit(self, y, period, start_date=None, x=None, metric="mse", val_size=None, verbose=False):
        """
        Build the model using best-tuned hyperparameter values.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param start_date: pd.datetime object, date of the first observation in training data
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
        :param metric: Str, the metric to score on.
        :return: Float, model performance metric.
        """
        y_pred = self.predict(horizon=len(y_true), x=x)
        metric_fun = get_metric(metric)
        return metric_fun(y_true, y_pred)

    def load_model(self, filepath):
        tmp_dict = joblib.load(filepath)
        self.__dict__.update(tmp_dict)

    def save_model(self, filepath):
        if self.model:
            with open(filepath, "wb"):
                joblib.dump(self.__dict__, filepath)
        else:
            raise Exception("No fitted model to save")

    def plot_predict(self, horizon, x=None):
        """
        Plot observed data for training period and predictions for testing period.
        :param horizon: Int, the number of observations in the future to predict
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        """
        if isinstance(self.y, pd.Series):
            self.y = self.y.values
        y_pred = self.predict(horizon=horizon, x=x)
        values = np.hstack([self.y, y_pred])
        df = pd.DataFrame({"Value": values, "Type": ["Observed"] * len(self.y) + ["Predicted"] * len(y_pred)})
        fig = px.line(df, y="Value", color="Type", title=f"{self.name} predictions")
        fig.update_layout(shapes=[
            dict(
                type='line',
                y0=values.min(), y1=values.max(),
                x0=len(self.y) - 1, x1=len(self.y) - 1
            )
        ])
        fig.show()

    def plot_score(self, y_true, x=None, metric="smape"):
        """
        Plot observed data for training and testing periods and predictions for the testing period.
        :param y_true: pd.Series or 1-D np.array, ground-truth values.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        :param metric: Str, the metric to produce.
        """
        if isinstance(self.y, pd.Series):
            self.y = self.y.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        y_pred = self.predict(horizon=len(y_true), x=x)
        values = np.hstack([self.y, y_pred])
        df = pd.DataFrame({"Value": values, "Type": ["Observed"] * len(self.y) + ["Predicted"] * len(y_pred)})
        df_true = pd.DataFrame({"Value": y_true, "Type": ["Observed"] * len(y_true)},
                               index=df.loc[df["Type"] == "Predicted"].index)
        df = pd.concat([df, df_true], sort=False)
        metric_fun = get_metric(metric)
        score = metric_fun(y_true, y_pred)
        fig = px.line(df, y="Value", color="Type", title=f"{self.name} test set {metric.upper()}: {np.round(score, 3)}")
        fig.update_layout(shapes=[
            dict(
                type='line',
                y0=values.min(), y1=values.max(),
                x0=len(self.y) - 1, x1=len(self.y) - 1
            )
        ])
        fig.show()
