import itertools
import warnings
import numpy as np
import pandas as pd
from fbprophet import Prophet

from tsaugur.utils import data_utils, model_utils, SuppressStdoutStderr
from tsaugur.models import base_model
from tsaugur.metrics import get_metric


class FbProphet(base_model.BaseModel):
    """
    Facebook's Prophet.
    """

    def _tune(self, y, period, start_date, x=None, metric="smape", val_size=None, verbose=False):
        """
        Tune hyperparameters of the model.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :param metric: Str, the metric used for model selection. One of: "mse", "mae", "mape", "smape", "rmse".
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        dates = data_utils.create_dates(start_date, period, length=len(y))
        val_size = int(len(y) * .1) if val_size is None else val_size
        y_train, y_val = model_utils.train_val_split(y, val_size=val_size)
        dates_train, dates_val = model_utils.train_val_split(dates, val_size=val_size)
        input_df = pd.DataFrame({"ds": dates_train, "y": y_train})
        future_df = pd.DataFrame({"ds": dates_val})
        if x is not None:
            x_train, x_val = model_utils.train_val_split(x, val_size=val_size)
            for variable_id, x_variable in enumerate(x_train.T):
                input_df[variable_id] = x_variable
            for variable_id, x_variable in enumerate(x_val.T):
                future_df[variable_id] = x_variable
        metric_fun = get_metric(metric)

        params_grid = {
            "seasonality": ["additive", "multiplicative"],
            "growth": ["linear", "logistic"],
            "changepoint_prior_scale": [0.005, 0.05, 0.5],
        }
        params_keys, params_values = zip(*params_grid.items())
        params_permutations = [dict(zip(params_keys, v)) for v in itertools.product(*params_values)]

        scores = []
        for permutation in params_permutations:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = Prophet(
                        seasonality_mode=permutation["seasonality"],
                        growth=permutation["growth"],
                        changepoint_prior_scale=permutation["changepoint_prior_scale"],
                    )
                    if x is not None:
                        variable_ids = list(sorted(set(input_df.columns).difference(set(["ds", "y"]))))
                        for variable_id in variable_ids:
                            model.add_regressor(variable_id)
                    with SuppressStdoutStderr():
                        model.fit(input_df)
                    y_pred = model.predict(future_df)["yhat"].values
                    score = metric_fun(y_val, y_pred)
                    scores.append(score)
            except:
                scores.append(np.inf)

        best_params = params_permutations[np.nanargmin(scores)]
        self.params.update(best_params)
        self.params["tuned"] = True

    def fit(self, y, period, start_date, x=None, metric="smape", val_size=None, verbose=False):
        """
        Build the model with using best-tuned hyperparameter values.
        :param y: pd.Series or 1-D np.array, time series to predict.
        :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
        for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
        data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
        "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :param start_date: pd.datetime object, date of the first observation in training data
        :param metric: Str, the metric used for model selection. One of: "mse", "mae", "mape", "smape", "rmse".
        :param val_size: Int, the number of most recent observations to use as validation set for tuning.
        :param verbose: Boolean, True for printing additional info while tuning.
        :return: None
        """
        self.y = y
        self.name = "Prophet"
        self.key = "prophet"
        self._tune(y=y, period=period, start_date=start_date, metric=metric, val_size=val_size, verbose=verbose)
        dates = data_utils.create_dates(start_date, period=self.period, length=len(y))
        input_df = pd.DataFrame({"ds": dates, "y": y})
        model = Prophet(
            seasonality_mode=self.params["seasonality"],
            growth=self.params["growth"],
            changepoint_prior_scale=self.params["changepoint_prior_scale"],
        )
        if x is not None:
            for variable_id, x_variable in enumerate(x.T):
                input_df[variable_id] = x_variable
                model.add_regressor(variable_id)
        with SuppressStdoutStderr():
            model = model.fit(input_df)
        self.model = model
        self.last_fitting_date = dates[-1]
        self.time_delta = dates[1] - dates[0]

    def predict(self, horizon, x=None):
        """
        Predict future values of the time series using the fitted model.
        :param horizon: Int, the number of observations in the future to predict
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors, optional
        :return: 1-D np.array with predictions
        """
        dates = data_utils.create_dates(self.last_fitting_date + self.time_delta, period=self.period, length=horizon)
        future_df = pd.DataFrame({"ds": dates})
        if x is not None:
            for variable_id, x_variable in enumerate(x.T):
                future_df[variable_id] = x_variable
        return self.model.predict(future_df)["yhat"].values
