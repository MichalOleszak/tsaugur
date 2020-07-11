import itertools
import warnings
import pydlm
import numpy as np

from tsaugur.utils import data_utils, model_utils, SuppressStdoutStderr
from tsaugur.models import base_model
from tsaugur.metrics import get_metric


class Bdlm(base_model.BaseModel):
    """
    Bayesian Dynamic Linear Model.
    """

    def _tune(self, y, period, x=None, metric="smape", val_size=None, verbose=False):
        """
        Tune hyperparameters of the model.
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
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        val_size = int(len(y) * .1) if val_size is None else val_size
        y_train, y_val = model_utils.train_val_split(y, val_size=val_size)
        if x is not None:
            x_train, x_val = model_utils.train_val_split(x, val_size=val_size)
        metric_fun = get_metric(metric)

        params_grid = {
            "trend": [0, 1, 2, 3],
            "ar": [None],
            # "ar": [None, 1, 2, 3],
        }
        params_keys, params_values = zip(*params_grid.items())
        params_permutations = [dict(zip(params_keys, v)) for v in itertools.product(*params_values)]

        scores = []
        for permutation in params_permutations:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = pydlm.dlm(y_train)
                    model = model + pydlm.trend(degree=permutation["trend"], discount=0.5)
                    model = model + pydlm.seasonality(period=self.period, discount=0.99)
                    if permutation["ar"] is not None:
                        model = model + pydlm.autoReg(degree=permutation["ar"], discount=0.99)
                    if x is not None:
                        for variable_id, x_variable in enumerate(x_train.T):
                            model = model + pydlm.dynamic(features=[[v] for v in x_variable], discount=0.99,
                                                          name=str(variable_id))
                    with SuppressStdoutStderr():
                        model.tune()
                        model.fit()
                    if x is not None:
                        x_val_dict = {}
                        for variable_id, x_variable in enumerate(x_val.T):
                            x_val_dict.update({str(variable_id): [[v] for v in x_variable]})
                    else:
                        x_val_dict = None
                    y_pred = model.predictN(date=model.n - 1, N=len(y_val), featureDict=x_val_dict)[0]

                    score = metric_fun(y_val, y_pred)
                    scores.append(score)
            except:
                scores.append(np.inf)

        best_params = params_permutations[np.nanargmin(scores)]
        self.params.update(best_params)
        self.params["tuned"] = True

    def fit(self, y, period, x=None, metric="smape", val_size=None, verbose=False):
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
        self.y = y
        self.name = "Bayesian Dynamic Linear Model"
        self.key = "bdlm"
        self._tune(y=y, period=period, x=x, metric=metric, val_size=val_size, verbose=verbose)
        self.model = pydlm.dlm(y)
        self.model = self.model + pydlm.trend(degree=self.params["trend"], discount=0.5)
        self.model = self.model + pydlm.seasonality(period=self.period, discount=0.99)
        if self.params["ar"] is not None:
            self.model = self.model + pydlm.autoReg(degree=self.params["ar"], discount=0.99)
        if x is not None:
            for variable_id, x_variable in enumerate(x.T):
                self.model = self.model + pydlm.dynamic(features=[[v] for v in x_variable], discount=0.99,
                                                        name=str(variable_id))
        with SuppressStdoutStderr():
            self.model.tune()
            self.model.fit()

    def predict(self, horizon, x=None):
        """
        Predict future values of the time series using the fitted model.
        :param horizon: Int, the number of observations in the future to predict
        :param x: pd.DataFrame or 2-D np.array, exogeneous predictors in the forecasted period, required if used in fit
        :return: 1-D np.array with predictions
        """
        if x is not None:
            x_val_dict = {}
            for variable_id, x_variable in enumerate(x.T):
                x_val_dict.update({str(variable_id): [[v] for v in x_variable]})
        else:
            x_val_dict = None
        return self.model.predictN(date=self.model.n - 1, N=horizon, featureDict=x_val_dict)[0]

