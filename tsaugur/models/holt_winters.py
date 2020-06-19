import itertools
import warnings
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tsaugur.utils import data_utils, model_utils
from tsaugur.models import base_model
from tsaugur.metrics import get_metric


class HoltWinters(base_model.BaseModel):

    def tune(self, y, period, x=None, metric="smape", val_size=None, verbose=False):

        if verbose:
            print("Tuning Holt-Winters Exponential Smoothing parameters...")
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        val_size = int(len(y) * .1) if val_size is None else val_size
        y_train, y_val = model_utils.train_val_split(y, val_size=val_size)
        metric_fun = get_metric(metric)

        params_grid = {
            "trend": ["add", "mul"],
            "seasonal": ["add", "mul"],
            "damped": [True, False],
            "use_boxcox": [True, False, "log"],
            "remove_bias": [True, False]
        }
        params_keys, params_values = zip(*params_grid.items())
        params_permutations = [dict(zip(params_keys, v)) for v in itertools.product(*params_values)]

        scores = []
        for permutation in params_permutations:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(y_train, seasonal_periods=self.period, trend=permutation["trend"],
                                                 seasonal=permutation["seasonal"], damped=permutation["damped"])
                    model = model.fit(use_boxcox=permutation["use_boxcox"], remove_bias=permutation["remove_bias"])
                    y_pred = model.forecast(len(y_val))
                    score = metric_fun(y_val, y_pred)
                    scores.append(score)
            except:
                scores.append(np.inf)

        best_params = params_permutations[np.nanargmin(scores)]
        self.params.update(best_params)
        self.params["tuned"] = True

    def fit(self, y, x=None):
        if not self.params["tuned"]:
            raise Exception("Tune the parameters first before fitting the model by calling `.tune()` "
                            "on the model object.")
        model = ExponentialSmoothing(y, seasonal_periods=self.period, trend=self.params["trend"],
                                     seasonal=self.params["seasonal"], damped=self.params["damped"])
        self.model = model.fit(use_boxcox=self.params["use_boxcox"], remove_bias=self.params["remove_bias"])

    def predict(self, horizon, x=None):
        return self.model.forecast(horizon)
