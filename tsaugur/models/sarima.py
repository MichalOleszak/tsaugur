from pmdarima import arima, auto_arima

from tsaugur.utils import data_utils
from tsaugur.models import base_model


class Sarima(base_model.BaseModel):

    def tune(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        if verbose:
            print("Tuning SARIMA parameters...")
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        val_size = int(len(y) * .1) if val_size is None else val_size
        model = auto_arima(y, m=self.period, seasonal=True, d=None, D=None, information_criterion='oob', maxiter=100,
                           error_action='ignore', suppress_warnings=True, stepwise=True, max_order=None,
                           out_of_sample_size=val_size, scoring=metric, exogenous=x)
        self.params.update(model.get_params())
        self.params["tuned"] = True

    def fit(self, y, x=None):
        if not self.params["tuned"]:
            raise Exception("Tune the parameters first before fitting the model by calling `.tune()` "
                            "on the model object.")
        model = arima.ARIMA(maxiter=100, order=self.params["order"], seasonal_order=self.params["seasonal_order"],
                            suppress_warnings=True)
        self.model = model.fit(y, exogenous=x)

    def predict(self, horizon, x=None):
        return self.model.predict(n_periods=horizon, exogenous=x)

