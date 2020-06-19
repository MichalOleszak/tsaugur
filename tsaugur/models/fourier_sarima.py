from pmdarima import arima, auto_arima, pipeline
from pmdarima.preprocessing import FourierFeaturizer

from tsaugur.utils import data_utils
from tsaugur.models import base_model


class FourierSarima(base_model.BaseModel):

    def tune(self, y, period, x=None, metric="mse", val_size=None, verbose=False):
        if verbose:
            print("Tuning FourierSARIMA parameters...")
        self.period = data_utils.period_to_int(period) if type(period) == str else period
        val_size = int(len(y) * .1) if val_size is None else val_size
        pipe = pipeline.Pipeline([
            ("fourier", FourierFeaturizer(self.period, self.period / 2)),
            ("arima", auto_arima(y, m=self.period, seasonal=False, d=None, information_criterion='oob', maxiter=100,
                                 error_action='ignore', suppress_warnings=True, stepwise=True, max_order=None,
                                 out_of_sample_size=val_size, scoring=metric, exogenous=x))
        ])
        self.params.update(pipe.steps[1][1].get_params())
        self.params["tuned"] = True

    def fit(self, y, x=None):
        if not self.params["tuned"]:
            raise Exception("Tune the parameters first before fitting the model by calling `.tune()` "
                            "on the model object.")
        pipe = pipeline.Pipeline([
            ("fourier", FourierFeaturizer(self.period, self.period / 2)),
            ("arima", arima.ARIMA(maxiter=100, order=self.params["order"], seasonal_order=self.params["seasonal_order"],
                                  suppress_warnings=True))
        ])
        self.model = pipe.fit(y, exogenous=x)

    def predict(self, horizon, x=None):
        return self.model.predict(n_periods=horizon, exogenous=x)

