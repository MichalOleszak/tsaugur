from tsaugur.metrics import get_metric


class BaseModel:

    def __init__(self):
        self.model = None
        self.period = None
        self.params = {"tuned": False}

    def tune(self, y, period, metric="mse", test_size=None, verbose=False):
        raise NotImplementedError

    def fit(self, y):
        raise NotImplementedError

    def predict(self, horizon):
        raise NotImplementedError

    def score(self, y_true, metric="smape"):
        y_pred = self.predict(horizon=len(y_true))
        metric_fun = get_metric(metric)
        return metric_fun(y_true, y_pred)
