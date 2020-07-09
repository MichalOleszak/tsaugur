from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from tsaugur.models.holt_winters import HoltWinters
from tsaugur.models.sarima import Sarima
from tsaugur.models.fourier_sarima import FourierSarima
from tsaugur.models.tbats import Tbats
from tsaugur.models.bdlm import Bdlm
from tsaugur.models.prophet import FbProphet


MODEL_KEY_HOLT_WINTERS = "holt_winters"
MODEL_KEY_SARIMA = "sarima"
MODEL_KEY_FOURIER_SARIMA = "fourier_sarima"
MODEL_KEY_TBATS = "tbats"
MODEL_KEY_BDLM = "bdlm"
MODEL_KEY_PROPHET = "prophet"

MODELS = {
    MODEL_KEY_SARIMA: Sarima(),
    MODEL_KEY_FOURIER_SARIMA: FourierSarima(),
    MODEL_KEY_HOLT_WINTERS: HoltWinters(),
    MODEL_KEY_TBATS: Tbats(),
    MODEL_KEY_BDLM: Bdlm(),
    MODEL_KEY_PROPHET: FbProphet(),
}


class ModelComparison:
    """
    A comparison object with methods to print and plot competing models' performance
    """

    def __init__(self, model_tags, y_train, y_test, period, x_train, x_test, val_size, start_date, metrics):
        self.scores = defaultdict(dict)
        self.models = defaultdict()
        self.metrics = metrics
        for model_tag in model_tags:
            self.scores.update({model_tag: {}})
            model = create_model(model_tag)
            if model_tag == "prophet":
                model.fit(y_train, x=x_train, period=period, val_size=val_size, start_date=start_date)
            else:
                model.fit(y_train, x=x_train, period=period, val_size=val_size)
            self.models.update({model_tag: model})
            for metric in metrics:
                self.scores[model_tag].update({metric: model.score(y_true=y_test, x=x_test, metric=metric)})

    def tabulate(self):
        return pd.DataFrame(self.scores).transpose().sort_values(self.metrics[0])

    def plot(self):
        tab = self.tabulate()
        tab = tab.apply(lambda x: x / max(x), axis=0)
        fig = go.Figure()
        for index, row in tab.iterrows():
            fig.add_trace(go.Scatter(x=tab.columns.str.upper(), y=row, mode="lines+markers",
                                     name=index, marker=dict(size=20)))
        fig.update_layout(yaxis=dict(range=[-.03, 1.03]),
                          title="Relative errors across metrics (lower is better)",
                          legend_title_text="Model",
                          template="none")
        fig.show()


def create_model(model_key):
    """
    Instantiate the model class.
    :param model_key: Str, a unique identifier of a tsaugur model class for a specific model.
    :return: The tsaugur model class for the requested model.
    """
    return MODELS[model_key]


def print_available_models():
    """
    Print the list of models supported by tsaugur with additional information.
    :return: None
    """
    print(
        "\n"
        "Model             Description                                           Exogeneous variables\n"
        "--------------------------------------------------------------------------------------------\n"
        "holt_winters      Holt-Winters Exponential Smoothing                    no\n"
        "sarima            Seasonal Auto-Regression Integrated Moving Average    yes\n"
        "fourier_sarima    Auto-Regression Integrated Moving Average             yes\n"
        "                  with seasonality captured by Fourier terms\n"
        "tbats             Trigonometric seasonality, Box-Cox transformation,    no\n"
        "                  ARMA errors, Trend and Seasonal components\n"
        "bdlm              Bayesian Dynamic Linear Model                         yes\n"
        "prophet           Facebook's Prophet model                              yes\n"
    )


def compare_models(model_tags, y_train, y_test, period, x_train=None, x_test=None, val_size=None,
                   start_date=None, metrics=("smape", "mae", "mse", "rmse")):
    return ModelComparison(model_tags=model_tags, y_train=y_train, y_test=y_test, period=period, x_train=x_train,
                           x_test=x_test, val_size=val_size, start_date=start_date, metrics=metrics)
