from tsaugur.models.holt_winters import HoltWinters
from tsaugur.models.sarima import Sarima
from tsaugur.models.fourier_sarima import FourierSarima
from tsaugur.models.tbats import Tbats
from tsaugur.models.bdlm import Bdlm


MODEL_KEY_HOLT_WINTERS = "holt_winters"
MODEL_KEY_SARIMA = "sarima"
MODEL_KEY_FOURIER_SARIMA = "fourier_sarima"
MODEL_KEY_TBATS = "tbats"
MODEL_KEY_BDLM = "bdlm"

MODELS = {
    MODEL_KEY_SARIMA: Sarima(),
    MODEL_KEY_FOURIER_SARIMA: FourierSarima(),
    MODEL_KEY_HOLT_WINTERS: HoltWinters(),
    MODEL_KEY_TBATS: Tbats(),
    MODEL_KEY_BDLM: Bdlm(),
}


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
        "prophet           -                                                     -\n"
    )
