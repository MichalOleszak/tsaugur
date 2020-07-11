from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from pmdarima.metrics import smape
from tsaugur.metrics.custom_metrics import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


METRIC_KEY_MSE = "mse"
METRIC_KEY_MAE = "mae"
METRIC_KEY_RMSE = "rmse"
METRIC_KEY_MAPE = "mape"
METRIC_KEY_SMAPE = "smape"

METRICS = {
    METRIC_KEY_MSE: mean_squared_error,
    METRIC_KEY_MAE: mean_absolute_error,
    METRIC_KEY_RMSE: root_mean_squared_error,
    METRIC_KEY_MAPE: mean_absolute_percentage_error,
    METRIC_KEY_SMAPE: smape,

}


def get_metric(metric_key):
    """
    Return a model performance metric function.
    :param metric_key: Str, a unique identifier of a model performance metric.
    :return: A metric function object.
    """
    return METRICS[metric_key]
