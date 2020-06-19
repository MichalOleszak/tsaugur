import numpy as np
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error of prediction.
    :param y_true: pd.Series or 1-D np.array, ground-truth values.
    :param y_pred: pd.Series or 1-D np.array, predicted values.
    :return: Float, the prediction error.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate root mean squared error of prediction.
    :param y_true: pd.Series or 1-D np.array, ground-truth values.
    :param y_pred: pd.Series or 1-D np.array, predicted values.
    :return: Float, the prediction error.
    """
    return mean_squared_error(y_true, y_pred, squared=False)
