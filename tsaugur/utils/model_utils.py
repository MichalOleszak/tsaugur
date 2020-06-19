def _train_val_split(y, val_size):
    """
    Split a time series into train and validation sets by setting most recent data aside for validation.
    :param y: pd.Series or 1-D np.array, time series to predict.
    :param val_size: Int, the number of most recent observations to use as validation set for tuning.
    :return: (pd.Series or 1-D np.array, pd.Series or 1-D np.array), train and validation data respectively.
    """
    return y[:val_size], y[val_size:]
