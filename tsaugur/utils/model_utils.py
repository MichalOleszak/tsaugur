def _train_val_split(d, val_size):
    """
    Split a time series into train and validation sets by setting most recent data aside for validation.
    :param y: pd.Series or 1-D np.array or 2-D np.array, time series to predict or exogeneous variables.
    :param val_size: Int, the number of most recent observations to use as validation set for tuning.
    :return: (pd.Series or 1-D np.array, pd.Series or 1-D np.array), train and validation data respectively.
    """
    return d[:(len(d) - val_size)], d[(len(d) - val_size):]
