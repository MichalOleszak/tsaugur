def train_val_split(y, val_size):
    return y[:val_size], y[val_size:]
