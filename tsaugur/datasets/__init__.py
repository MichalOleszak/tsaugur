import os
import pandas as pd


def load_dataset(key):
    return pd.read_csv(os.path.join("tsaugur", "datasets", key + ".csv"))
