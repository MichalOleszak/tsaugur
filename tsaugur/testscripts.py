import pandas as pd
import matplotlib.pyplot as plt
from tsaugur.models import create_model
from tsaugur.models import compare_models
from tsaugur.models import load_model
from tsaugur.datasets import load_dataset



# d = pd.read_csv("datasets/kaggle_sales.csv").sales
# y_train = d[(len(d) - 300):(len(d) - 100)]
# y_test = d[(len(d) - 100):]
# period = 7

d = load_dataset("air_passengers")
dates = pd.to_datetime(d.month)
d = d.passengers
y_train = d[:134]
y_test = d[134:]
period = 12

m = create_model("holt_winters")
m.fit(y_train, period=12)

# m.save_model("mod.joblib")
# m = load_model("mod.joblib")

m.plot_predict(55)
m.plot_score(y_test)

comp = compare_models(["holt_winters", "sarima", "tbats"],
                      y_train, y_test, period=period)
comp.tabulate()
comp.plot()
