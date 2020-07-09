import pandas as pd
import matplotlib.pyplot as plt
from tsaugur.models import compare_models

d = pd.read_csv("datasets/air_passengers.csv")
dates = pd.to_datetime(d.month)
d = d.passengers
y_train = d[:134]
y_test = d[134:]
period = 12

# d = pd.read_csv("datasets/kaggle_sales.csv").sales
# y_train = d[(len(d) - 300):(len(d) - 100)]
# y_test = d[(len(d) - 100):]
# period = 7

comp = compare_models(["holt_winters", "sarima", "tbats", "prophet"],
                      y_train, y_test, period=period, start_date=dates[0])
comp.tabulate()
comp.plot()



# preds = {}
# models = [
#    # "sarima",
#    # "fourier_sarima",
#    #  "holt_winters",
#    #  "tbats",
#    #  "bdlm",
#     "prophet"
# ]
# for m in models:
#     mod = create_model(m)
#     if m == "prophet":
#         mod.fit(y_train, period=period, val_size=10, start_date=dates[0])
#     else:
#         mod.fit(y_train, period=period, val_size=10)
#     preds.update({m: mod.predict(len(y_test))})
#     score = mod.score(y_test)
#     print(f"{m} score: {score}\n")
#
# plt.figure(figsize=(15, 10))
# plt.plot(d, c="black")
# if "sarima" in models:
#     sarima = pd.Series(preds["sarima"])
#     sarima.index = y_test.index
#     plt.plot(sarima, c="blue")
# if "fourier_sarima" in models:
#     fourier_sarima = pd.Series(preds["fourier_sarima"])
#     fourier_sarima.index = y_test.index
#     plt.plot(fourier_sarima, c="red")
# if "holt_winters" in models:
#     hw = pd.Series(preds["holt_winters"])
#     hw.index = y_test.index
#     plt.plot(hw, c="orange")
# if "tbats" in models:
#     tb = pd.Series(preds["tbats"])
#     tb.index = y_test.index
#     plt.plot(tb, c="purple")
# if "bdlm" in models:
#     bdlm = pd.Series(preds["bdlm"])
#     bdlm.index = y_test.index
#     plt.plot(bdlm, c="green")
# if "prophet" in models:
#     prophet = pd.Series(preds["prophet"])
#     prophet.index = y_test.index
#     plt.plot(prophet, c="green")
#
# plt.show()
