import pandas as pd
import matplotlib.pyplot as plt
from tsaugur.models import create_model

d = pd.read_csv("datasets/air_passengers.csv").passengers
y_train = d[:134]
y_test = d[134:]

preds = {}
for m in ["sarima", "fourier_sarima", "holt_winters"]:
    mod = create_model(m)
    mod.tune(y_train, period=12, verbose=True)
    mod.fit(y_train)
    preds.update({m: mod.predict(len(y_test))})
    score = mod.score(y_test)
    print(f"{m} score: {score}")


sarima = pd.Series(preds["sarima"])
sarima.index = y_test.index

fourier_sarima = pd.Series(preds["fourier_sarima"])
fourier_sarima.index = y_test.index

hw = pd.Series(preds["holt_winters"])
hw.index = y_test.index

plt.figure(figsize=(15, 10))
plt.plot(d, c="black")
plt.plot(sarima, c="blue")
plt.plot(fourier_sarima, c="red")
plt.plot(hw, c="orange")
plt.show()


