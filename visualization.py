import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("demand_data.csv")

milk = df[df["item"] == "Milk"]

plt.plot(milk["day"], milk["demand"])

plt.title("Milk Demand Over Time")

plt.xlabel("Day")

plt.ylabel("Demand")

plt.show()