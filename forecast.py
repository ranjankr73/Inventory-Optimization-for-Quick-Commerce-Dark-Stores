import pandas as pd

df = pd.read_csv("demand_data.csv")

def forecast(item):

    item_data = df[df["item"] == item]

    last_7 = item_data.tail(7)

    prediction = last_7["demand"].mean()

    return round(prediction)

items = df["item"].unique()

for item in items:

    pred = forecast(item)

    print(f"{item} forecast demand tomorrow: {pred}")