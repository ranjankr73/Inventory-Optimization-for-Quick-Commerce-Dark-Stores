import pandas as pd
import numpy as np

np.random.seed(42)

items = ["Milk", "Bread", "Chips", "Eggs", "Soft Drinks"]

days = 60

data = []

for item in items:
    base = np.random.randint(40, 100)

    for day in range(days):
        demand = int(np.random.normal(base, 10))
        demand = max(0, demand)

        data.append({
            "day": day+1,
            "item": item,
            "demand": demand
        })

df = pd.DataFrame(data)

df.to_csv("demand_data.csv", index=False)

print(df.head())