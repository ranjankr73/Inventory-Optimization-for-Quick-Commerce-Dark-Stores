import math

def eoq(demand, ordering_cost, holding_cost):

    return math.sqrt((2 * demand * ordering_cost) / holding_cost)


items = {
    "Milk": {"demand": 80, "current_stock": 60},
    "Bread": {"demand": 50, "current_stock": 30},
    "Chips": {"demand": 70, "current_stock": 50}
}

ordering_cost = 100
holding_cost = 2

for item, data in items.items():

    optimal_order = eoq(data["demand"], ordering_cost, holding_cost)

    suggested = max(0, optimal_order - data["current_stock"])

    print(item)

    print("Suggested order:", int(suggested))

    print()