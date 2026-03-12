import streamlit as st

st.title("Dark Store Inventory Optimizer")

item = st.selectbox(
    "Select Item",
    ["Milk", "Bread", "Chips"]
)

current_stock = st.number_input("Current Stock")

forecast = 80

recommended = max(0, forecast - current_stock)

st.write("Forecast Demand:", forecast)

st.write("Recommended Order:", recommended)