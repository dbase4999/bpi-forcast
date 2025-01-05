import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
import re

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])

    def extract_number(value, default=0.0):
        if pd.notnull(value):
            match = re.search(r"\d+(\.\d+)?", str(value))
            return float(match.group()) if match else default
        return default

    data["Inch"] = data["Inch"].apply(extract_number)
    data["Thickness"] = data["Thickness"].apply(extract_number)
    data["Weight"] = data["Weight"].apply(lambda x: float(str(x).replace(" tons", "")) if pd.notnull(x) else 0)
    data["Total Order Weight (kg)"] = pd.to_numeric(data["Total Order Weight (kg)"], errors="coerce")
    data["Total Order Bars"] = pd.to_numeric(data["Total Order Bars"], errors="coerce")

    return data

# Train model
@st.cache_data
def train_model(data):
    # Validasi kolom sebelum encoding
    if "Project Name" not in data.columns or "Support Item" not in data.columns:
        st.error("Kolom 'Project Name' atau 'Support Item' tidak ditemukan dalam data.")
        st.stop()

    data_encoded = pd.get_dummies(data, columns=["Project Name", "Support Item"], drop_first=True)
    features = [
        "Inch", "Thickness", "Weight", "Total Order Bars", "Total Order Weight (kg)",
        *[col for col in data_encoded.columns if col.startswith("Project Name_") or col.startswith("Support Item_")]
    ]
    target = "Final Quantity (Total Requests for Support Items in Field Operations)"

    # Validasi kolom hasil encoding
    encoded_columns = data_encoded.columns
    missing_columns = [col for col in features if col not in encoded_columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan dalam data setelah encoding: {missing_columns}")
        st.stop()

    X = data_encoded[features]
    y = data_encoded[target]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model, features, data_encoded

# File path
file_path = r"/workspaces/bpi-forcast/FixUpdated_Dummy_Data_with_BPI-Compatible_Pipe_Variants.csv"
data = load_data(file_path)
model, features, data_encoded = train_model(data)

# Streamlit UI
st.sidebar.image(
    r"/workspaces/bpi-forcast/654db0b264142.webp",
    use_column_width=True
)
st.sidebar.title("PT Bakrie Pipe Industries")
st.sidebar.write("Forecasting System for Technical Warehouse Management")

st.title("Support Item Stock Forecasting System")
st.write("This system predicts the stock of support items required for technical warehouse management.")

# User input for prediction
st.sidebar.header("Input Project Details")
project_name = st.sidebar.selectbox("Project Name", data["Project Name"].unique())
inch = st.sidebar.number_input("Inch (Diameter)", min_value=0.0, step=0.1)
thickness = st.sidebar.number_input("Thickness (mm)", min_value=0.0, step=0.1)
weight = st.sidebar.number_input("Weight (tons)", min_value=0.0, step=0.1)
total_order_bars = st.sidebar.number_input("Total Order Bars", min_value=0, step=1)
total_order_weight = st.sidebar.number_input("Total Order Weight (kg)", min_value=0.0, step=0.1)

# Create input for prediction
input_data = pd.DataFrame({
    "Inch": [inch],
    "Thickness": [thickness],
    "Weight": [weight],
    "Total Order Bars": [total_order_bars],
    "Total Order Weight (kg)": [total_order_weight],
    **{col: [1] if col == f"Project Name_{project_name}" else [0] for col in features if col.startswith("Project Name_")},
    **{col: [1] if col.startswith("Support Item_") else [0] for col in features if col.startswith("Support Item_")}
})

# Ensure all columns are in the input_data
for col in features:
    if col not in input_data:
        input_data[col] = 0

# Validate inputs and handle predictions
if project_name and (inch > 0 or thickness > 0 or weight > 0 or total_order_bars > 0 or total_order_weight > 0):
    # Predict and display result
    st.subheader("Forecast Result")
    all_support_items = [col.replace("Support Item_", "") for col in features if col.startswith("Support Item_")]
    support_item_predictions = {}

    for support_item in all_support_items:
        temp_input = input_data.copy()
        temp_input[f"Support Item_{support_item}"] = 1
        prediction = model.predict(temp_input)[0]
        support_item_predictions[support_item] = max(0, int(prediction))

    # Display predictions in a table
    st.write("Predicted Support Item Stock Requirements:")
    predicted_table = pd.DataFrame(list(support_item_predictions.items()), columns=["Support Item", "Predicted Quantity"])
    st.table(predicted_table)
else:
    st.subheader("Forecast Result")
    st.write("Silakan masukkan data terkait yang ingin dilakukan forecast.")

# Visualization
st.subheader("Historical Data Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
agg_data = data.groupby("Date").sum()["Final Quantity (Total Requests for Support Items in Field Operations)"]
ax.plot(agg_data.index, agg_data.values, label="Historical Data")
ax.set_title("Total Requests Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Total Requests")
ax.legend()
st.pyplot(fig)