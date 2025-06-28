import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("best_random_forest_model.pkl")

st.set_page_config(page_title="DC Delivery Time Improvement Predictor", layout="wide")
st.title("📦 New DC Delivery Time Improvement Predictor")

st.markdown("This tool predicts whether delivery time will improve if a customer is served from a proposed new Distribution Center (DC).")

# --- User Input Section ---
st.sidebar.header("🔧 Input Features")

# Numeric Inputs
product_category = st.sidebar.selectbox("Product Category (encoded)", [0, 1, 2])
num_of_item = st.sidebar.slider("Number of Items", 1, 10, 1)
country = st.sidebar.selectbox("Country (encoded)", [0, 1, 2])
user_latitude = st.sidebar.number_input("User Latitude", value=1.0)
distribution_center_latitude = st.sidebar.number_input("Old DC Latitude", value=1.0)
distribution_center_longitude = st.sidebar.number_input("Old DC Longitude", value=1.0)
new_dc_latitude = st.sidebar.number_input("New DC Latitude", value=1.0)
new_dc_longitude = st.sidebar.number_input("New DC Longitude", value=1.0)
delivery_time_days = st.sidebar.number_input("Old Delivery Time (days)", value=3.0)
processing_delay_time_days = st.sidebar.number_input("Processing Delay (days)", value=1.0)
order_dayofweek = st.sidebar.selectbox("Order Day of Week", list(range(7)))
order_month = st.sidebar.selectbox("Order Month", list(range(1, 13)))
order_volume = st.sidebar.slider("Order Volume", 1, 100, 10)
cluster = st.sidebar.selectbox("Cluster ID", list(range(10)))
delivery_time_hour = st.sidebar.number_input("Delivery Time (hours)", value=24.0)
delivery_speed_kmph = st.sidebar.number_input("Delivery Speed (km/h)", value=60.0)

# Boolean Inputs
is_weekend_order = st.sidebar.checkbox("Is Weekend Order", value=False)
status_Returned = st.sidebar.checkbox("Was Returned", value=False)
product_department_Women = st.sidebar.checkbox("Women's Product", value=False)
gender_M = st.sidebar.checkbox("Customer is Male", value=False)

# Traffic Source
traffic_source_Email = st.sidebar.checkbox("Source: Email", value=False)
traffic_source_Facebook = st.sidebar.checkbox("Source: Facebook", value=False)
traffic_source_Organic = st.sidebar.checkbox("Source: Organic", value=True)
traffic_source_Search = st.sidebar.checkbox("Source: Search", value=False)

# DC Names (One-hot encoded)
st.sidebar.markdown("### Old DC Location")
dc_cols = [
    "Chicago IL", "Houston TX", "Los Angeles CA", "Memphis TN", "Mobile AL",
    "New Orleans LA", "Philadelphia PA", "Port Authority of New York/New Jersey NY/NJ", "Savannah GA"
]
dc_selected = st.sidebar.selectbox("Old DC Name", dc_cols)
dc_encoded = [dc_selected == name for name in dc_cols]

# Age group
user_age_group_senior = st.sidebar.checkbox("User is Senior", value=False)
user_age_group_teen = st.sidebar.checkbox("User is Teen", value=False)

# Log values (replace with transformation if needed)
log_cost = np.log(10)
log_product_retail_price = np.log(50)
log_sale_price = np.log(40)
avg_delivery_time_days = st.sidebar.number_input("Average Delivery Time (days)", value=3.5)

# --- Prepare Input for Model ---
input_data = pd.DataFrame([[
    product_category,
    num_of_item,
    country,
    user_latitude,
    distribution_center_latitude,
    distribution_center_longitude,
    delivery_time_days,
    processing_delay_time_days,
    order_dayofweek,
    order_month,
    is_weekend_order,
    user_latitude,  # state_latitude = assume same as user_latitude
    log_cost,
    log_product_retail_price,
    log_sale_price,
    order_volume,
    avg_delivery_time_days,
    cluster,
    new_dc_latitude,
    new_dc_longitude,
    delivery_time_hour,
    delivery_speed_kmph,
    status_Returned,
    product_department_Women,
    gender_M,
    traffic_source_Email,
    traffic_source_Facebook,
    traffic_source_Organic,
    traffic_source_Search,
    *dc_encoded,
    user_age_group_senior,
    user_age_group_teen
]], columns=[
    'product_category', 'num_of_item', 'country', 'user_latitude',
    'distribution_center_latitude', 'distribution_center_longitude',
    'delivery_time_days', 'processing_delay_time_days', 'order_dayofweek',
    'order_month', 'is_weekend_order', 'state_latitude', 'log_cost',
    'log_product_retail_price', 'log_sale_price', 'order_volume',
    'avg_delivery_time_days', 'cluster', 'new_dc_latitude', 'new_dc_longitude',
    'delivery_time_hour', 'delivery_speed_kmph', 'status_Returned',
    'product_department_Women', 'gender_M', 'traffic_source_Email',
    'traffic_source_Facebook', 'traffic_source_Organic', 'traffic_source_Search',
    'distribution_center_name_Chicago IL', 'distribution_center_name_Houston TX',
    'distribution_center_name_Los Angeles CA', 'distribution_center_name_Memphis TN',
    'distribution_center_name_Mobile AL', 'distribution_center_name_New Orleans LA',
    'distribution_center_name_Philadelphia PA', 'distribution_center_name_Port Authority of New York/New Jersey NY/NJ',
    'distribution_center_name_Savannah GA', 'user_age_group_senior', 'user_age_group_teen'
])

# --- Make Prediction ---
if st.button("🔍 Predict Delivery Time Improvement"):
    prediction = model.predict(input_data)[0]
    st.success("✅ Delivery time will improve!" if prediction == 1 else "❌ No improvement in delivery time.")
