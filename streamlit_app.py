# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np
# import joblib
# from preprocessing_utils import HighCardinalityDropper, ColumnDropper

# import requests

# # Load Together AI key from secrets
# TOGETHER_API_KEY = st.secrets["togetherai"]["api_key"]

# def generate_llama_response(prompt):
#     url = "https://api.together.xyz/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {TOGETHER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "meta-llama/Llama-3-8b-chat-hf",
#         "messages": [
#             {"role": "system", "content": "You are a logistics expert assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 512
#     }
#     response = requests.post(url, json=payload, headers=headers)
#     result = response.json()

#     return result["choices"][0]["message"]["content"]

# # --- Page Setup ---
# st.set_page_config(page_title="DC Placement App", layout="wide")
# st.title("Distribution Center Suggestion Dashboard")

# # --- Load Data ---
# @st.cache_data
# def load_data():
#     # Replace with your actual GitHub raw URL
#     url = "https://raw.githubusercontent.com/Jojooyee/NewDCPlacement-app/main/test_df.csv"
#     df = pd.read_csv(url)
#     return df
    
# df = load_data()

# # --- Load Pipeline ---
# preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")

# # --- Load trained prediction model ---
# model = joblib.load("delivery_improvement_model.pkl")

# # --- Create Tabs Instead of Sidebar Navigation ---
# tab1, tab2, tab3 = st.tabs(["Cluster-based DC Suggestion", "Manual Proposed DC Location","Comparison Summary"])

# # --- TAB 1: New DC Suggestion ---
# with tab1:
#     st.header("New Distribution Center Suggestion")
#     st.markdown("Use one of the options below to explore potential locations for new distribution centers.")

#     result_option = st.selectbox("Select Suggestion Result:", [
#         "New DC Location",
#         "Clustering Report"
#     ])

#     state_level_df = df.drop_duplicates(subset=["state"])[
#         ["state", "order_volume", "avg_delivery_time_days", "state_latitude", "state_longitude", "cluster", "new_dc_latitude", "new_dc_longitude"]
#     ].sort_values("state")

#     state_level_df["cluster"] = state_level_df["cluster"].astype(str)

#     if result_option == "New DC Location":
#         # Section 1: List of new dc location
#         st.markdown("### Suggested New DC Coordinates")
#         st.markdown("The list below shows the location of new dc by cluster.")

#         unique_dc_locations = state_level_df.groupby("cluster")[["new_dc_latitude", "new_dc_longitude"]].first().reset_index()

#         for _, row in unique_dc_locations.iterrows():
#             st.markdown(f"**Cluster {row['cluster']}**: ({row['new_dc_latitude']:.4f}, {row['new_dc_longitude']:.4f})")

#         # Section 2: New dc location in map
#         st.markdown("### New Distribution Center Location")
#         st.markdown("The map below shows the location of new dc.")

#         fig_map = px.scatter_mapbox(
#             state_level_df,
#             lat="new_dc_latitude",
#             lon="new_dc_longitude",
#             color="cluster",
#             hover_name="state",
#             hover_data={"cluster": True},
#             zoom=2,
#             height=500
#          )
#         fig_map.update_layout(mapbox_style="open-street-map")
#         fig_map.update_traces(marker=dict(size=10, opacity=1.0))
#         st.plotly_chart(fig_map, use_container_width=True)

#         # Transform input features
#         processed_df = preprocessing_pipeline.transform(df)
        
#         # Predict
#         predictions = model.predict(processed_df)
#         prediction_probs = model.predict_proba(processed_df)[:, 1]
        
#         # Append predictions to the ORIGINAL dataframe
#         df["delivery_time_improvement_pred"] = predictions
#         df["improvement_probability"] = prediction_probs

#         st.markdown("### Prediction Summary")
#         improve_count = (df["delivery_time_improvement_pred"] == 1).sum()
#         no_improve_count = (df["delivery_time_improvement_pred"] == 0).sum()
            
#         st.write(f"**Users with Improved Delivery**: {improve_count}")
#         st.write(f"**Users with No Improvement**: {no_improve_count}")
            
#         # Optional: Pie chart
#         fig_pie = px.pie(
#             names=["Improved", "No Improvement"],
#             values=[improve_count, no_improve_count],
#             title="Delivery Improvement Prediction Results"
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#         # After prediction
#         st.markdown("### AI-Powered Recommendation")
        
#         prompt = f"""
#         Based on delivery improvement prediction:
#         - Improved: {improve_count}
#         - Not Improved: {no_improve_count}
        
#         Explain why the suggested new DC locations are effective and what logistics reasoning supports it.
#         """
        
#         with st.spinner("Generating AI recommendation..."):
#             try:
#                 llama_output = generate_llama_response(prompt)
#                 st.success("AI-generated Recommendation:")
#                 st.markdown(llama_output)
#             except Exception as e:
#                 st.error(f"Error generating recommendation: {e}")
                
#     elif result_option == "Clustering Report":
#         # Section 1: Order volume & Avg delivery time
#         st.markdown("### State-Level Summary")
#         st.markdown("Below are the visualizations of order volume and average delivery time by state from the clustered dataset.")

#         col1, col2 = st.columns(2)
#         with col1:
#             top_n = st.selectbox("Select number of states to display", [10, 20, 30, 40, 50], index=1)
#         with col2:
#             sort_order = st.selectbox("Sort by:", ["Lowest", "Highest"])

#         sort_ascending = sort_order == "Lowest"

#         top_order_volume_df = state_level_df.sort_values("order_volume", ascending=sort_ascending).head(top_n)
#         fig_order = px.bar(
#             top_order_volume_df,
#             x="state",
#             y="order_volume",
#             title="Order Volume",
#             labels={"order_volume": "Order Volume", "state": "State"},
#             color="order_volume",
#             color_continuous_scale="Blues"
#         )
#         st.plotly_chart(fig_order, use_container_width=True)

#         top_delivery_df = state_level_df.sort_values("avg_delivery_time_days", ascending=sort_ascending).head(top_n)
#         fig_delivery = px.bar(
#             top_delivery_df,
#             x="state",
#             y="avg_delivery_time_days",
#             title="Avg Delivery Time (Days)",
#             labels={"avg_delivery_time_days": "Avg Delivery Time (Days)", "state": "State"},
#             color="avg_delivery_time_days",
#             color_continuous_scale="Oranges"
#         )
#         st.plotly_chart(fig_delivery, use_container_width=True)

#         # Section 2: Cluster map
#         st.markdown("### Cluster Map")
#         st.markdown("The map below shows the location of each state, colored by assigned cluster.")

#         fig_map = px.scatter_mapbox(
#             state_level_df,
#             lat="state_latitude",
#             lon="state_longitude",
#             color="cluster",
#             hover_name="state",
#             hover_data={"cluster": True},
#             zoom=2,
#             height=500
#         )
#         fig_map.update_layout(mapbox_style="open-street-map")
#         fig_map.update_traces(marker=dict(size=10, opacity=1.0))
#         st.plotly_chart(fig_map, use_container_width=True)

#         # Section 3: Cluster demand ranking
#         st.markdown("### Demand Ranking")
#         st.markdown("Clusters are ranked based on their total composite weight (indicating demand concentration).")
#         df["cluster"] = df["cluster"].astype(str)

#         cluster_ranking = (
#             df.groupby("cluster")["composite_weight"]
#             .sum()
#             .sort_values(ascending=False)
#             .reset_index()
#         )

#         for i, row in cluster_ranking.iterrows():
#             cluster_id = row["cluster"]
#             st.markdown(f"**{i+1}. Cluster {cluster_id}**")

#             cluster_data = state_level_df[state_level_df["cluster"] == cluster_id]
#             total_states = cluster_data["state"].nunique()
#             list_states = sorted(cluster_data["state"].unique().tolist())
#             total_order_volume = cluster_data["order_volume"].mean()
#             avg_delivery_time = cluster_data["avg_delivery_time_days"].mean()

#             with st.expander("Expand for detail"):
#                 st.markdown(f"**Total number of states**: `{total_states}`")
#                 st.markdown(f"**Average order volume**: `{total_order_volume:,}`")
#                 st.markdown(f"**Average delivery time (days)**: `{avg_delivery_time:.2f}`")

# # --- TAB 2: Manual Proposed DC Location---
# with tab2:
#     st.subheader("Manual DC Simulation")
#     st.markdown("Simulate multiple proposed DC locations by entering coordinates below.")

#     # Step 1: Let user input how many DCs they want to enter
#     num_points = st.number_input("Enter number of proposed DC locations:", min_value=1, max_value=10, value=1, step=1)

#     # Step 2: Show input fields dynamically based on that number
#     new_dc_locations = []
#     for i in range(int(num_points)):
#         st.markdown(f"#### DC Location {i + 1}")
#         col1, col2 = st.columns(2)
#         with col1:
#             lat = st.number_input(f"Latitude {i + 1}", key=f"lat_{i}", value=0.0, format="%.6f")
#         with col2:
#             lon = st.number_input(f"Longitude {i + 1}", key=f"lon_{i}", value=0.0, format="%.6f")
#         new_dc_locations.append((lat, lon))

#     # Step 3: Simulate Button — FIXED with a key
#     if st.button("Simulate", key="simulate_dc_locations"):
    
#         # Step 2.5: Plot user-input coordinates on a map
#         if new_dc_locations:
#             st.markdown("### 🗺️ Preview of Proposed DC Locations")
#             user_dc_df = pd.DataFrame(new_dc_locations, columns=["lat", "lon"])
#             user_dc_df["dc_id"] = [f"DC {i+1}" for i in range(len(new_dc_locations))]
    
#             fig_user_dc = px.scatter_mapbox(
#                 user_dc_df,
#                 lat="lat",
#                 lon="lon",
#                 hover_name="dc_id",
#                 zoom=2,
#                 height=400
#             )
#             fig_user_dc.update_layout(mapbox_style="open-street-map")
#             fig_user_dc.update_traces(marker=dict(size=10, color="red"))
#             st.plotly_chart(fig_user_dc, use_container_width=True)

#         def haversine(lat1, lon1, lat2, lon2):
#             R = 6371  # km
#             lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

#             dlat = lat2 - lat1
#             dlon = lon2 - lon1

#             a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
#             c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

#             return R * c

#         updated_rows = []

#         for idx, row in df.iterrows():
#             user_lat = row["user_latitude"]
#             user_lon = row["user_longitude"]

#             # Calculate distance from user to every manually entered DC
#             distances = []
#             for dc_lat, dc_lon in new_dc_locations:
#                 distance = haversine(user_lat, user_lon, dc_lat, dc_lon)
#                 distances.append((distance, dc_lat, dc_lon))

#             # Find the nearest DC
#             min_distance, nearest_lat, nearest_lon = min(distances, key=lambda x: x[0])

#             # Create updated row
#             updated_row = row.copy()
#             updated_row["new_dc_latitude"] = nearest_lat
#             updated_row["new_dc_longitude"] = nearest_lon
#             updated_row["distance_new_dc_to_user_km"] = min_distance

#             updated_rows.append(updated_row)

#         # Create a new DataFrame with the updated values
#         simulated_df = pd.DataFrame(updated_rows)
        
#         # Convert delivery time from days to hours
#         simulated_df['delivery_time_hour'] = simulated_df["delivery_time_days"] * 24
#         # Calculate delivery speed (km/hour)
#         simulated_df["delivery_speed_kmph"] = simulated_df["distance_dc_to_user_km"] / simulated_df['delivery_time_hour']
#         # Estimate new delivery time in hours using the same speed
#         simulated_df["estimated_new_delivery_time"] = simulated_df["distance_new_dc_to_user_km"] / simulated_df["delivery_speed_kmph"]
#         # Calculate improvement in hours
#         simulated_df["delivery_time_improvement"] = simulated_df["delivery_time_hour"] - simulated_df["estimated_new_delivery_time"]

#         simulated_processed = preprocessing_pipeline.transform(simulated_df)

#         # --- Make prediction (binary classification: 1 = improvement, 0 = no improvement) ---
#         predictions = model.predict(simulated_processed)
#         prediction_probs = model.predict_proba(simulated_processed)[:, 1]  # Probabilities for class 1

#         # --- Append predictions to DataFrame ---
#         simulated_df["delivery_time_improvement_pred"] = predictions
#         simulated_df["improvement_probability"] = prediction_probs

#         st.markdown("### Prediction Summary")
#         improve_count = (simulated_df["delivery_time_improvement_pred"] == 1).sum()
#         no_improve_count = (simulated_df["delivery_time_improvement_pred"] == 0).sum()
            
#         st.write(f"**Users with Improved Delivery**: {improve_count}")
#         st.write(f"**Users with No Improvement**: {no_improve_count}")
            
#         # Optional: Pie chart
#         fig_pie = px.pie(
#             names=["Improved", "No Improvement"],
#             values=[improve_count, no_improve_count],
#             title="Delivery Improvement Prediction Results"
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#         # --- AI-Powered Recommendation for Manual DC Location ---
#         st.markdown("### AI-Powered Recommendation (Manual DC)")
        
#         manual_prompt = f"""
#         You are a logistics expert.
        
#         The user has manually proposed new DC (Distribution Center) locations.
#         Prediction results based on these inputs show:
#         - Improved delivery cases: {improve_count}
#         - No improvement cases: {no_improve_count}
        
#         Generate a short logistics insight explaining whether the proposed DCs are strategically placed. 
#         Explain possible reasons for their effectiveness or shortcomings.
#         """
        
#         with st.spinner("Generating AI recommendation..."):
#             try:
#                 llama_output_manual = generate_llama_response(manual_prompt)
#                 st.success("AI-generated Recommendation:")
#                 st.markdown(llama_output_manual)
#             except Exception as e:
#                 st.error(f"Error generating recommendation: {e}")


# # --- TAB 3: Comparison Summary ---
# with tab3:
#     st.header("Comparison of Cluster-based vs Manual DC Placement")

#     if "simulated_df" not in locals() or "df" not in locals():
#         st.warning("Please run predictions in both tabs first (Cluster-based and Manual) to view comparison.")
#     else:
#         # Get prediction counts
#         cluster_improve = (df["delivery_time_improvement_pred"] == 1).sum()
#         cluster_no_improve = (df["delivery_time_improvement_pred"] == 0).sum()

#         manual_improve = (simulated_df["delivery_time_improvement_pred"] == 1).sum()
#         manual_no_improve = (simulated_df["delivery_time_improvement_pred"] == 0).sum()

#         st.subheader("Numeric Comparison")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Cluster-based Improved", cluster_improve)
#             st.metric("Cluster-based No Improvement", cluster_no_improve)
#         with col2:
#             st.metric("Manual Improved", manual_improve)
#             st.metric("Manual No Improvement", manual_no_improve)

#         # Pie Charts Side by Side
#         st.subheader("Visual Comparison")

#         col1, col2 = st.columns(2)
#         with col1:
#             fig1 = px.pie(
#                 names=["Improved", "No Improvement"],
#                 values=[cluster_improve, cluster_no_improve],
#                 title="Cluster-based DCs"
#             )
#             st.plotly_chart(fig1, use_container_width=True)
#         with col2:
#             fig2 = px.pie(
#                 names=["Improved", "No Improvement"],
#                 values=[manual_improve, manual_no_improve],
#                 title="Manual DCs"
#             )
#             st.plotly_chart(fig2, use_container_width=True)

#         # Optional: Bar Chart Summary
#         st.subheader("Bar Chart Comparison")
#         compare_df = pd.DataFrame({
#             "Method": ["Cluster-based", "Cluster-based", "Manual", "Manual"],
#             "Outcome": ["Improved", "No Improvement", "Improved", "No Improvement"],
#             "Count": [cluster_improve, cluster_no_improve, manual_improve, manual_no_improve]
#         })
#         fig_bar = px.bar(compare_df, x="Method", y="Count", color="Outcome", barmode="group", title="Prediction Outcome Comparison")
#         st.plotly_chart(fig_bar, use_container_width=True)

# --- Imports ---
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import requests
from preprocessing_utils import HighCardinalityDropper, ColumnDropper

# --- Load Secrets ---
TOGETHER_API_KEY = st.secrets["togetherai"]["api_key"]

# --- Helper Functions ---
def generate_llama_response(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a logistics expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    response = requests.post(url, json=payload, headers=headers)
    result = response.json()
    return result["choices"][0]["message"]["content"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def predict_and_display(df_input, model, title_prefix=""):
    processed = preprocessing_pipeline.transform(df_input)
    preds = model.predict(processed)
    probs = model.predict_proba(processed)[:, 1]
    df_input["delivery_time_improvement_pred"] = preds
    df_input["improvement_probability"] = probs
    
    improved = (preds == 1).sum()
    not_improved = (preds == 0).sum()

    st.markdown(f"### {title_prefix} Prediction Summary")
    st.write(f"**Users with Improved Delivery**: {improved}")
    st.write(f"**Users with No Improvement**: {not_improved}")

    fig = px.pie(
        names=["Improved", "No Improvement"],
        values=[improved, not_improved],
        title=f"{title_prefix} Delivery Improvement Prediction Results"
    )
    st.plotly_chart(fig, use_container_width=True)

    return df_input, improved, not_improved

def generate_ai_summary(improved, not_improved, context_label):
    prompt = f"""
    You are a logistics expert.
    The user has proposed a {context_label} distribution center (DC) strategy.
    Results:
    - Improved: {improved}
    - Not Improved: {not_improved}
    Provide a logistics-based explanation of the outcome.
    """
    with st.spinner("Generating AI recommendation..."):
        try:
            summary = generate_llama_response(prompt)
            st.success("AI-generated Recommendation:")
            st.markdown(summary)
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
    return summary

# --- Page Setup ---
st.set_page_config(page_title="DC Placement App", layout="wide")
st.title("Distribution Center Suggestion Dashboard")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Jojooyee/NewDCPlacement-app/main/test_df.csv"
    return pd.read_csv(url)

df = load_data()
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("delivery_improvement_model.pkl")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Cluster-based DC Suggestion", "Manual Proposed DC Location", "Comparison Summary"])

# --- Tab 1 ---
with tab1:
    st.header("Cluster-based DC Suggestion")
    option = st.selectbox("Select Suggestion Result:", ["New DC Location", "Clustering Report"])
    df["cluster"] = df["cluster"].astype(str)
    state_df = df.drop_duplicates("state")[
        ["state", "order_volume", "avg_delivery_time_days", "state_latitude", "state_longitude", "cluster", "new_dc_latitude", "new_dc_longitude"]
    ].sort_values("state")

    if option == "New DC Location":
        st.markdown("### Suggested New DC Coordinates")
        coords = state_df.groupby("cluster")[["new_dc_latitude", "new_dc_longitude"]].first().reset_index()
        for _, row in coords.iterrows():
            st.markdown(f"**Cluster {row['cluster']}**: ({row['new_dc_latitude']:.4f}, {row['new_dc_longitude']:.4f})")

        st.markdown("### Map of New DC Locations")
        fig_map = px.scatter_mapbox(
            state_df, lat="new_dc_latitude", lon="new_dc_longitude", color="cluster", hover_name="state",
            hover_data={"cluster": True}, zoom=2, height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_traces(marker=dict(size=10, opacity=1.0))
        st.plotly_chart(fig_map, use_container_width=True)

        df, improved1, not_improved1 = predict_and_display(df, model, "Cluster-based")
        generate_ai_summary(improved1, not_improved1, "cluster-based")

    elif option == "Clustering Report":
        st.markdown("### State-Level Summary")
        col1, col2 = st.columns(2)
        top_n = col1.selectbox("States to display", [10, 20, 30, 40, 50], 1)
        sort_order = col2.selectbox("Sort by", ["Lowest", "Highest"])
        asc = sort_order == "Lowest"

        vol_df = state_df.sort_values("order_volume", ascending=asc).head(top_n)
        del_df = state_df.sort_values("avg_delivery_time_days", ascending=asc).head(top_n)

        st.plotly_chart(px.bar(vol_df, x="state", y="order_volume", color="order_volume", title="Order Volume"), use_container_width=True)
        st.plotly_chart(px.bar(del_df, x="state", y="avg_delivery_time_days", color="avg_delivery_time_days", title="Avg Delivery Time (Days)"), use_container_width=True)

        st.markdown("### Cluster Map")
        map_fig = px.scatter_mapbox(state_df, lat="state_latitude", lon="state_longitude", color="cluster", hover_name="state", zoom=2)
        map_fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(map_fig, use_container_width=True)

        st.markdown("### Demand Ranking")
        ranking = df.groupby("cluster")["composite_weight"].sum().sort_values(ascending=False).reset_index()
        for i, row in ranking.iterrows():
            st.markdown(f"**{i+1}. Cluster {row['cluster']}**")
            cl_df = state_df[state_df["cluster"] == row["cluster"]]
            with st.expander("Expand for detail"):
                st.markdown(f"States: `{cl_df['state'].nunique()}`")
                st.markdown(f"Avg Order Volume: `{cl_df['order_volume'].mean():,.0f}`")
                st.markdown(f"Avg Delivery Time: `{cl_df['avg_delivery_time_days'].mean():.2f}`")

# --- Tab 2 ---
with tab2:
    st.header("Manual DC Simulation")
    num_points = st.number_input("Number of DC locations:", 1, 10, 1)
    user_locations = [(st.number_input(f"Lat {i+1}", key=f"lat_{i}"), st.number_input(f"Lon {i+1}", key=f"lon_{i}")) for i in range(num_points)]

    if st.button("Simulate", key="simulate_dc"):
        st.markdown("### Proposed DC Locations Map")
        df_user_dc = pd.DataFrame(user_locations, columns=["lat", "lon"])
        df_user_dc["dc_id"] = [f"DC {i+1}" for i in range(num_points)]
        map_fig = px.scatter_mapbox(df_user_dc, lat="lat", lon="lon", hover_name="dc_id", zoom=2)
        map_fig.update_layout(mapbox_style="open-street-map")
        map_fig.update_traces(marker=dict(size=10, color="red"))
        st.plotly_chart(map_fig, use_container_width=True)

        new_rows = []
        for _, row in df.iterrows():
            dists = [(haversine(row.user_latitude, row.user_longitude, lat, lon), lat, lon) for lat, lon in user_locations]
            nearest = min(dists, key=lambda x: x[0])
            updated = row.copy()
            updated["new_dc_latitude"], updated["new_dc_longitude"] = nearest[1], nearest[2]
            updated["distance_new_dc_to_user_km"] = nearest[0]
            new_rows.append(updated)

        sim_df = pd.DataFrame(new_rows)
        sim_df["delivery_time_hour"] = sim_df["delivery_time_days"] * 24
        sim_df["delivery_speed_kmph"] = sim_df["distance_dc_to_user_km"] / sim_df["delivery_time_hour"]
        sim_df["estimated_new_delivery_time"] = sim_df["distance_new_dc_to_user_km"] / sim_df["delivery_speed_kmph"]
        sim_df["delivery_time_improvement"] = sim_df["delivery_time_hour"] - sim_df["estimated_new_delivery_time"]

        sim_df, improved2, not_improved2 = predict_and_display(sim_df, model, "Manual")
        generate_ai_summary(improved2, not_improved2, "manual")

# --- Tab 3 ---
with tab3:
    st.header("Comparison of Cluster-based vs Manual DC Placement")
    if "sim_df" not in locals() or "df" not in locals():
        st.warning("Please run predictions in both tabs first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cluster Improved", improved1)
            st.metric("Cluster No Improvement", not_improved1)
        with col2:
            st.metric("Manual Improved", improved2)
            st.metric("Manual No Improvement", not_improved2)

        comp_df = pd.DataFrame({
            "Method": ["Cluster", "Cluster", "Manual", "Manual"],
            "Outcome": ["Improved", "No Improvement"]*2,
            "Count": [improved1, not_improved1, improved2, not_improved2]
        })
        st.plotly_chart(px.bar(comp_df, x="Method", y="Count", color="Outcome", barmode="group"), use_container_width=True)
        st.plotly_chart(px.pie(comp_df, names="Outcome", values="Count", title="Overall Comparison"), use_container_width=True)
