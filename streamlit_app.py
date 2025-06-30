import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from preprocessing_utils import HighCardinalityDropper, ColumnDropper

# --- Page Setup ---
st.set_page_config(page_title="DC Placement App", layout="wide")
st.title("Distribution Center Optimization Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Replace with your actual file ID
    file_id = "1VW3Kng9mu7EzAs7tkCZf-CUzllKaaFpf"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- Load Pipeline ---
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")

# --- Load trained prediction model ---
model = joblib.load("delivery_improvement_model.pkl")

# --- Create Tabs Instead of Sidebar Navigation ---
tab1, tab2 = st.tabs(["New DC Placement", "Delivery Time Improvement Prediction"])

# --- TAB 1: New DC Suggestion ---
with tab1:
    st.header("New Distribution Center Suggestion")
    st.markdown("Use one of the options below to explore potential locations for new distribution centers.")

    suggestion_option = st.selectbox("Select Suggestion Method:", [
        "Cluster-Based DC Suggestion",
        "Manual DC Simulation"
    ])

    if suggestion_option == "Cluster-Based DC Suggestion":
        result_option = st.selectbox("Select Suggestion Result:", [
            "New DC Location",
            "Clustering Report"
        ])

        state_level_df = df.drop_duplicates(subset=["state"])[
            ["state", "order_volume", "avg_delivery_time_days", "state_latitude", "state_longitude", "cluster", "new_dc_latitude", "new_dc_longitude"]
        ].sort_values("state")

        state_level_df["cluster"] = state_level_df["cluster"].astype(str)

        if result_option == "New DC Location":
            # Section 1: List of new dc location
            st.markdown("### Suggested New DC Coordinates")
            st.markdown("The list below shows the location of new dc by cluster.")

            unique_dc_locations = state_level_df.groupby("cluster")[["new_dc_latitude", "new_dc_longitude"]].first().reset_index()

            for _, row in unique_dc_locations.iterrows():
                st.markdown(f"**Cluster {row['cluster']}**: ({row['new_dc_latitude']:.4f}, {row['new_dc_longitude']:.4f})")

            # Section 2: New dc location in map
            st.markdown("### New Distribution Center Location")
            st.markdown("The map below shows the location of new dc.")

            fig_map = px.scatter_mapbox(
                state_level_df,
                lat="new_dc_latitude",
                lon="new_dc_longitude",
                color="cluster",
                hover_name="state",
                hover_data={"cluster": True},
                zoom=2,
                height=500
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_traces(marker=dict(size=10, opacity=1.0))
            st.plotly_chart(fig_map, use_container_width=True)

        elif result_option == "Clustering Report":
            # Section 1: Order volume & Avg delivery time
            st.markdown("### State-Level Summary")
            st.markdown("Below are the visualizations of order volume and average delivery time by state from the clustered dataset.")

            col1, col2 = st.columns(2)
            with col1:
                top_n = st.selectbox("Select number of states to display", [10, 20, 30, 40, 50], index=1)
            with col2:
                sort_order = st.selectbox("Sort by:", ["Lowest", "Highest"])

            sort_ascending = sort_order == "Lowest"

            top_order_volume_df = state_level_df.sort_values("order_volume", ascending=sort_ascending).head(top_n)
            fig_order = px.bar(
                top_order_volume_df,
                x="state",
                y="order_volume",
                title="Order Volume",
                labels={"order_volume": "Order Volume", "state": "State"},
                color="order_volume",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_order, use_container_width=True)

            top_delivery_df = state_level_df.sort_values("avg_delivery_time_days", ascending=sort_ascending).head(top_n)
            fig_delivery = px.bar(
                top_delivery_df,
                x="state",
                y="avg_delivery_time_days",
                title="Avg Delivery Time (Days)",
                labels={"avg_delivery_time_days": "Avg Delivery Time (Days)", "state": "State"},
                color="avg_delivery_time_days",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig_delivery, use_container_width=True)

            # Section 2: Cluster map
            st.markdown("### Cluster Map")
            st.markdown("The map below shows the location of each state, colored by assigned cluster.")

            fig_map = px.scatter_mapbox(
                state_level_df,
                lat="state_latitude",
                lon="state_longitude",
                color="cluster",
                hover_name="state",
                hover_data={"cluster": True},
                zoom=2,
                height=500
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_traces(marker=dict(size=10, opacity=1.0))
            st.plotly_chart(fig_map, use_container_width=True)

            # Section 3: Cluster demand ranking
            st.markdown("### Demand Ranking")
            st.markdown("Clusters are ranked based on their total composite weight (indicating demand concentration).")
            df["cluster"] = df["cluster"].astype(str)

            cluster_ranking = (
                df.groupby("cluster")["composite_weight"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            for i, row in cluster_ranking.iterrows():
                cluster_id = row["cluster"]
                st.markdown(f"**{i+1}. Cluster {cluster_id}**")

                cluster_data = state_level_df[state_level_df["cluster"] == cluster_id]
                total_states = cluster_data["state"].nunique()
                list_states = sorted(cluster_data["state"].unique().tolist())
                total_order_volume = cluster_data["order_volume"].mean()
                avg_delivery_time = cluster_data["avg_delivery_time_days"].mean()

                with st.expander("Expand for detail"):
                    st.markdown(f"**Total number of states**: `{total_states}`")
                    st.markdown(f"**Average order volume**: `{total_order_volume:,}`")
                    st.markdown(f"**Average delivery time (days)**: `{avg_delivery_time:.2f}`")

    elif suggestion_option == "Manual DC Simulation":
        st.subheader("Manual DC Simulation")
        st.markdown("Simulate multiple proposed DC locations by entering coordinates below.")

        # Step 1: Let user input how many DCs they want to enter
        num_points = st.number_input("Enter number of proposed DC locations:", min_value=1, max_value=10, value=1, step=1)

        # Step 2: Show input fields dynamically based on that number
        new_dc_locations = []
        for i in range(int(num_points)):
            st.markdown(f"#### 📍 DC Location {i + 1}")
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input(f"Latitude {i + 1}", key=f"lat_{i}", value=2.5, format="%.6f")
            with col2:
                lon = st.number_input(f"Longitude {i + 1}", key=f"lon_{i}", value=102.5, format="%.6f")
            new_dc_locations.append((lat, lon))

        # Step 3: Simulate Button — FIXED with a key
        if st.button("Simulate", key="simulate_dc_locations"):
    
            # Step 2.5: Plot user-input coordinates on a map
            if new_dc_locations:
                st.markdown("### 🗺️ Preview of Proposed DC Locations")
                user_dc_df = pd.DataFrame(new_dc_locations, columns=["lat", "lon"])
                user_dc_df["dc_id"] = [f"DC {i+1}" for i in range(len(new_dc_locations))]
    
                fig_user_dc = px.scatter_mapbox(
                    user_dc_df,
                    lat="lat",
                    lon="lon",
                    hover_name="dc_id",
                    zoom=4,
                    height=400
                )
                fig_user_dc.update_layout(mapbox_style="open-street-map")
                fig_user_dc.update_traces(marker=dict(size=10, color="red"))
                st.plotly_chart(fig_user_dc, use_container_width=True)

            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # km
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

                return R * c

            updated_rows = []

            for idx, row in df.iterrows():
                user_lat = row["user_latitude"]
                user_lon = row["user_longitude"]

                # Calculate distance from user to every manually entered DC
                distances = []
                for dc_lat, dc_lon in new_dc_locations:
                    distance = haversine(user_lat, user_lon, dc_lat, dc_lon)
                    distances.append((distance, dc_lat, dc_lon))

                # Find the nearest DC
                min_distance, nearest_lat, nearest_lon = min(distances, key=lambda x: x[0])

                # Create updated row
                updated_row = row.copy()
                updated_row["new_dc_latitude"] = nearest_lat
                updated_row["new_dc_longitude"] = nearest_lon
                updated_row["distance_new_dc_to_user_km"] = min_distance

                updated_rows.append(updated_row)

            # Create a new DataFrame with the updated values
            simulated_df = pd.DataFrame(updated_rows)

            # Convert delivery time from days to hours
            simulated_df['delivery_time_hour'] = simulated_df["delivery_time_days"] * 24

            # Calculate delivery speed (km/hour)
            simulated_df["delivery_speed_kmph"] = simulated_df["distance_dc_to_user_km"] / simulated_df['delivery_time_hour']

            # Estimate new delivery time in hours using the same speed
            simulated_df["estimated_new_delivery_time"] = simulated_df["distance_new_dc_to_user_km"] / simulated_df["delivery_speed_kmph"]

            # Calculate improvement in hours
            simulated_df["delivery_time_improvement"] = simulated_df["delivery_time_hour"] - simulated_df["estimated_new_delivery_time"]

            simulated_processed = preprocessing_pipeline.transform(simulated_df)

            # --- Make prediction (binary classification: 1 = improvement, 0 = no improvement) ---
            predictions = model.predict(simulated_processed)
            prediction_probs = model.predict_proba(simulated_processed)[:, 1]  # Probabilities for class 1

            # --- Append predictions to DataFrame ---
            simulated_df["delivery_time_improvement_pred"] = predictions
            simulated_df["improvement_probability"] = prediction_probs

            st.markdown("### 📊 Prediction Summary")
            improve_count = (simulated_df["delivery_time_improvement_pred"] == 1).sum()
            no_improve_count = (simulated_df["delivery_time_improvement_pred"] == 0).sum()
            
            st.write(f"**Users with Improved Delivery**: {improve_count}")
            st.write(f"**Users with No Improvement**: {no_improve_count}")
            
            # Optional: Pie chart
            fig_pie = px.pie(
                names=["Improved", "No Improvement"],
                values=[improve_count, no_improve_count],
                title="Delivery Improvement Prediction Results"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.success("Prediction completed. Results shown above.")

# --- TAB 2: Delivery Time Improvement Prediction ---
with tab2:
    st.header("Delivery Time Improvement Prediction")
    st.markdown("This section uses a trained ML model to predict whether delivery time improves with a new DC.")
    st.success("Delivery Time Improvement Prediction interface loaded!")
