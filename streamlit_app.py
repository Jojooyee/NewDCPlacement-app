import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from preprocessing_utils import HighCardinalityDropper, ColumnDropper
import requests

# Load Data
@st.cache_data
def load_data():
    # Replace with your actual GitHub raw URL
    url = "https://raw.githubusercontent.com/Jojooyee/NewDCPlacement-app/main/test_df.csv"
    df = pd.read_csv(url)
    return df
    
df = load_data()

# Load Preprocess Pipeline
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")

# Load trained prediction model
model = joblib.load("delivery_improvement_model.pkl")

# Load Together AI key from secrets
TOGETHER_API_KEY = st.secrets["togetherai"]["api_key"]

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

def show_ai_recommendation(improve_count, no_improve_count, context="suggested clustered-based DC locations"):
    st.markdown("### AI-Generated Recommendation")

    prompt = f"""
    Based on delivery improvement prediction:
    - Improved: {improve_count}
    - Not Improved: {no_improve_count}

    Give short explanation on why the {context} are effective and what logistics reasoning supports it.
    """

    with st.spinner("Generating AI recommendation..."):
        try:
            llama_output = generate_llama_response(prompt)
            st.success("AI-generated Recommendation:")
            st.markdown(llama_output)
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# --- Page Setup ---
st.set_page_config(page_title="DC Placement App", layout="wide")
st.title("Distribution Center Suggestion Dashboard")

# --- Create Tabs Instead of Sidebar Navigation ---
tab1, tab2, tab3 = st.tabs(["Cluster-based DC Suggestion", "Manual Proposed DC Location","Comparison Summary"])

# --- TAB 1: New DC Suggestion ---
with tab1:
    st.header("New Distribution Center Suggestion")
    st.markdown("Use one of the options below to explore potential locations for new distribution centers.")

    result_option = st.selectbox("Select Suggestion Result:", ["New DC Location", "Clustering Report"])

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

        # Transform input features
        processed_df = preprocessing_pipeline.transform(df)
        
        # Predict
        predictions = model.predict(processed_df)
        prediction_probs = model.predict_proba(processed_df)[:, 1]
        
        # Append predictions to the ORIGINAL dataframe
        df["delivery_time_improvement_pred"] = predictions
        df["improvement_probability"] = prediction_probs

        st.markdown("### Prediction Summary")
        improve_count = (df["delivery_time_improvement_pred"] == 1).sum()
        no_improve_count = (df["delivery_time_improvement_pred"] == 0).sum()
            
        st.write(f"**Users with Improved Delivery**: {improve_count}")
        st.write(f"**Users with No Improvement**: {no_improve_count}")
            
        # Pie chart
        fig_pie = px.pie(
            names=["Improved", "No Improvement"],
            values=[improve_count, no_improve_count],
            title="Delivery Improvement Prediction Results"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # AI Recommendation
        # st.markdown("### AI-Generated Recommendation")
        show_ai_recommendation(improve_count, no_improve_count)
        
        # prompt = f"""
        # Based on delivery improvement prediction:
        # - Improved: {improve_count}
        # - Not Improved: {no_improve_count}
        # Explain why the suggested new DC locations are effective and what logistics reasoning supports it.
        # """
        
        # with st.spinner("Generating AI recommendation..."):
        #     try:
        #         llama_output = generate_llama_response(prompt)
        #         st.success("AI-generated Recommendation:")
        #         st.markdown(llama_output)
        #     except Exception as e:
        #         st.error(f"Error generating recommendation: {e}")
                
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

# --- TAB 2: Manual Proposed DC Location---
with tab2:
    st.subheader("Manual DC Simulation")
    st.markdown("Simulate multiple proposed DC locations by entering coordinates below.")

    # Step 1: Let user input how many DCs they want to enter
    num_points = st.number_input("Enter number of proposed DC locations:", min_value=1, max_value=10, value=1, step=1)

    # Step 2: Show input fields dynamically based on that number
    new_dc_locations = []
    for i in range(int(num_points)):
        st.markdown(f"#### DC Location {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input(f"Latitude {i + 1}", key=f"lat_{i}", value=0.0, format="%.6f")
        with col2:
            lon = st.number_input(f"Longitude {i + 1}", key=f"lon_{i}", value=0.0, format="%.6f")
        new_dc_locations.append((lat, lon))

    # Step 3: Simulate Button — FIXED with a key
    if st.button("Simulate", key="simulate_dc_locations"):
    
        # Step 2.5: Plot user-input coordinates on a map
        if new_dc_locations:
            st.markdown("### Preview of Proposed DC Locations")
            user_dc_df = pd.DataFrame(new_dc_locations, columns=["lat", "lon"])
            user_dc_df["dc_id"] = [f"DC {i+1}" for i in range(len(new_dc_locations))]
    
            fig_user_dc = px.scatter_mapbox(
                user_dc_df,
                lat="lat",
                lon="lon",
                hover_name="dc_id",
                zoom=2,
                height=400
            )
            fig_user_dc.update_layout(mapbox_style="open-street-map")
            fig_user_dc.update_traces(marker=dict(size=10, color="red"))
            st.plotly_chart(fig_user_dc, use_container_width=True)

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

        st.markdown("### Prediction Summary")
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

        # --- AI-Powered Recommendation for Manual DC Location ---
        # st.markdown("### AI-Powered Recommendation (Manual DC)")
        show_ai_recommendation(improve_count, no_improve_count, context="manual proposed DC locations")
        
        # manual_prompt = f"""
        # You are a logistics expert.
        
        # The user has manually proposed new DC (Distribution Center) locations.
        # Prediction results based on these inputs show:
        # - Improved delivery cases: {improve_count}
        # - No improvement cases: {no_improve_count}
        
        # Generate a short logistics insight explaining whether the proposed DCs are strategically placed. 
        # Explain possible reasons for their effectiveness or shortcomings.
        # """
        
        # with st.spinner("Generating AI recommendation..."):
        #     try:
        #         llama_output_manual = generate_llama_response(manual_prompt)
        #         st.success("AI-generated Recommendation:")
        #         st.markdown(llama_output_manual)
        #     except Exception as e:
        #         st.error(f"Error generating recommendation: {e}")


# --- TAB 3: Comparison Summary ---
with tab3:
    st.header("Comparison of Cluster-based vs Manual DC Placement")

    if "simulated_df" not in locals() or "df" not in locals():
        st.warning("Please run predictions in both tabs first (Cluster-based and Manual) to view comparison.")
    else:
        # Get prediction counts
        cluster_improve = (df["delivery_time_improvement_pred"] == 1).sum()
        cluster_no_improve = (df["delivery_time_improvement_pred"] == 0).sum()

        manual_improve = (simulated_df["delivery_time_improvement_pred"] == 1).sum()
        manual_no_improve = (simulated_df["delivery_time_improvement_pred"] == 0).sum()

        st.subheader("Numeric Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cluster-based Improved", cluster_improve)
            st.metric("Cluster-based No Improvement", cluster_no_improve)
        with col2:
            st.metric("Manual Improved", manual_improve)
            st.metric("Manual No Improvement", manual_no_improve)

        # Pie Charts Side by Side
        st.subheader("Visual Comparison")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(
                names=["Improved", "No Improvement"],
                values=[cluster_improve, cluster_no_improve],
                title="Cluster-based DCs"
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(
                names=["Improved", "No Improvement"],
                values=[manual_improve, manual_no_improve],
                title="Manual DCs"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Optional: Bar Chart Summary
        st.subheader("Bar Chart Comparison")
        compare_df = pd.DataFrame({
            "Method": ["Cluster-based", "Cluster-based", "Manual", "Manual"],
            "Outcome": ["Improved", "No Improvement", "Improved", "No Improvement"],
            "Count": [cluster_improve, cluster_no_improve, manual_improve, manual_no_improve]
        })
        fig_bar = px.bar(compare_df, x="Method", y="Count", color="Outcome", barmode="group", title="Prediction Outcome Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)
