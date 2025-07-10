import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load models and data
product_sim_df = joblib.load("product_similarity.pkl")
product_names = joblib.load("product_names.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# 🧠 Function: Product Recommender
# -------------------------------
def recommend_by_product_name(product_name, top_n=5):
    def get_code_by_name(name):
        for code, desc in product_names.items():
            if name.lower() in desc.lower():
                return code
        return None

    def recommend_products(stock_code, top_n):
        if stock_code not in product_sim_df:
            return [f"❌ StockCode {stock_code} not found."]
        sim_scores = product_sim_df[stock_code].sort_values(ascending=False).drop(stock_code)
        top_similar = sim_scores.head(top_n).index
        return [f"{code} - {product_names.get(code, 'Unknown Product')}" for code in top_similar]

    code = get_code_by_name(product_name)
    if code:
        return recommend_products(code, top_n)
    else:
        return [f"❌ Product name '{product_name}' not found."]

# -------------------------------
# 🧠 Function: Segment Predictor
# -------------------------------
def predict_customer_segment(recency, frequency, monetary):
    input_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(input_scaled)[0]

    # Map cluster to segment name manually (adjust based on your actual cluster analysis)
    cluster_map = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }

    return cluster_map.get(cluster, "Unknown")

# -------------------------------
# 🎯 Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum")
st.markdown("**Customer Segmentation & Product Recommendation App**")

tab1, tab2 = st.tabs(["📱 Product Recommendation", "👤 Customer Segmentation"])

# ---------------------------------
# 📱 Product Recommendation Module
# ---------------------------------
with tab1:
    st.header("🎯 Product Recommendation")

    # Create dropdown from product name list (sorted for easier browsing)
    product_desc_list = sorted(list(set(product_names.values())))
    selected_product = st.selectbox("Select a product:", product_desc_list)

    if st.button("🔍 Get Recommendations"):
        if selected_product:
            recs = recommend_by_product_name(selected_product)
            st.subheader("Top 5 Similar Products")
            for i, r in enumerate(recs, 1):
                st.success(f"{i}. {r}")
        else:
            st.warning("Please select a product.")

# ---------------------------------
# 👤 Customer Segmentation Module
# ---------------------------------
with tab2:
    st.header("🎯 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)

    if st.button("🧠 Predict Cluster"):
        segment = predict_customer_segment(recency, frequency, monetary)
        st.success(f"🧾 This customer belongs to the **{segment}** segment.")
