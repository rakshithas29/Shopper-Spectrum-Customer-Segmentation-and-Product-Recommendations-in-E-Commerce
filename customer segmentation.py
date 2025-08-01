import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ✅ Set full paths to your model files
MODEL_DIR = r"C:\Users\Rakshitha\customer\models"
kmeans_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
similarity_path = os.path.join(MODEL_DIR, "similarity_matrix.pkl")

# ✅ Check all model files exist
missing_files = [p for p in [kmeans_path, scaler_path, similarity_path] if not os.path.exists(p)]
if missing_files:
    st.error(f"❌ Missing model files: {missing_files}")
    st.stop()

# ✅ Load models safely
with open(kmeans_path, "rb") as f:
    kmeans = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(similarity_path, "rb") as f:
    similarity_df = pickle.load(f)

# ✅ Load product descriptions from the original dataset
DATA_PATH = r"C:\Users\Rakshitha\Downloads\online_retail.csv"
df_products = pd.read_csv(DATA_PATH)
df_products.dropna(subset=['StockCode', 'Description'], inplace=True)
code_to_name = df_products.drop_duplicates(subset='StockCode')[['StockCode', 'Description']].set_index('StockCode')['Description'].to_dict()
name_to_code = {v.upper(): k for k, v in code_to_name.items()}

# ----------------------------
# Streamlit App Setup
# ----------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ----------------------------
# Project Title (Centered)
# ----------------------------
st.markdown("""
    <h1 style='text-align: center; color: #f05454; font-size: 2.4rem; font-weight: 700; margin-bottom: 2rem;'>
        🛍️ Shopper Spectrum: Customer Insights & Product Recommendations
    </h1>
""", unsafe_allow_html=True)

# ----------------------------
# Custom CSS for Enhanced Styling
# ----------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        .stButton button {
            background-color: white;
            color: red;
            border: 1px solid red;
            padding: 8px 20px;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            background-color: red;
            color: white;
            transform: scale(1.05);
        }

        .recommend-box {
            background-color: #f9f9f9;
            border-left: 5px solid red;
            padding: 1rem 1.5rem;
            margin-top: 1rem;
            border-radius: 6px;
        }

        .product-header {
            font-size: 1.3rem;
            font-weight: bold;
            color: #ffffff;
            margin-top: 1rem;
        }

        h2.section-title {
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .sidebar .sidebar-content {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🛍️ Shopper Spectrum")
option = st.sidebar.radio("📊 Select Module", ["Clustering", "Recommendation"])

# ----------------------------
# Clustering Module
# ----------------------------
if option == "Clustering":
    st.markdown("<h2 class='section-title'>👥 Customer Segmentation</h2>", unsafe_allow_html=True)

    recency = st.number_input("🕒 Recency (in days)", min_value=0, value=30)
    frequency = st.number_input("🔁 Frequency (Number of Purchases)", min_value=0, value=5)
    monetary = st.number_input("💰 Monetary (Total Spend)", min_value=0.0, value=100.0)

    if st.button("🔍 Predict Segment"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]

        segment_map = {
            0: "🌟 High-Value",
            1: "📦 Regular",
            2: "⏳ Occasional",
            3: "⚠️ At-Risk"
        }

        segment = segment_map.get(cluster, "Unknown")
        st.success(f"Predicted Customer Segment: **{segment}**")

# ----------------------------
# Product Recommendation Module
# ----------------------------
elif option == "Recommendation":
    st.markdown("<h2 class='section-title'>🛒 Product Recommender</h2>", unsafe_allow_html=True)

    product_name = st.selectbox(
        "🔍 Select a Product",
        options=sorted(name_to_code.keys()),
        index=0
    )

    def recommend_products(stock_code, top_n=5):
        if stock_code not in similarity_df.columns:
            return []
        similar_items = similarity_df[stock_code].sort_values(ascending=False).drop(stock_code)
        return similar_items.head(top_n).index.tolist()

    if st.button("📦 Recommend"):
        matched_code = name_to_code.get(product_name.upper())
        if matched_code:
            recommendations = recommend_products(matched_code)
            if recommendations:
                st.markdown('<div class="product-header">📌 Recommended Products:</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommend-box">', unsafe_allow_html=True)
                for item in recommendations:
                    desc = code_to_name.get(item, "Description not found")
                    st.markdown(f"- {desc}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No similar products found.")
        else:
            st.error("Product not found. Please check the name and try again.")
