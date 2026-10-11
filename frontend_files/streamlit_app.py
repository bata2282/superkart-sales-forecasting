import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000")

FEATURES = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

st.set_page_config(page_title="SuperKart Sales Forecasting", layout="wide")
st.title("SuperKart Sales Forecasting")
st.write("Predict product sales using the trained machine learning model.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Predict sales for one product")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_id = st.text_input("Product ID", "FD001")
            product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
            sugar = st.selectbox(
                "Sugar Content",
                ["Low Sugar", "Regular", "No Sugar"]
            )
            allocated_area = st.number_input(
                "Product Allocated Area", min_value=0.0, value=0.05
            )
            product_type = st.text_input("Product Type", "Dairy")
            product_mrp = st.number_input("Product MRP", min_value=0.0, value=100.0)

        with col2:
            store_id = st.text_input("Store ID", "OUT049")
            establishment_year = st.number_input(
                "Store Establishment Year", min_value=1900, max_value=2100, value=2000
            )
            store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
            city_type = st.selectbox(
                "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
            )
            store_type = st.text_input("Store Type", "Supermarket Type1")

        submitted = st.form_submit_button("Predict Sales")

    if submitted:
        payload = {
            "Product_Id": product_id,
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type
        }

        try:
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=60
            )
            if response.ok:
                prediction = response.json()["predicted_sales"]
                st.success(f"Predicted Sales: {prediction:,.2f}")
            else:
                st.error(f"Prediction failed: {response.text}")
        except requests.RequestException as error:
            st.error(f"Could not connect to the inference API: {error}")

with tab2:
    st.subheader("Batch sales prediction")
    st.write("Upload a CSV containing all 11 model input features.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())

        missing = [c for c in FEATURES if c not in batch_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button("Run Batch Prediction"):
            try:
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, (_, row) in enumerate(batch_df.iterrows()):
                    payload = row[FEATURES].to_dict()
                    response = requests.post(
                        f"{API_URL}/predict", json=payload, timeout=60
                    )
                    response.raise_for_status()
                    results.append(response.json()["predicted_sales"])
                    progress.progress((i + 1) / len(batch_df))
                    status.write(f"Processed {i + 1} of {len(batch_df)} rows")

                batch_df["Predicted_Sales"] = results
                st.success("Batch prediction completed.")
                st.dataframe(batch_df)

                st.download_button(
                    "Download Predictions CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "superkart_predictions.csv",
                    "text/csv"
                )

            except (requests.RequestException, ValueError, TypeError) as error:
                st.error(f"Batch prediction failed: {error}")
