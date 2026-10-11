
with tab2:
    st.subheader("Batch sales prediction")
    st.write(
        "Upload a CSV containing the model input features. "
        "If columns are missing, provide appropriate defaults."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.strip()

        st.write("Uploaded data preview")
        st.dataframe(batch_df.head())

        missing = [c for c in FEATURES if c not in batch_df.columns]

        if missing:
            st.warning(f"Missing columns: {missing}")
            st.write("Provide a default value for each missing column.")

            defaults = {}

            for col in missing:
                if col in [
                    "Product_Weight",
                    "Product_Allocated_Area",
                    "Product_MRP",
                    "Store_Establishment_Year"
                ]:
                    defaults[col] = st.number_input(
                        f"Default for {col}",
                        value=1.0,
                        key=f"default_{col}"
                    )
                else:
                    defaults[col] = st.text_input(
                        f"Default for {col}",
                        key=f"default_{col}"
                    )

            if st.button("Apply Defaults"):
                for col, value in defaults.items():
                    batch_df[col] = value

                st.session_state["batch_df"] = batch_df
                st.success("Defaults applied. Review the data before predicting.")

        if "batch_df" in st.session_state:
            prediction_df = st.session_state["batch_df"]

            st.dataframe(prediction_df.head())

            if st.button("Run Batch Prediction"):
                try:
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    for i, (_, row) in enumerate(prediction_df.iterrows()):
                        payload = {
                            col: row[col]
                            for col in FEATURES
                        }

                        response = requests.post(
                            f"{API_URL}/predict",
                            json=payload,
                            timeout=60
                        )
                        response.raise_for_status()
                        results.append(
                            response.json()["predicted_sales"]
                        )
                        progress.progress((i + 1) / len(prediction_df))
                        status.write(
                            f"Processed {i + 1} of {len(prediction_df)} rows"
                        )

                    prediction_df["Predicted_Sales"] = results
                    st.success("Batch prediction completed.")
                    st.dataframe(prediction_df)

                    st.download_button(
                        "Download Predictions CSV",
                        prediction_df.to_csv(index=False).encode("utf-8"),
                        "superkart_predictions.csv",
                        "text/csv"
                    )

                except (
                    requests.RequestException,
                    ValueError,
                    TypeError
                ) as error:
                    st.error(f"Batch prediction failed: {error}")
