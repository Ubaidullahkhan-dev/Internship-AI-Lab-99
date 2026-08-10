import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

st.markdown("""
    <h2 style='text-align: center; color: #1E88E5;'>🏡 House Price Prediction Portal</h2>
    <p style='text-align: center; color: #666;'>  Automated Machine Learning Prediction Engine</p>
    <hr>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Model and Model Columns
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('best_model.pkl')
        # Agar exact model columns file mojood hai
        feature_columns = joblib.load('model_columns.pkl')
        return model, feature_columns
    except Exception:
        try:
            model = joblib.load('best_model.pkl')
            # Fallback to model's inherent feature names
            feature_columns = list(model.feature_names_in_)
            return model, feature_columns
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            return None, None

model, feature_columns = load_resources()

# ---------------------------------------------------------
# User Inputs Interface
# ---------------------------------------------------------
if model is not None and feature_columns is not None:
    st.subheader("📋 Enter Key House Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual = st.slider("Overall Quality (OverallQual)", 1, 10, 7)
        gr_liv_area = st.number_input("Above Grade Area SqFt (GrLivArea)", min_value=300, max_value=10000, value=1500)

    with col2:
        year_built = st.number_input("Year Built (YearBuilt)", min_value=1800, max_value=2026, value=2005)
        total_bsmt_sf = st.number_input("Total Basement SqFt (TotalBsmtSF)", min_value=0, max_value=5000, value=800)

    with col3:
        garage_cars = st.number_input("Garage Capacity Cars (GarageCars)", min_value=0, max_value=5, value=2)
        tot_rms = st.number_input("Total Rooms Above Grade (TotRmsAbvGrd)", min_value=1, max_value=15, value=6)

    st.divider()

    if st.button("🚀 Calculate Estimated Price", type="primary", use_container_width=True):
        try:
            # Create a dataframe with 0s matching EXACT model expected features
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

            # Assign user inputs to matching feature columns
            if 'OverallQual' in input_df.columns: input_df['OverallQual'] = overall_qual
            if 'GrLivArea' in input_df.columns: input_df['GrLivArea'] = gr_liv_area
            if 'YearBuilt' in input_df.columns: input_df['YearBuilt'] = year_built
            if 'TotalBsmtSF' in input_df.columns: input_df['TotalBsmtSF'] = total_bsmt_sf
            if 'GarageCars' in input_df.columns: input_df['GarageCars'] = garage_cars
            if 'TotRmsAbvGrd' in input_df.columns: input_df['TotRmsAbvGrd'] = tot_rms

            # Predict
            predicted_price = model.predict(input_df)[0]

            st.success("Price Calculated Successfully!")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="💰 Predicted House Price", value=f"${predicted_price:,.2f}")
            
            with res_col2:
                st.info(f"💡 Input formatted to match all {len(feature_columns)} trained model features.")

        except Exception as err:
            st.error(f"Prediction Error: {err}")

st.divider()
st.markdown("<p style='text-align: center; color: gray;'>House Price Prediction System | Ubaidullah Khan</p>", unsafe_allow_html=True)