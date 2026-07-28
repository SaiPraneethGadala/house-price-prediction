"""
Streamlit app for predicting Hyderabad house prices using the trained
Linear Regression model.

Run:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/house_price_model.joblib"

AREAS = [
    "Patancheru", "Bowenpally", "Gachibowli", "Nizampet", "Nallagandla",
    "Shamshabad", "Hitech City", "Bachupally", "Kukatpally", "Tellapur",
    "Attapur", "Ameerpet", "Kondapur", "Miyapur", "Pocharam", "Kompally",
    "Jubilee Hills", "LB Nagar", "Madhapur", "Nagole", "Begumpet",
    "Manikonda",
]

st.set_page_config(page_title="Hyderabad House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("🏠 Hyderabad House Price Predictor")
    st.write(
        "Estimate a property's price using a Linear Regression model trained on "
        "synthetic data across popular Hyderabad localities."
    )

    model = load_model()

    st.subheader("Enter property details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Locality / Area", sorted(AREAS))
        sqft = st.number_input("Built-up Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.selectbox("BHK (Bedrooms)", [1, 2, 3, 4, 5], index=1)
    with col2:
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
        age_years = st.slider("Property Age (years)", 0, 30, 5)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame(
            [{
                "sqft": sqft,
                "bhk": bhk,
                "bathrooms": bathrooms,
                "age_years": age_years,
                "area": area,
            }]
        )

        prediction_lakhs = model.predict(input_df)[0]
        prediction_lakhs = max(prediction_lakhs, 0)

        st.success(f"### Estimated Price: ₹ {prediction_lakhs:,.2f} Lakhs")
        st.caption(
            f"≈ ₹ {prediction_lakhs * 100000:,.0f} "
            f"(₹ {prediction_lakhs * 100000 / sqft:,.0f} per sqft)"
        )

    with st.expander("About this project"):
        st.markdown(
            """
            - **Model**: Linear Regression (Scikit-learn)
            - **Features**: locality, built-up area, BHK, bathrooms, property age
            - **Data**: synthetic dataset generated for demonstration
              (no real listings were used)
            - **Stack**: Python, Pandas, NumPy, Scikit-learn, Streamlit
            """
        )


if __name__ == "__main__":
    main()
