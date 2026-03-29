import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")

st.title("Wellness Tourism Package - Purchase Predictor")
st.write("Enter the customer details below to predict if they will purchase the new Wellness Tourism Package.")

# Load the saved model from the Hugging Face model hub
@st.cache_resource
def load_model():
    # Replace with your actual Hugging Face username
    repo_id = "jimmyabhinav28/Wellness-Tourism-XGBoost-Model" 
    model_path = hf_hub_download(repo_id=repo_id, filename="best_xgboost_model.joblib", repo_type="model")
    return joblib.load(model_path)

model = load_model()

# Get inputs and save them into a dataframe
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    monthly_income = st.number_input("Monthly Income", min_value=10000.0, value=25000.0)

with col2:
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=3)
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=1)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    own_car = st.selectbox("Owns Car?", [1, 0])
    passport = st.selectbox("Has Passport?", [1, 0])

with col3:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, value=15.0)
    num_followups = st.number_input("Number of Followups", min_value=0.0, value=3.0)
    num_trips = st.number_input("Number of Trips", min_value=1.0, value=2.0)

# Create DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase Likelihood", type="primary"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"The customer is **LIKELY** to purchase the package! (Confidence: {probabilities[1]:.1%})")
    else:
        st.error(f"The customer is **UNLIKELY** to purchase the package. (Confidence: {probabilities[0]:.1%})")
