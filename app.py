import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the best model
model = joblib.load('best_model.pkl')

# Load data for encoding and scaling
data = pd.read_csv('onlinefoods.csv')

# Required columns for training
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 
                    'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Ensure only the required columns are present
data = data[required_columns]

# Data preprocessing
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'Monthly Income', 'Pin code', 'latitude', 'longitude']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Function to process user input
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Default value for unknown categories
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    # Ensure that all required columns are present
    for col in numeric_features:
        if col not in processed_input.columns:
            processed_input[col] = [0]  # or another appropriate default value
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

st.markdown("""
    <style>
    .main {
        background-color: #F4F4F4; /* Light Gray background */
    }
    h1 {
        color: #333333; /* Dark Gray color */
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #555555; /* Medium Gray color */
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        padding: 12px 25px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker Green on hover */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<h1>Prediksi Feedback Pelanggan Online Food</h1>', unsafe_allow_html=True)

# User inputs
age = st.number_input('Age', min_value=18, max_value=100, step=1)
gender = st.radio('Gender', ['Male', 'Female'])
marital_status = st.radio('Marital Status', ['Single', 'Married'])
occupation = st.radio('Occupation', ['Student', 'Employee', 'Self Employed'])
monthly_income = st.number_input('Monthly Income', min_value=0, step=1)
educational_qualifications = st.radio('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'])
family_size = st.number_input('Family size', min_value=1, max_value=20, step=1)
latitude = st.number_input('Latitude', format="%.6f")
longitude = st.number_input('Longitude', format="%.6f")
pin_code = st.number_input('Pin code', min_value=0, step=1)

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

# Map numbers to labels
label_mapping = {0: 'No', 1: 'Yes'}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        # Replace number with appropriate label
        prediction_label = label_mapping.get(prediction[0], 'Unknown')
        st.write(f'Prediction: {prediction_label}')
        
        # Additional information about the prediction
        st.markdown("""
            <div class="info-box">
                <div class="info-title">Informasi Tentang Hasil Prediksi:</div>
                <div class="info-content">
                    <p><strong>Yes:</strong> Pelanggan melakukan tindakan tertentu, seperti melakukan pembelian ulang, memberikan ulasan positif, atau merespons promosi.</p>
                    <p><strong>No:</strong> Pelanggan tidak melakukan tindakan tersebut, mungkin karena tidak puas dengan layanan, tidak tertarik, atau tidak menerima tawaran promosi.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
