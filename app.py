import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoodss.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Family size']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications']
numerical_features = ['Age', 'Family size']

# Membuat encoders dan scaler
label_encoders = {column: LabelEncoder().fit(data[column].astype(str)) for column in categorical_features}
scaler = StandardScaler().fit(data[numerical_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = pd.DataFrame([user_input])
    
    # Pra-pemrosesan data
    for column in categorical_features:
        if column in processed_input:
            if processed_input[column][0] in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform(processed_input[column])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = -1
    
    processed_input[numerical_features] = scaler.transform(processed_input[numerical_features])
    return processed_input

# CSS for styling
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
st.title("Prediksi Feedback Pelanggan Online Food")

st.markdown("""
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Input pengguna
age = st.number_input('Age', min_value=18, max_value=100, step=1)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed'])
educational_qualifications = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'])
family_size = st.number_input('Family size', min_value=1, max_value=20, step=1)

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
}

# Pemetaan angka ke label
label_mapping = {0: 'No', 1: 'Yes'}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        # Ganti angka dengan label yang sesuai
        prediction_label = label_mapping.get(prediction[0], 'Unknown')
        st.write(f'Prediction: {prediction_label}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
