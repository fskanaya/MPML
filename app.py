import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation','Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
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

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5; /* Light Gray background */
    }
    h1 {
        color: #333333; /* Dark Gray color */
        text-align: center;
        margin-bottom: 25px;
    }
    h3 {
        color: #333333; /* Dark Gray color */
    }
    .stButton>button {
        background-color: #4B4B4B; /* Dark Gray button */
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #2C2C2C; /* Darker Gray on hover */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<h1>Prediksi Feedback Pelanggan Online Food</h1>', unsafe_allow_html=True)

st.markdown("""
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Input pengguna
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed'])
monthly_income = st.number_input('Monthly Income', min_value=0)
educational_qualifications = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'])
family_size = st.number_input('Family size', min_value=1, max_value=20)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')
pin_code = st.number_input('Pin code', min_value=0)

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

# Pemetaan angka ke label
label_mapping = {0: 'No', 1: 'Yes'}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        # Ganti angka dengan label yang sesuai
        prediction_label = label_mapping.get(prediction[0], 'Unknown')
        st.write(f'Prediction: {prediction_label}')
        
        # Informasi tambahan tentang prediksi
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
