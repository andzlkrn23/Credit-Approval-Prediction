import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("PREDIKSI APPROVAL CREDIT")

@st.cache_data
def fetch_data():
    df = pd.read_csv('df_train_test.csv')
    return df

df = fetch_data()


st.write('INFORMASI CALON NASABAH:')

# user input

Gender = st.radio(label='Jenis Kelamin:', options=['Male', 'Female'])
Married = st.radio(label='Apakah Sudah Menikah?:', options=['Yes', 'No'])
Dependents = st.radio(label='Berapa Banyak Tanggungan Anda?:', options=['0', '1', '2', '3+'])
Education = st.radio(label='Jenjang Pendidikan Anda:', options=['Graduate', 'Not Graduate'])
Self_Employed = st.radio(label='Apakah Anda Merupakan Wiraswasta?:', options=['Yes', 'No'])
ApplicantIncome = st.number_input(label='Berapa Pendapatan Bulanan Anda?:', min_value=0.00, max_value=999999.99, value=0.00, step=0.01)
CoapplicantIncome  = st.number_input(label='Berapa Pendapatan Penjamin Anda Per-Bulanan?:', min_value=0.00, max_value=999999.99, value=0.00, step=0.01)
LoanAmount = st.number_input(label='Berapa Rencana Pinjaman Anda?:', min_value=0.00, max_value=999999.99, value=0.00, step=0.01)
Loan_Amount_Term = st.number_input(label='Berapa Rencana Durasi Pembayaran Pinjaman Anda?:', min_value=0.00, max_value=999999.99, value=0.00, step=0.01)
Credit_History = st.radio(label='Apakah Sebelumnya Anda Memiliki Riwayat Meminjam? 0 (Tidak Ada Riwayat Kredit) dan 1 (Ada Riwayat Kredit):', options=[1., 0.])
Property_Area = st.radio(label='Dimana Tempat Tinggal Anda:', options=['Urban', 'Rural', 'Semiurban'])

#membuat df

data = {
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area],
}
input = pd.DataFrame(data, index=[0])


st.subheader('Data Calon Nasabah')
st.write(input)

load_model = joblib.load("model_nb_save")

st.write(load_model)
if st.button('Prediksi'):
    prediction = load_model.predict(input)

    if prediction == 1.0:
        prediction = 'Disetujui'
    elif prediction == 0.0:
        prediction = 'Ditolak'

    st.write('Berikut Merupakan Prediksi Pengajuan Kredit Anda : ')
    st.write(prediction)




    
