# -------------------------------
# 1. Import library
# -------------------------------
import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# 2. Load model
# -------------------------------
model = pickle.load(open("diabetesPredict.sav", "rb"))

# -------------------------------
# 3. Setup halaman dan judul
# -------------------------------
st.set_page_config(page_title="Diagnosa Diabetes", layout="centered")

st.title("Aplikasi Diagnosa Diabetes")
st.markdown("""
Aplikasi ini menggunakan Machine Learning (Logistic Regression) untuk memprediksi apakah seseorang terindikasi mengidap diabetes berdasarkan data medis yang dimasukkan.
""")

st.sidebar.header("Formulir Input Pasien")

# -------------------------------
# 4. Input user dari sidebar
# -------------------------------
pregnancies = st.sidebar.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glukosa", min_value=0.0, max_value=300.0, value=0.0)
blood_pressure = st.sidebar.number_input("Tekanan Darah", min_value=0.0, max_value=200.0, value=0.0)
skin_thickness = st.sidebar.number_input("Ketebalan Kulit", min_value=0.0, max_value=100.0, value=0.0)
insulin = st.sidebar.number_input("Insulin", min_value=0.0, max_value=1000.0, value=0.0)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
age = st.sidebar.number_input("Usia", min_value=1, max_value=120, value=1)

input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# -------------------------------
# 5. Prediksi & tampilan hasil
# -------------------------------
if st.sidebar.button("Cek Disini"):

    # Reshape input
    data_np = np.asarray(input_data).reshape(1, -1)
    
    # Prediksi
    prediksi = model.predict(data_np)

    st.subheader("Data yang Dimasukkan")
    df_input = pd.DataFrame([input_data], columns=[
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ])
    st.dataframe(df_input)

    # Hasil Diagnosa
    st.subheader("Hasil Pengecekan")
    if prediksi[0] == 1:
        st.error("Anda TERDETEKSI Diabetes")
    else:
        st.success("Anda TIDAK TERDETEKSI Diabetes")

    # -------------------------------
    # 6. Grafik Perbandingan
    # -------------------------------
    st.subheader("Perbandingan dengan Rata-rata Orang Normal")

    df_diabetes = pd.read_csv("diabetes.csv")

    # Hitung rata-rata data normal (Outcome == 0)
    avg_normal = df_diabetes[df_diabetes["Outcome"] == 0].drop(columns="Outcome").mean()

    # Buat DataFrame perbandingan
    df_compare = pd.DataFrame({
        "User Input": input_data,
        "Rata-rata Normal": avg_normal
    })

    # Plot diagram batang menggunakan matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    df_compare.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nilai")
    plt.title("Perbandingan Data Pasien dengan Rata-rata Orang Normal")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    # -------------------------------
    # 7. Simpan ke riwayat (opsional)
    # -------------------------------
    hasil = int(prediksi[0])
    df_input["Prediction"] = hasil

    if not os.path.exists("riwayat_diagnosa.csv"):
        df_input.to_csv("riwayat_diagnosa.csv", index=False)
    else:
        df_input.to_csv("riwayat_diagnosa.csv", mode='a', header=False, index=False)
