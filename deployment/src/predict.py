import streamlit as st
import pandas as pd
import pickle
import gdown
import os

# =========================
# CONFIG
# =========================
MODEL_ID = "1SGMp9ahIUQdN3LEI-0i0NKmgxyq_V-ZL"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "rf_baseline_pipeline.pkl"
DATA_PATH = "./src/hotel_bookings.csv"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# =========================
# LOAD DATA FOR OPTIONS
# =========================
@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)

    options = {
        "hotel": sorted(df["hotel"].unique()),
        "arrival_date_month": [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ],
        "meal": sorted(df["meal"].dropna().unique()),
        "market_segment": sorted(df["market_segment"].unique()),
        "distribution_channel": sorted(df["distribution_channel"].unique()),
        "reserved_room_type": sorted(df["reserved_room_type"].unique()),
        "assigned_room_type": sorted(df["assigned_room_type"].unique()),
        "deposit_type": sorted(df["deposit_type"].unique()),
        "customer_type": sorted(df["customer_type"].unique()),
        "country": sorted(df["country"].dropna().unique()),
        "agent": sorted(df["agent"].fillna(0).astype(int).unique()),
        "company": sorted(df["company"].fillna(0).astype(int).unique())
    }

    return options

model = load_model()
opts = load_reference_data()

# =========================
# STREAMLIT PAGE
# =========================
def run_predict():
    st.subheader("🔮 Prediksi Pembatalan Reservasi Hotel")

    st.markdown(
        """
        Masukkan detail reservasi berdasarkan **data historis hotel**
        untuk memprediksi **risiko pembatalan reservasi**.
        """
    )

    with st.form("prediction_form"):
        st.markdown("### 🏨 Informasi Reservasi")

        hotel = st.selectbox("Tipe Hotel", opts["hotel"])
        lead_time = st.number_input("Lead Time (hari)", 0, 500, 60)

        arrival_year = st.selectbox("Tahun Kedatangan", [2015, 2016, 2017])
        arrival_month = st.selectbox("Bulan Kedatangan", opts["arrival_date_month"])
        arrival_week = st.number_input("Minggu ke-", 1, 53, 25)
        arrival_day = st.number_input("Hari ke-", 1, 31, 15)

        st.markdown("### 👨‍👩‍👧 Detail Tamu")

        adults = st.number_input("Jumlah Dewasa", 1, 10, 2)
        children = st.number_input("Jumlah Anak", 0, 10, 0)
        babies = st.number_input("Jumlah Bayi", 0, 5, 0)

        weekend_nights = st.number_input("Malam Akhir Pekan", 0, 20, 1)
        weekday_nights = st.number_input("Malam Hari Kerja", 0, 20, 2)

        st.markdown("### 💰 Harga & Komitmen")

        adr = st.number_input("Average Daily Rate (ADR)", 0.0, 1000.0, 100.0)
        deposit_type = st.selectbox("Jenis Deposit", opts["deposit_type"])
        booking_changes = st.number_input("Booking Changes", 0, 20, 0)

        st.markdown("### 🌍 Informasi Tambahan")

        country = st.selectbox("Negara Asal", opts["country"])
        meal = st.selectbox("Meal", opts["meal"])
        market_segment = st.selectbox("Market Segment", opts["market_segment"])
        distribution_channel = st.selectbox("Distribution Channel", opts["distribution_channel"])
        customer_type = st.selectbox("Customer Type", opts["customer_type"])

        reserved_room = st.selectbox("Reserved Room Type", opts["reserved_room_type"])
        assigned_room = st.selectbox("Assigned Room Type", opts["assigned_room_type"])

        agent = st.selectbox("Agent", opts["agent"])
        company = st.selectbox("Company", opts["company"])

        special_requests = st.number_input("Special Requests", 0, 5, 0)
        parking = st.number_input("Car Parking Spaces", 0, 5, 0)

        submit = st.form_submit_button("🔍 Prediksi")

    if submit:
        input_df = pd.DataFrame([{
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_year,
            "arrival_date_month": arrival_month,
            "arrival_date_week_number": arrival_week,
            "arrival_date_day_of_month": arrival_day,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": weekday_nights,
            "adults": adults,
            "children": float(children),
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": reserved_room,
            "assigned_room_type": assigned_room,
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "agent": str(agent),
            "company": str(company),
            "days_in_waiting_list": 0,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": special_requests
        }])
        st.subheader("📁 Tamu Yang Akan Diprediksi")
        st.dataframe(input_df)

        pred_class = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0, 1]

        st.subheader("📊 Hasil Prediksi")

        st.dataframe(pd.DataFrame({
            "Prediksi": ["Cancel" if pred_class == 1 else "Tidak Cancel"],
            "Probabilitas Cancel": [round(pred_prob, 3)]
        }))

        if pred_prob >= 0.6:
            st.error("⚠️ Risiko pembatalan TINGGI")
        elif pred_prob >= 0.4:
            st.warning("⚠️ Risiko pembatalan SEDANG")
        else:
            st.success("✅ Risiko pembatalan RENDAH")


if __name__ == "__main__":
    run_predict()