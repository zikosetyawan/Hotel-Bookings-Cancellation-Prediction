import streamlit as st
import base64
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="wide"
)

from eda import run_eda
from predict import run_predict

menu = ["Home Page", "EDA", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home Page":

    with open("./src/Hotel_image.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    html = f"""
    <div style="
        background: radial-gradient(circle at top, #0f172a, #020617);
        padding: 70px 50px;
        border-radius: 28px;
        box-shadow: 0 30px 60px rgba(0,0,0,0.6);
        max-width: 900px;
        margin: 60px auto;
        text-align: center;
        color: white;
        font-family: Inter, system-ui, -apple-system, sans-serif;
    ">

        <img src="data:image/png;base64,{img_base64}"
             style="width:500px; margin-bottom:30px;" />

        <h1 style="
            font-size:64px;
            font-weight:800;
            margin-bottom:18px;
            letter-spacing:-1px;
        ">
            Hotel Booking<br/>Cancellation Prediction
        </h1>

        <p style="
            font-size:24px;
            color:#94a3b8;
            margin-bottom:36px;
        ">
            Machine Learning untuk Memprediksi Risiko Pembatalan Reservasi Hotel
        </p>

        <div style="
            max-width:720px;
            margin: 0 auto 40px;
            text-align:left;
            font-size:18px;
            line-height:1.8;
            color:#e5e7eb;
        ">
            <p>
                Pembatalan reservasi hotel menciptakan ketidakpastian okupansi dan
                berpotensi menyebabkan <b>lost revenue</b>.
            </p>

            <p>
                Aplikasi ini menggunakan <b>Random Forest</b> untuk memprediksi
                apakah sebuah reservasi berisiko dibatalkan, sehingga hotel dapat:
            </p>

            <ul style="margin-left:20px;">
                <li>Mengurangi false negative pembatalan</li>
                <li>Mengoptimalkan strategi overbooking</li>
                <li>Mendukung keputusan revenue management</li>
            </ul>
        </div>

        <p style="
            font-size:14px;
            letter-spacing:2px;
            color:#38bdf8;
        ">
            GUNAKAN MENU DI SEBELAH KIRI UNTUK MEMULAI
        </p>

    </div>
    """

    components.html(html, height=3000)

elif choice == "EDA":
    run_eda()

elif choice == "Predict":
    run_predict()