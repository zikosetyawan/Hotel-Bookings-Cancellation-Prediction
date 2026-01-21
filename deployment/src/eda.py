import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def run_eda():
    st.subheader("📊 Exploratory Data Analysis")
    st.markdown(
    """
    Halaman ini menampilkan ringkasan eksplorasi data untuk memahami
    pola pembatalan reservasi hotel dan faktor-faktor yang memengaruhinya.
    """)

    @st.cache_data
    def load_data():
        return pd.read_csv("./src/hotel_bookings.csv")
    
    df = load_data()

    tab1, tab2 = st.tabs(["📁 Dataset", "📈 Data Explorer"])
    with tab1:
        st.markdown("### Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Cancellation Rate", f"{df['is_canceled'].mean()*100:.2f}%")

        st.markdown("### Sample Data")
        st.dataframe(df.head())

        st.markdown("### Data Types")
        st.dataframe(df.dtypes.reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        ))
    
    with tab2:
        eda = df.copy()
        st.markdown("## 1️⃣ Distribusi Pembatalan Reservasi")
        count_data = eda['is_canceled'].value_counts()
        prop_data = eda['is_canceled'].value_counts(normalize=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(
            x=count_data.index.astype(str),
            y=count_data.values,
            palette='pastel',
            ax=axes[0]
        )
        axes[0].set_title("Jumlah Reservasi")
        axes[0].set_xlabel("Status Pembatalan")
        axes[0].set_ylabel("Jumlah")

        sns.barplot(
            x=prop_data.index.astype(str),
            y=prop_data.values,
            palette='pastel',
            ax=axes[1]
        )
        axes[1].set_title("Proporsi Reservasi")
        axes[1].set_xlabel("Status Pembatalan")
        axes[1].set_ylabel("Proporsi")

        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Sekitar **28% reservasi dibatalkan**, sementara **72% tidak dibatalkan**.
            Hal ini menunjukkan bahwa dataset memiliki **ketidakseimbangan kelas**,
            sehingga metrik evaluasi seperti **recall** lebih relevan dibandingkan accuracy.
            """)
        
        st.markdown("## 2️⃣ Lead Time dan Risiko Pembatalan")

        fig = plt.figure(figsize=(12,5))
        sns.histplot(
            data=eda,
            x="lead_time",
            hue="is_canceled",
            bins=30,
            element="step",
            stat="density"
        )
        plt.title("Distribusi Lead Time berdasarkan Status Pembatalan")
        plt.xlabel("Lead Time (hari)")
        plt.ylabel("Density")
        plt.xlim(0, 400)
        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Reservasi dengan **lead time yang lebih panjang** cenderung
            memiliki **risiko pembatalan lebih tinggi**.
            
            Hal ini masuk akal secara bisnis karena semakin jauh jarak
            waktu pemesanan dengan tanggal kedatangan, semakin besar
            kemungkinan perubahan rencana tamu.
            """)

        st.markdown("## 3️⃣ Harga (ADR) dan Pembatalan")

        fig = plt.figure(figsize=(12,4))
        sns.histplot(
            data=eda,
            x="adr",
            hue="is_canceled",
            kde=True,
            element="step"
        )
        plt.title("Distribusi ADR berdasarkan Status Pembatalan")
        plt.xlabel("Average Daily Rate (ADR)")
        plt.xlim(0, 500)
        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Distribusi ADR antara reservasi batal dan tidak batal
            **saling tumpang tindih cukup besar**.
            
            Ini menunjukkan bahwa **harga bukan satu-satunya faktor utama**
            dalam menentukan pembatalan, meskipun terdapat indikasi bahwa
            reservasi dengan ADR lebih rendah sedikit lebih rentan dibatalkan.
            """)

        st.markdown("## 4️⃣ Komitmen Reservasi (Deposit Type)")

        fig = plt.figure(figsize=(10,4))
        sns.barplot(
            data=eda,
            x="deposit_type",
            y="is_canceled",
            palette="Set3"
        )
        plt.title("Tingkat Pembatalan berdasarkan Jenis Deposit")
        plt.ylabel("Cancellation Rate")
        plt.xlabel("Deposit Type")
        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Reservasi dengan **Refundable** memiliki tingkat pembatalan paling rendah, diikuti oleh **No Deposit**.
            Sementara itu, reservasi dengan **Non Refund** menunjukkan tingkat pembatalan 
            yang sangat tinggi. Namun, temuan ini kemungkinan dipengaruhi oleh
            **mekanisme pencatatan data**, di mana reservasi non-refundable yang dibatalkan
            tetap dicatat sebagai *canceled*, meskipun secara kebijakan tidak dapat dikembalikan.  
            Oleh karena itu, temuan ini menguatkan bahwa **komitmen finansial berperan penting
            dalam menurunkan risiko pembatalan**, tetapi interpretasi pada kategori *Non Refund* perlu dilakukan dengan hati-hati.
            """)
        
        st.markdown("## 5️⃣ Tipe Hotel dan Tingkat Pembatalan")

        hotel_order = (
            eda.groupby("hotel")["is_canceled"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.countplot(
            data=eda,
            x="hotel",
            hue="is_canceled",
            order=hotel_order,
            palette="viridis",
            ax=axes[0]
        )
        axes[0].set_title("Jumlah Reservasi per Tipe Hotel")
        axes[0].tick_params(axis="x", rotation=45)

        sns.barplot(
            data=eda,
            x="hotel",
            y="is_canceled",
            order=hotel_order,
            palette="viridis",
            ax=axes[1]
        )
        axes[1].set_title("Tingkat Pembatalan per Tipe Hotel")
        axes[1].set_ylim(0, 1)

        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            **City Hotel** memiliki jumlah reservasi lebih tinggi sekaligus
            **tingkat pembatalan yang lebih besar** dibandingkan Resort Hotel.
            
            Hal ini mengindikasikan bahwa City Hotel menghadapi risiko
            pembatalan yang lebih tinggi dan memerlukan strategi mitigasi khusus.
            """)

        st.markdown("## 6️⃣ Pola Musiman Pembatalan")

        month_order = [
            'January','February','March','April','May','June',
            'July','August','September','October','November','December'
        ]

        monthly = (
            eda.groupby('arrival_date_month')['is_canceled']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'total_booking', 'mean': 'cancel_rate'})
            .reindex(month_order)
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.countplot(
            data=eda,
            x='arrival_date_month',
            hue='is_canceled',
            order=month_order,
            palette='pastel',
            ax=axes[0]
        )
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_title("Jumlah Reservasi per Bulan")

        sns.barplot(
            data=monthly.reset_index(),
            x='arrival_date_month',
            y='cancel_rate',
            order=month_order,
            palette='pastel',
            ax=axes[1]
        )
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_title("Tingkat Pembatalan per Bulan")

        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Volume reservasi tertinggi terjadi antara **April–Agustus**, yang juga
            diikuti dengan peningkatan tingkat pembatalan.
            
            Bulan **Desember** memiliki volume reservasi rendah namun tingkat
            pembatalan relatif tinggi, menunjukkan pola perilaku yang berbeda.
            """)
            
        st.markdown("## 7️⃣ Negara Asal Tamu (Top 10)")

        top_countries = eda['country'].value_counts().head(10).index

        country_stats = (
            eda[eda['country'].isin(top_countries)]
            .groupby('country')['is_canceled']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'total_booking', 'mean': 'cancel_rate'})
            .loc[top_countries]
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(
            x=country_stats.index,
            y=country_stats['total_booking'],
            palette='Set3',
            ax=axes[0]
        )
        axes[0].set_title("Jumlah Reservasi")

        sns.barplot(
            x=country_stats.index,
            y=country_stats['cancel_rate'],
            palette='Set3',
            ax=axes[1]
        )
        axes[1].set_title("Tingkat Pembatalan")

        st.pyplot(fig)

        with st.expander("📌 Insight"):
            st.markdown("""
            Portugal (PRT) mendominasi jumlah reservasi sekaligus memiliki
            tingkat pembatalan yang relatif tinggi.
            
            Beberapa negara dengan volume kecil seperti **ITA** dan **BRA**
            menunjukkan tingkat pembatalan yang tinggi, sementara **GBR**
            memiliki volume tinggi dengan pembatalan yang relatif rendah.
            """)
        
        st.markdown("---")
        st.info(
            "EDA ini bertujuan memberikan pemahaman ringkas dan relevan "
            "terhadap faktor-faktor utama pembatalan reservasi hotel."
        )      


if __name__=='__main__':
    run_eda()