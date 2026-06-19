"""
views/about.py
About Page — Project context, datasets, algorithms, references and group contributors.
"""

import streamlit as st

def show_about():
    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hdr">
        <div>
            <h1>About the Project</h1>
            <p>Customer Segmentation in Digital Marketing using Q-Learning Differential Evolution integrated with K-Means.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Info ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="sl">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        Dashboard ini adalah hasil akhir dari **ML Final Project (Kelompok 6)**. Project ini mengimplementasikan segmentasi pelanggan secara hybrid pada data retail digital:
        
        1. **Data Preprocessing & Noise Isolation**: Melakukan pembersihan data, ekstraksi fitur RFM++, dan memisahkan pelanggan anomali (outliers) menggunakan algoritma **DBSCAN**.
        2. **Unsupervised Clustering**: Melakukan pengelompokan pelanggan menggunakan **K-Means Clustering** yang dioptimasi dengan algoritma metaheuristik **Q-Learning based Differential Evolution (QLDE)** untuk mencegah centroid terjebak dalam local optima.
        3. **Supervised Classifier**: Melatih pengklasifikasi (**Random Forest**) menggunakan label segmen optimal sebagai target klasifikasi untuk prediksi segmen secara real-time.
        """)
        
        st.markdown('<div class="sl">Dataset Reference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Dataset**: UCI Online Retail Dataset.
        - **Deskripsi**: Data transaksi ritel transnasional yang berisi semua transaksi yang terjadi antara tanggal 01/12/2010 dan 09/12/2011 untuk ritel online non-toko yang terdaftar di Inggris.
        - **Engineered Features**:
            - **Recency**: Jumlah hari sejak transaksi terakhir pelanggan.
            - **Frequency**: Total jumlah transaksi yang berhasil dilakukan.
            - **Monetary**: Total nilai pengeluaran belanja pelanggan.
            - **AvgSpending**: Rata-rata nilai belanja per barang.
            - **UniqueProducts**: Jumlah produk unik yang dibeli.
            - **CancelFrequency**: Total transaksi pembatalan (return/refund).
            - **AvgMonthlySpending**: Rata-rata belanja bulanan pelanggan.
        """)

    with col_right:
        st.markdown('<div class="sl">Project Team (Kelompok 6)</div>', unsafe_allow_html=True)
        
        team_members = [
            ("Anggota 1", "Data Engineer", "Bertanggung jawab pada data pipeline, Feature Engineering (RFM++), dan Outlier Detection."),
            ("Anggota 2", "Unsupervised ML Specialist", "Mengimplementasikan optimasi centroid K-Means menggunakan Q-Learning based DE (QLDE)."),
            ("Anggota 3", "Supervised ML Specialist", "Membangun model klasifikasi supervised untuk prediksi segmen real-time."),
        ]
        
        for name, role, desc in team_members:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 12px;">
                <div style="font-weight: 900; color: var(--color-paper); font-size: 16px; font-family: var(--font-raveo-variable);">{name}</div>
                <div style="font-size: 9px; text-transform: uppercase; letter-spacing: 0.02em; color: var(--color-steel); margin-top: 4px; font-weight: 500; font-family: var(--font-geist-mono);">{role}</div>
                <div style="font-size: 14px; color: var(--color-steel); margin-top: 8px; line-height: 1.6; letter-spacing: -0.14px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sl">Academic Reference</div>', unsafe_allow_html=True)
        st.markdown("""
        Implementasi optimasi centroid metaheuristik pada project ini mengacu pada jurnal:
        
        *“Customer segmentation in the digital marketing using a Q-learning based differential evolution algorithm integrated with K-means clustering”* (PLOS One, 2025).
        """)

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:var(--color-ash); font-size:12px; font-family:var(--font-geist-mono); letter-spacing:0.02em;'>"
        "KELOMPOK 6 · ML FINAL PROJECT · CUSTOMER SEGMENTATION IN DIGITAL MARKETING"
        "</div>",
        unsafe_allow_html=True,
    )
