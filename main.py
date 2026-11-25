import os
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PUBG Mobile Map Analytics", layout="wide")

# Memuat parameter model dari model.json
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.json")

with open(model_path, "r") as f:
    model_data = json.load(f)

kelas = model_data["kelas"]
prior = np.array(model_data["prior"], dtype=float)
theta = np.array(model_data["theta"], dtype=float)
std = np.sqrt(np.array(model_data.get("var", np.zeros_like(theta)), dtype=float))

# Identifikasi map paling ramai dan paling sepi dari nilai prior asli
idx_ramai = int(np.argmax(prior))
idx_sepi = int(np.argmin(prior))
map_ramai = kelas[idx_ramai]
map_sepi = kelas[idx_sepi]
persen_ramai = prior[idx_ramai] * 100
persen_sepi = prior[idx_sepi] * 100
rasio = prior[idx_ramai] / prior[idx_sepi]

# Radius sebaran pertempuran (km) dari standar deviasi koordinat (100.000 unit = 1 km)
radius_km = [np.hypot(std[i][1], std[i][2]) / 100000 for i in range(len(kelas))]
idx_luas = int(np.argmax(radius_km))

st.title("Analisis Map PUBG Untuk Developer")
st.caption(
    f"Analisis distribusi pemain dan rekomendasi keputusan menggunakan model Gaussian Naive Bayes "
    f"(Sensitivitas Deteksi Map Minoritas: {model_data.get('recall_miramar', 0) * 100:.1f}%)."
)

# 1. Ringkasan Metrik Utama
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Map Orang Jago (Paling Rame)",
        value=f"{map_ramai} ({persen_ramai:.2f}%)",
        delta="Prioritas Optimasi Performa",
        delta_color="normal"
    )
with col2:
    st.metric(
        label="Map Bot (Paling Sepi)",
        value=f"{map_sepi} ({persen_sepi:.2f}%)",
        delta="Prioritas Event & Insentif",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Ketimpangan Trafik",
        value=f"{rasio:.1f}x Lipat",
        delta=f"Selisih {persen_ramai - persen_sepi:.2f}%",
        delta_color="off"
    )
with col4:
    st.metric(
        label="Sebaran Zona Tempur Terluas",
        value=f"{kelas[idx_luas]} (±{radius_km[idx_luas]:.2f} km)",
        delta="Berdasarkan Sebaran Koordinat",
        delta_color="off"
    )

st.divider()

# 2. Grafik Distribusi Pemain & Rekomendasi Aksi
col_grafik, col_aksi = st.columns([1.1, 1.3])

with col_grafik:
    st.subheader("Distribusi Aktivitas Pemain per Map")
    df_distribusi = pd.DataFrame({
        "Map": kelas,
        "Persentase Pemain (%)": np.round(prior * 100, 2)
    }).set_index("Map")
    st.bar_chart(df_distribusi, height=320)

with col_aksi:
    st.subheader("Saran Buat Developer")

    st.warning(
        f"**1. Optimasi Teknis & Server di {map_ramai} ({persen_ramai:.2f}%)**\n"
        f"- **Alokasi Server:** Alokasikan ±{persen_ramai:.0f}% kapasitas *matchmaking* dan *tick-rate* server khusus {map_ramai} karena menampung mayoritas trafik pemain.\n"
        f"- **Optimasi Performa Klien:** Fokuskan perbaikan *frame drop* (FPS) dan *rendering* objek di area pusat peta (`X ≈ {theta[idx_ramai][1]:,.0f}`, `Y ≈ {theta[idx_ramai][2]:,.0f}`) dengan radius kepadatan `±{radius_km[idx_ramai]:.2f} km`."
    )

    waktu_sepi_menit = theta[idx_sepi][6] / 60
    saran_rotasi = (
        "Tingkatkan *spawn rate* kendaraan dan percepat fase awal *Bluezone* agar rotasi pemain lebih padat."
        if radius_km[idx_sepi] >= radius_km[idx_ramai]
        else "Pertahankan distribusi *loot* di area pinggiran peta agar tempo tetap seimbang."
    )
    st.info(
        f"**2. Strategi Event & Boosting di {map_sepi} ({persen_sepi:.2f}%)**\n"
        f"- **Event Khusus Map:** Tambahkan misi harian eksklusif {map_sepi} dengan hadiah *Double BP/XP* atau *skin* terbatas untuk memangkas selisih trafik sebesar {persen_ramai - persen_sepi:.2f}%.\n"
        f"- **Penyesuaian Tempo Gameplay:** Rata-rata eliminasi di {map_sepi} terjadi pada menit ke-{waktu_sepi_menit:.1f} ({theta[idx_sepi][6]:.0f} detik) dengan sebaran pertempuran `±{radius_km[idx_sepi]:.2f} km`. {saran_rotasi}"
    )

st.divider()

# 3. Tabel Perbandingan Karakteristik Pertempuran
st.subheader("Perbandingan Pola Pertempuran Antar Map (Rata-rata Dataset)")
df_pola = pd.DataFrame({
    "Indikator Pertempuran": [
        "Proporsi Trafik Pertandingan (%)",
        "Kebanyakan Orang Jago Bertahan Sampe Diposisi (Alive)",
        "Kebanyakan Orang Cupu Mati Ada Diposisi (Alive)",
        "Pusat Keramaian Koordinat X",
        "Pusat Keramaian Koordinat Y",
        "Radius Sebaran Zona Tempur",
        "Rata-rata Waktu Terjadi Kill (Detik)"
    ],
    kelas[0]: [
        f"{prior[0]*100:.2f}%",
        f"{theta[0][0]:.0f} (±{std[0][0]:.0f})",
        f"{theta[0][3]:.0f} (±{std[0][3]:.0f})",
        f"{theta[0][1]:,.0f} (±{std[0][1]:,.0f})",
        f"{theta[0][2]:,.0f} (±{std[0][2]:,.0f})",
        f"±{radius_km[0]:.2f} km",
        f"{theta[0][6]:.0f} detik ({theta[0][6]/60:.1f} menit)"
    ],
    kelas[1]: [
        f"{prior[1]*100:.2f}%",
        f"{theta[1][0]:.0f} (±{std[1][0]:.0f})",
        f"{theta[1][3]:.0f} (±{std[1][3]:.0f})",
        f"{theta[1][1]:,.0f} (±{std[1][1]:,.0f})",
        f"{theta[1][2]:,.0f} (±{std[1][2]:,.0f})",
        f"±{radius_km[1]:.2f} km",
        f"{theta[1][6]:.0f} detik ({theta[1][6]/60:.1f} menit)"
    ]
})
st.dataframe(df_pola, use_container_width=True, hide_index=True)