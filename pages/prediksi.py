import os
import json
import numpy as np
import streamlit as st

st.set_page_config(page_title="Uji Prediksi Map PUBG Mobile", layout="centered")

# Memuat parameter model dari model.json
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.json")

with open(model_path, "r") as f:
    model_data = json.load(f)

fitur = model_data["fitur"]
kelas = model_data["kelas"]
prior = np.array(model_data["prior"])
theta = np.array(model_data["theta"])
var = np.array(model_data["var"])

# Preset sampel log telemetri pertandingan
SAMPEL_LOG = {
    "Log #1: Early Game (Menit ke-2)": {
        "deskripsi": "Pertempuran awal setelah mendarat, peringkat akhir masih acak dan jarak tembak dekat.",
        "killer_p": 42,
        "killer_xy": "545210, 582400",
        "victim_p": 89,
        "victim_xy": "543100, 580950",
        "waktu": 135,
    },
    "Log #2: Mid Game (Menit ke-12)": {
        "deskripsi": "Rotasi zona tengah pertandingan dengan koordinat mendekati pusat peta.",
        "killer_p": 14,
        "killer_xy": "402150, 388420",
        "victim_p": 27,
        "victim_xy": "405800, 391100",
        "waktu": 725,
    },
    "Log #3: Late Game (Menit ke-24)": {
        "deskripsi": "Duel zona akhir (Top 5) menuju Chicken Dinner dengan posisi sangat berdekatan.",
        "killer_p": 1,
        "killer_xy": "312450, 445120",
        "victim_p": 4,
        "victim_xy": "313200, 444890",
        "waktu": 1460,
    },
}

default_log = SAMPEL_LOG["Log #2: Mid Game (Menit ke-12)"]
if "killer_p" not in st.session_state:
    st.session_state["killer_p"] = default_log["killer_p"]
    st.session_state["killer_xy"] = default_log["killer_xy"]
    st.session_state["victim_p"] = default_log["victim_p"]
    st.session_state["victim_xy"] = default_log["victim_xy"]
    st.session_state["waktu"] = default_log["waktu"]
    st.session_state["info_log"] = f"**Log #2: Mid Game (Menit ke-12)** — {default_log['deskripsi']}"

def muat_sampel(nama_log):
    data = SAMPEL_LOG[nama_log]
    st.session_state["killer_p"] = data["killer_p"]
    st.session_state["killer_xy"] = data["killer_xy"]
    st.session_state["victim_p"] = data["victim_p"]
    st.session_state["victim_xy"] = data["victim_xy"]
    st.session_state["waktu"] = data["waktu"]
    st.session_state["info_log"] = f"**{nama_log}** — {data['deskripsi']}"

def parse_koordinat(teks, label):
    bagian = teks.replace(",", " ").split()
    if len(bagian) != 2:
        raise ValueError(f"Format **{label}** harus berisi dua angka (X, Y), contoh: `402150, 388420`.")
    return float(bagian[0]), float(bagian[1])

st.title("Prediksi Map PUBG Mobile")
st.write("Simulasi pembacaan data log telemetri eliminasi pemain menggunakan model Gaussian Naive Bayes.")

# Preset Log Pertandingan
st.subheader("Muat Sampel Log Pertandingan")
cols_preset = st.columns(3)
for idx, nama_log in enumerate(SAMPEL_LOG.keys()):
    with cols_preset[idx]:
        st.button(nama_log, on_click=muat_sampel, args=(nama_log,), use_container_width=True)

st.info(st.session_state["info_log"])

# Form Input Data Log
col1, col2 = st.columns(2)
with col1:
    killer_placement = st.number_input(
        "Killer # Berapa?",
        min_value=1,
        max_value=100,
        step=1,
        key="killer_p",
        help="Urutan finish si pembunuh di match ini (1 = Chicken Dinner)."
    )
with col2:
    killer_xy = st.text_input(
        "Titik Lokasi Pembunuh di Map (X, Y)",
        key="killer_xy",
        help="Koordinat posisi pembunuh dari log telemetri (skala 0 - 800.000)."
    )

col3, col4 = st.columns(2)
with col3:
    victim_placement = st.number_input(
        "Victim # Berapa?",
        min_value=1,
        max_value=100,
        step=1,
        key="victim_p",
        help="Urutan finish si korban saat tereliminasi (1 - 100)."
    )
with col4:
    victim_xy = st.text_input(
        "Titik Lokasi Korban di Map (X, Y)",
        key="victim_xy",
        help="Koordinat posisi korban dari log telemetri (skala 0 - 800.000)."
    )

waktu = st.number_input(
    "Waktu Kejadian Kill (Detik ke-...)",
    min_value=0,
    step=1,
    key="waktu",
    help="Detik ke-berapa kill tercatat sejak pertandingan dimulai."
)

if st.button("Prediksi Map", type="primary"):
    try:
        killer_x, killer_y = parse_koordinat(killer_xy, "Titik Lokasi Pembunuh di Map (X, Y)")
        victim_x, victim_y = parse_koordinat(victim_xy, "Titik Lokasi Korban di Map (X, Y)")

        nilai_fitur = {
            "killer_placement": float(killer_placement),
            "killer_position_x": killer_x,
            "killer_position_y": killer_y,
            "victim_placement": float(victim_placement),
            "victim_position_x": victim_x,
            "victim_position_y": victim_y,
            "time": float(waktu),
        }

        x = np.array([nilai_fitur[f] for f in fitur])

        log_probs = []
        for idx in range(len(kelas)):
            log_prior = np.log(prior[idx])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var[idx]) + ((x - theta[idx]) ** 2) / var[idx]
            )
            log_probs.append(log_prior + log_likelihood)

        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)

        hasil_kelas = kelas[np.argmax(probs)]
        st.success(f"Hasil Prediksi Map: **{hasil_kelas}**")

        for k, p in zip(kelas, probs):
            st.write(f"- Probabilitas **{k}**: {p * 100:.2f}%")

    except ValueError as e:
        st.error(f"Input tidak valid: {e}")