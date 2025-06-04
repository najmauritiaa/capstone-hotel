import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.express as px
st.set_page_config(layout="wide")
import base64
import textwrap

# ---------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('indonesia_hotels.csv')
    df = df.dropna()
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# === Gambar lokal (misal: 'drone/DJI_0330.JPG') ===
with open("assets/header.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()


tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "Cari Berdasarkan Mood & Budget", "Cari Berdasarkan Peta", "Tentang Kami"])
with tab1:
    st.markdown(f"""
<style>
.hero-container {{
    position: relative;
    width: 100%;
    max-width: 100%;
    height: 450px;
    margin: 0 auto;
    border-radius: 30px;
    overflow: hidden;
}}
.hero-background {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 30px;
    display: block;
}}
.overlay {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0,0,0,0.6);
    border-radius: 30px;
    z-index: 1;
}}
.hero-content {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
    padding: 0 20px;
    z-index: 2;
}}
.hero-content h1 {{
    font-size: 3rem;
    font-weight: bold;
    white-space: pre-line;
}}
.hero-content h2 {{
    font-size: 2rem;
    color: #BDB395;
    font-weight: 600;
}}
.hero-content p {{
    font-size: 1.1rem;
    max-width: 700px;
    margin-top: 10px;
}}
.stats {{
    display: flex;
    gap: 50px;
    margin-top: 30px;
}}
.stat {{
    text-align: center;
}}
.stat-value {{
    font-size: 1.8rem;
    font-weight: bold;
}}
.stat-label {{
    font-size: 0.9rem;
}}
</style>

<div class="hero-container">
    <img src="data:image/jpeg;base64,{encoded_image}" class="hero-background"/>
    <div class="overlay"></div>
    <div class="hero-content">
        <h1>Temukan Hotel Idaman Kamu!</h1>
        <h2>Hotel Hunt</h2>
        <p>Cari hotel kini nggak ribet! Cukup pilih mood kamu hari ini, tentukan budget, dan klik lokasi favorit di peta—Hotel Hunt siap kasih rekomendasi hotel paling cocok buat kamu. Aplikasi pintar buat staycation, healing, atau traveling bareng bestie. Yuk mulai petualanganmu sekarang!</p>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">2</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat">
                <div class="stat-value">1036</div>
                <div class="stat-label">Hotels</div>
            </div>
            <div class="stat">
                <div class="stat-value">5</div>
                <div class="stat-label">Provinsi</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    # Layout tambahan di bawah
    # Fungsi untuk encode gambar lokal menjadi base64
    # Tampilkan judul utama
    st.write("---")
    st.markdown(
    """
    <div style='text-align: center; margin-top: 30px;'>
        <p style='font-size: 24px; font-weight: bold;'>Sesuaikan Pilihan Hotel Dengan Suasana Hati Kamu</p>
    </div>
    """,
    unsafe_allow_html=True
    )
    def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

    # Data kartu
    cards = [
        {"title": "Meditation", "description": "Temukan ketenangan dan fokus dengan sesi meditasi harian.", "image": "assets/me_time.jpg"},
        {"title": "Sociolized", "description": "Nikmati waktu bersama teman dan aktivitas sosial yang menyenangkan.", "image": "assets/date.jpg"},
        {"title": "Me Time", "description": "Waktu untuk diri sendiri, recharge dengan aktivitas favoritmu.", "image": "assets/me_time.jpg"},
        {"title": "Adventure", "description": "Cari petualangan seru di alam terbuka atau tempat baru.", "image": "assets/adventure.jpg"},
        {"title": "Date", "description": "Rencanakan momen romantis bersama pasangan spesialmu.", "image": "assets/date.jpg"},
        {"title": "Sport", "description": "Tingkatkan energi dan kesehatan dengan aktivitas olahraga.", "image": "assets/sport.jpg"},
    ]

    # CSS styling
    st.markdown("""
        <style>
            .card-row-scroll {
                display: flex;
                flex-direction: row;
                gap: 20px;
                margin-top: 20px;
                overflow-x: auto;
                padding-bottom: 20px;
                scrollbar-width: thin;
            }
            .card {
                flex: 0 0 auto;
                width: 250px;
                border: 1px solid #ddd;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background-color: white;
                transition: transform 0.2s;
            }
            .card:hover {
                transform: scale(1.02);
            }
            .card img {
                width: 100%;
                height: 250px;
                object-fit: cover;
            }
            .card-content {
                padding: 15px;
            }
            .card-title {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            }
            .card-description {
                font-size: 14px;
                color: #444;
            }
        </style>
    """, unsafe_allow_html=True)

    # Render cards: 1 baris horizontal scroll
    card_html = '<div class="card-row-scroll">'
    for card in cards:
        try:
            img_base64 = image_to_base64(card["image"])
            card_html += (
                f'<div class="card">'
                f'<img src="data:image/jpeg;base64,{img_base64}" alt="{card["title"]}">'
                f'<div class="card-content">'
                f'<div class="card-title">{card["title"]}</div>'
                f'<div class="card-description">{card["description"]}</div>'
                f'</div>'
                f'</div>'
            )
        except FileNotFoundError:
            card_html += (
                f'<div class="card">'
                f'<div class="card-content">'
                f'<div class="card-title">{card["title"]}</div>'
                f'<div class="card-description">❌ Gambar \'{card["image"]}\' tidak ditemukan.</div>'
                f'</div>'
                f'</div>'
            )
    card_html += '</div>'
    st.markdown(card_html, unsafe_allow_html=True)
    st.write("---")
    st.markdown(
    """
    <div style='text-align: center; margin-top: 30px;'>
        <p style='font-size: 24px; font-weight: bold;'>Sesuaikan Pilihan Hotel Dengan Lokasi Impianmu!</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    # PIE CHART JUMLAH HOTEL PER PROVINSI
    # Pastikan df sudah di-load dan punya kolom 'Provinsi'
    provinsi_counts = df['Provinsi'].value_counts().reset_index()
    provinsi_counts.columns = ['Provinsi', 'Jumlah Hotel']

    fig = px.pie(
        provinsi_counts,
        names='Provinsi',
        values='Jumlah Hotel',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)

    # ...lanjutkan dengan slider card lokasi atau konten lain...
with tab2:
    st.write("Silahkan ikuti langkah-langkah di bawah ini untuk menemukan hotel terbaik!")
    # ---------------------- LOAD DATASET -----------------------
    # Load dataset
    @st.cache_data
    def load_data():
        df = pd.read_csv('indonesia_hotels.csv')
        df = df.dropna()

        # Tukar Min dan Max jika Min > Max
        df[['Min', 'Max']] = df.apply(
            lambda row: pd.Series([row['Max'], row['Min']]) if row['Min'] > row['Max'] else pd.Series([row['Min'], row['Max']]),
            axis=1
        )

        def parse_fasilitas(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return [val]
            elif isinstance(val, list):
                return val
            else:
                return []

        df['list_fasilitas'] = df['list_fasilitas'].apply(parse_fasilitas)
        return df

    indonesia_hotels = load_data()

    # ================= UI =================
    st.subheader("Langkah 1: Jawab pertanyaan berikut ini")

    questions = [
        "Apa yang kamu rasakan saat ini?",
        "Apakah kamu merasa kesepian hari ini?",
        "Apakah kamu merasa stres atau kewalahan akhir-akhir ini?",
        "Apakah kamu ingin berbicara dengan seseorang saat ini?",
        "Apakah kamu merasa bosan dengan rutinitasmu?",
        "Apakah kamu ingin mencoba hal baru hari ini?",
        "Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?",
        "Apakah kamu sedang merindukan seseorang secara romantis?",
        "Apakah kamu ingin menyendiri dan menjauh dari keramaian?",
        "Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?",
        "Apakah kamu merasa senang ketika dikelilingi banyak orang?",
        "Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?",
        "Apakah kamu merasa butuh memahami dirimu lebih dalam?",
        "Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?",
        "Kalau bisa memilih aktivitas sekarang, kamu ingin:"
    ]

    answer_options = {
        "Apa yang kamu rasakan saat ini?": ["senang", "cemas", "sedih"],
        "Kalau bisa memilih aktivitas sekarang, kamu ingin:": ["tidur", "menyendiri", "pacaran", "jalan-jalan"]
    }

    answers = {}
    for q in questions:
        opts = answer_options.get(q, ["ya", "tidak"])
        answers[q] = st.selectbox(q, opts)

    st.subheader("Langkah 2: Masukkan Budget")
    budget_min = st.number_input("Budget Minimum (Rp)", min_value=0, value=300000, step=50000)
    budget_max = st.number_input("Budget Maksimum (Rp)", min_value=0, value=800000, step=50000)

    if st.button("🎯 Cari Rekomendasi Hotel"):

        rules = {
            "Meditation": [
                ("Apa yang kamu rasakan saat ini?", ["cemas", "sedih"]),
                ("Apakah kamu merasa stres atau kewalahan akhir-akhir ini?", ["ya"]),
                ("Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?", ["ya"]),
                ("Apakah kamu merasa butuh memahami dirimu lebih dalam?", ["ya"])
            ],
            "Me-time": [
                ("Apakah kamu ingin menyendiri dan menjauh dari keramaian?", ["ya"]),
                ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["menyendiri", "tidur"])
            ],
            "Socialized": [
                ("Apakah kamu merasa kesepian hari ini?", ["ya"]),
                ("Apakah kamu ingin berbicara dengan seseorang saat ini?", ["ya"]),
                ("Apakah kamu merasa senang ketika dikelilingi banyak orang?", ["ya"])
            ],
            "Date": [
                ("Apakah kamu sedang merindukan seseorang secara romantis?", ["ya"]),
                ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["pacaran"])
            ],
            "Adventure": [
                ("Apakah kamu merasa bosan dengan rutinitasmu?", ["ya"]),
                ("Apakah kamu ingin mencoba hal baru hari ini?", ["ya"]),
                ("Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?", ["ya"]),
                ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["jalan-jalan"])
            ],
            "Support": [
                ("Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?", ["ya"]),
                ("Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?", ["tidak"])
            ]
        }

        scores = {}
        for need, condition_list in rules.items():
            score = 0
            for q, valid_answers in condition_list:
                if answers.get(q) in valid_answers:
                    score += 1
            scores[need] = score

        dominant_need = max(scores, key=scores.get)

        st.success(f"✅ Kebutuhan Emosionalmu Saat Ini: {dominant_need}")

        need_to_facilities = {
            "Meditation": ['spa', 'pijat', 'yoga', 'perpustakaan', 'taman', 'gazebo', 'atmosfer tenang'],
            "Adventure": ['bersepeda', 'mendaki', 'memancing', 'kayak', 'menyelam', 'snorkeling', 'berkuda'],
            "Date": ['bar', 'private', 'romantis', 'restoran', 'teras', 'laut'],
            "Socialized": ['klub', 'karaoke', 'bar', 'game', 'anak', 'umum'],
            "Me-time": ['perpustakaan', 'spa', 'yoga', 'tv', 'teras', 'tenang'],
            "Support": ['layanan', 'resepsionis', 'laundry', 'keamanan', 'apotek', 'pelayanan']
        }

        needed_keywords = need_to_facilities.get(dominant_need, [])

        def count_matching_facilities(hotel_facilities, needed_keywords):
            return sum(any(kw.lower() in f.lower() for f in hotel_facilities) for kw in needed_keywords)

        filtered_hotels = indonesia_hotels[
            (indonesia_hotels['Min'] >= budget_min) & (indonesia_hotels['Max'] <= budget_max)
        ].copy()

        if not filtered_hotels.empty:
            filtered_hotels['matching_score'] = filtered_hotels['list_fasilitas'].apply(
                lambda x: count_matching_facilities(x, needed_keywords)
            )
            top_5 = filtered_hotels.sort_values(by='matching_score', ascending=False).head(5)
            st.subheader("🏨 Rekomendasi Hotel Berdasarkan Emosi & Budget (Top 5)")
            st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'matching_score']])
        else:
            st.warning("⚠ Tidak ditemukan hotel dalam budget tersebut. Menampilkan tanpa filter budget.")
            indonesia_hotels['matching_score'] = indonesia_hotels['list_fasilitas'].apply(
                lambda x: count_matching_facilities(x, needed_keywords)
            )
            top_5 = indonesia_hotels.sort_values(by='matching_score', ascending=False).head(5)
            st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'matching_score']])

        # ============ Content-based Filtering ============
        indonesia_hotels['fasilitas_str'] = indonesia_hotels['list_fasilitas'].apply(lambda x: ' '.join(map(str, x)))
        vectorizer = CountVectorizer()
        facility_matrix = vectorizer.fit_transform(indonesia_hotels['fasilitas_str'])
        facility_sim = cosine_similarity(facility_matrix)

        indonesia_hotels['avg_price'] = (indonesia_hotels['Min'] + indonesia_hotels['Max']) / 2
        scaler = MinMaxScaler()
        normalized_price = scaler.fit_transform(indonesia_hotels[['avg_price']])
        price_diff_matrix = np.abs(normalized_price - normalized_price.T)
        price_sim = 1 - price_diff_matrix

        combined_sim = 0.7 * facility_sim + 0.3 * price_sim
        top_5_indices = top_5.index.tolist()
        sim_scores = np.mean(combined_sim[top_5_indices], axis=0)

        sorted_indices = np.argsort(sim_scores)[::-1]
        content_based_indices = [i for i in sorted_indices if i not in top_5_indices][:10]

        top_10_cb = indonesia_hotels.iloc[content_based_indices].copy()
        top_10_cb['similarity_score'] = sim_scores[content_based_indices]

        st.subheader("🔁 Rekomendasi Hotel Serupa (Content-Based)")
        st.dataframe(top_10_cb[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'similarity_score']])

# ---------------------- FILTERING FUNCTION -----------------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    fasilitas_encoded = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    fitur_df = pd.DataFrame(fasilitas_encoded, columns=mlb.classes_, index=df.index)

    cosine_sim = cosine_similarity(fitur_df)

    try:
        idx = df.index[df['Hotel Name'] == hotel_name][0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    top_indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[top_indices]

# ---------------------- MAIN INTERFACE -----------------------
with tab3:
    st.write("Pilih lokasi strategis hotel idamanmu!")
# Buat peta
    m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        if row['Hotel Rating'] != 'Belum ada rating':
            image_url = row['Hotel Image'] if 'Hotel Image' in row and pd.notna(row['Hotel Image']) else ""
            popup_content = f"""
                <div style="width:200px">
                    <b>{row['Hotel Name']}</b><br>
                    Rating: {row['Hotel Rating']}<br>
                    {'<img src="' + image_url + '" width="160">' if image_url else ''}
                </div>
            """
            folium.Marker(
                location=[row['Lattitute'], row['Longitude']],
                popup=row['Hotel Name'],
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)

    # Tampilkan peta
    map_data = st_folium(m, width=800, height=500)

    # ---------------------- HOTEL KLIK DETEKSI -----------------------
    if map_data and map_data.get("last_object_clicked_popup"):
        selected_hotel_name = map_data["last_object_clicked_popup"]

        st.success(f"🏨 Hotel dipilih: *{selected_hotel_name}*")

        selected_hotel = df[df['Hotel Name'] == selected_hotel_name].iloc[0]

        st.subheader("📋 Detail Hotel")
        if pd.notna(selected_hotel['Hotel Image']):
            st.image(selected_hotel['Hotel Image'], width=400)
        st.markdown(f"*Nama:* {selected_hotel['Hotel Name']}")
        st.markdown(f"*Lokasi:* {selected_hotel['City']}, {selected_hotel['Provinsi']}")
        st.markdown(f"*Rating:* {selected_hotel['Hotel Rating']}")
        st.markdown(f"*Harga:* Rp {int(selected_hotel['Min'])} - Rp {int(selected_hotel['Max'])}")
        st.markdown(f"*Fasilitas:* {', '.join(selected_hotel['list_fasilitas'])}")

        # ---------------------- Rekomendasi Serupa -----------------------
        st.subheader("Rekomendasi Hotel Serupa")

        rekomendasi = content_based_recommendation(df, selected_hotel_name)

        for _, row in rekomendasi.iterrows():
            st.markdown(f"### 🏨 {row['Hotel Name']}")
            if pd.notna(row['Hotel Image']):
                st.image(row['Hotel Image'], width=400)
            st.write(f"📍 {row['City']} - {row['Provinsi']}")
            st.write(f"💰 Rp {int(row['Min'])} - Rp {int(row['Max'])}")
            st.write(f"⭐ Rating: {row['Hotel Rating']}")
            st.write("*Fasilitas:*", ", ".join(row['list_fasilitas']))
            st.markdown("---")
with tab4:
    # Judul bagian
    st.write("Tentang Kami")

    # Subjudul dengan HTML
    st.markdown('<h2 style="color:blue;">Together We Grow</h2>', unsafe_allow_html=True)

    # Deskripsi aplikasi
    st.write("""
    *Hotel Hunt* adalah aplikasi pencarian hotel yang dirancang untuk memberikan pengalaman pencarian yang cepat, mudah, dan personal. 
    Menggunakan pendekatan berbasis *rule-based* dan *content-based recommendation*, aplikasi ini membantu pengguna menemukan hotel yang paling sesuai dengan suasana hati, preferensi, dan lokasi yang diinginkan.

    Dengan antarmuka yang intuitif dan teknologi yang cerdas, Hotel Hunt hadir sebagai solusi bagi siapa saja yang ingin merencanakan perjalanan dengan lebih efisien dan menyenangkan.

    ---
    """)

    # Judul bagian tim pengembang dengan center alignment
    st.markdown('<h2 style="text-align: center; margin-bottom: 32px;">Tim Pengembang</h2>', unsafe_allow_html=True)

    # Foto tim pengembang sejajar di tengah
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 60px; margin-bottom: 32px;">
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{}" width="200" style="border-radius: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"/>
            <div style="margin-top: 10px; font-size: 18px;">Nikita Farah A</div>
        </div>
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{}" width="200" style="border-radius: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);"/>
            <div style="margin-top: 10px; font-size: 18px;">Najmatul Ilma M D</div>
        </div>
    </div>
    """.format(
        image_to_base64("assets/me_time.jpg"),
        image_to_base64("assets/me_time.jpg")
    ), unsafe_allow_html=True)

    # Garis pembatas dan pesan penutup
    st.write("---")
    st.markdown(
        "<div style='text-align: center;'>Kami percaya bahwa pencarian hotel seharusnya tidak rumit.<br>Dengan <strong>Hotel Hunt</strong>, temukan hotel terbaik sesuai suasana hatimu.</div>",
        unsafe_allow_html=True
    )
