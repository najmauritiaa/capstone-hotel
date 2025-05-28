import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# -------------------- CONFIG --------------------
st.set_page_config(layout="wide")

# -------------------- HERO SECTION (Unsplash Background) --------------------
st.markdown(
    """
    <style>
    .hero {
        position: relative;
        background-image: url("https://images.unsplash.com/photo-1501117716987-c8bd955fa90c");
        background-size: cover;
        background-position: center;
        height: 92vh;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 60px;
        color: white;
    }
    .hero h1 {
        font-size: 3.2em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .hero h2 {
        font-size: 1.8em;
        color: #e0c07c;
        margin-bottom: 0.5em;
    }
    .hero p {
        font-size: 1.2em;
        max-width: 700px;
    }
    .hero .stats {
        display: flex;
        gap: 80px;
        margin-top: 40px;
    }
    .hero .stat {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
    }
    </style>
    <div class="hero">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Hotel_font_awesome.svg/1024px-Hotel_font_awesome.svg.png" width="60" style="margin-bottom:15px">
        <h1>Hidup Nyaman & Elegan<br>di Jantung Kota Indonesia</h1>
        <h2>HotelHunt Smart Stay</h2>
        <p>
            Temukan hotel terbaik yang cocok dengan mood kamu — dari healing, adventure, sampai me-time. 
            Rekomendasi berdasarkan lokasi, fasilitas, dan kebutuhan emosionalmu. Coba sekarang!
        </p>
        <div class="stats">
            <div class="stat">50+<br>Destinasi</div>
            <div class="stat">#1<br>Rekomendasi Emosional</div>
            <div class="stat">500+<br>Hotel Terdaftar</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📋 Listing Hotel", "🗺️ Peta & Fasilitas"])

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("indonesia_hotels.csv")
    df = df.dropna()
    df[['Min', 'Max']] = df.apply(
        lambda row: pd.Series([row['Max'], row['Min']]) if row['Min'] > row['Max'] else pd.Series([row['Min'], row['Max']]),
        axis=1
    )
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# -------------------- TAB 2: Listing --------------------
with tab2:
    st.subheader("Rekomendasi Berdasarkan Mood & Budget")
    questions = {
        "Apa yang kamu rasakan saat ini?": ["senang", "cemas", "sedih"],
        "Kalau bisa memilih aktivitas sekarang, kamu ingin:": ["tidur", "menyendiri", "pacaran", "jalan-jalan"]
    }
    others = [
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
        "Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?"
    ]

    answers = {}
    for q, opts in questions.items():
        answers[q] = st.selectbox(q, opts)
    for q in others:
        answers[q] = st.selectbox(q, ["ya", "tidak"])

    budget_min = st.number_input("Budget Minimum (Rp)", min_value=0, value=300000, step=50000)
    budget_max = st.number_input("Budget Maksimum (Rp)", min_value=0, value=800000, step=50000)

    if st.button("🎯 Cari Hotel"):
        rules = {
            "Meditation": [("Apa yang kamu rasakan saat ini?", ["cemas", "sedih"]),
                           ("Apakah kamu merasa stres atau kewalahan akhir-akhir ini?", ["ya"])],
            "Me-time": [("Apakah kamu ingin menyendiri dan menjauh dari keramaian?", ["ya"]),
                        ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["menyendiri", "tidur"])],
            "Socialized": [("Apakah kamu ingin berbicara dengan seseorang saat ini?", ["ya"]),
                           ("Apakah kamu merasa senang ketika dikelilingi banyak orang?", ["ya"])],
            "Date": [("Apakah kamu sedang merindukan seseorang secara romantis?", ["ya"]),
                     ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["pacaran"])],
            "Adventure": [("Apakah kamu merasa bosan dengan rutinitasmu?", ["ya"]),
                          ("Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?", ["ya"])],
        }

        scores = {need: sum(answers[q] in a for q, a in conds) for need, conds in rules.items()}
        mood = max(scores, key=scores.get)
        st.success(f"Mood Terdeteksi: {mood}")

        keywords = {
            "Meditation": ['spa', 'yoga', 'tenang'],
            "Me-time": ['perpustakaan', 'tv', 'teras'],
            "Socialized": ['bar', 'karaoke'],
            "Date": ['romantis', 'laut', 'private'],
            "Adventure": ['hiking', 'menyelam', 'bersepeda']
        }.get(mood, [])

        def match_score(fasilitas): return sum(any(k in f.lower() for f in fasilitas) for k in keywords)

        filtered = df[(df['Min'] >= budget_min) & (df['Max'] <= budget_max)].copy()
        filtered['score'] = filtered['list_fasilitas'].apply(match_score)
        top = filtered.sort_values(by='score', ascending=False).head(5)
        st.dataframe(top[['Hotel Name', 'City', 'Min', 'Max', 'score']])

# -------------------- TAB 3: PETA --------------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    enc = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    sim = cosine_similarity(enc)
    idx = df.index[df['Hotel Name'] == hotel_name][0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in scores if i != idx][:top_n]
    return df.iloc[top_ids]

with tab3:
    st.subheader("🗺️ Klik Hotel di Peta untuk Lihat Detail & Rekomendasi")
    m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        if row['Hotel Rating'] != 'Belum ada rating':
            html = f"""
            <div style='width:200px'>
                <b>{row['Hotel Name']}</b><br>
                Rating: {row['Hotel Rating']}<br>
                {'<img src="' + row['Hotel Image'] + '" width="160">' if pd.notna(row['Hotel Image']) else ''}
            </div>
            """
            folium.Marker(
                location=[row['Lattitute'], row['Longitude']],
                popup=row['Hotel Name'],
                icon=folium.Icon(color='blue')
            ).add_to(marker_cluster)

    map_data = st_folium(m, width=800, height=500)

    if map_data and map_data.get("last_object_clicked_popup"):
        selected = map_data["last_object_clicked_popup"]
        data = df[df['Hotel Name'] == selected].iloc[0]
        st.success(f"Hotel dipilih: {selected}")
        if pd.notna(data['Hotel Image']):
            st.image(data['Hotel Image'], width=400)
        st.write(f"📍 {data['City']} - {data['Provinsi']}")
        st.write(f"💰 Rp {int(data['Min'])} - Rp {int(data['Max'])}")
        st.write(f"⭐ {data['Hotel Rating']}")
        st.write("Fasilitas:", ", ".join(data['list_fasilitas']))
        st.subheader("Hotel Serupa:")
        for _, rec in content_based_recommendation(df, selected).iterrows():
            st.markdown(f"### 🏨 {rec['Hotel Name']}")
            if pd.notna(rec['Hotel Image']):
                st.image(rec['Hotel Image'], width=400)
            st.write(f"{rec['City']} | Rp {int(rec['Min'])}-{int(rec['Max'])} | ⭐ {rec['Hotel Rating']}")
            st.write(", ".join(rec['list_fasilitas']))
            st.markdown("---")
