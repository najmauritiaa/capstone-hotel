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

st.set_page_config(layout="wide")

# --------- LOAD DATA ----------
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

# --------- TABS ---------
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📋 Listing Hotel", "🗺️ Peta & Fasilitas"])

# --------- TAB 1: BERANDA ---------
with tab1:
    st.title("🏨 HOTEL HUNT")
    st.markdown("""
        ### Hidup Nyaman & Elegan di Jantung Kota Indonesia  
        **HotelHunt Smart Stay**  
        Temukan hotel terbaik sesuai mood kamu — dari healing, adventure, sampai me-time.  
        Rekomendasi berdasarkan lokasi, fasilitas, dan kebutuhan emosionalmu.  
        """)
    st.markdown("---")
    st.markdown("📍 500+ Hotel | 🌍 50+ Destinasi | 💡 Rekomendasi berbasis AI")

# --------- TAB 2: LISTING ---------
with tab2:
    st.header("Cari Hotel Berdasarkan Mood & Budget")
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

# --------- TAB 3: PETA & FASILITAS ---------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    enc = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    sim = cosine_similarity(enc)
    idx = df.index[df['Hotel Name'] == hotel_name][0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in scores if i != idx][:top_n]
    return df.iloc[top_ids]

with tab3:
    st.header("Peta Hotel & Rekomendasi Serupa")
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
