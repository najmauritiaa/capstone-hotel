import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------- CSS: Background Area Tab -----------------------
st.markdown(
    """
    <style>
    .tab-content {
        background-image: url('https://images.unsplash.com/photo-1501117716987-c8bd955fa90c');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        padding: 20px;
        border-radius: 12px;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- JUDUL -----------------------
st.markdown(
    """
    <div style="background-color:#e0f7fa;padding:10px 20px;border-radius:8px">
        <h1 style="color:#006064;">🏨 HOTEL HUNT</h1>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🏠", "Cari Berdasarkan Mood & Budget", "Cari Berdasarkan Peta"])

# ---------------------- DATASET -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('indonesia_hotels.csv')
    df = df.dropna()
    df[['Min', 'Max']] = df.apply(
        lambda row: pd.Series([row['Max'], row['Min']]) if row['Min'] > row['Max'] else pd.Series([row['Min'], row['Max']]),
        axis=1
    )
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# ---------------------- TAB 1 -----------------------
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Selamat Datang di Hotel Hunt! 🏨✨")
    st.write("""
        Temukan hotel impianmu dengan mudah dan cepat!  
        Sesuaikan pencarian berdasarkan suasana hati, anggaran, dan lokasi favoritmu.  
        Mari mulai petualangan mencari penginapan terbaik yang pas untuk kamu!
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- TAB 2 -----------------------
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

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
            "Meditation": [("Apa yang kamu rasakan saat ini?", ["cemas", "sedih"]),
                           ("Apakah kamu merasa stres atau kewalahan akhir-akhir ini?", ["ya"]),
                           ("Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?", ["ya"]),
                           ("Apakah kamu merasa butuh memahami dirimu lebih dalam?", ["ya"])],
            "Me-time": [("Apakah kamu ingin menyendiri dan menjauh dari keramaian?", ["ya"]),
                        ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["menyendiri", "tidur"])],
            "Socialized": [("Apakah kamu merasa kesepian hari ini?", ["ya"]),
                           ("Apakah kamu ingin berbicara dengan seseorang saat ini?", ["ya"]),
                           ("Apakah kamu merasa senang ketika dikelilingi banyak orang?", ["ya"])],
            "Date": [("Apakah kamu sedang merindukan seseorang secara romantis?", ["ya"]),
                     ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["pacaran"])],
            "Adventure": [("Apakah kamu merasa bosan dengan rutinitasmu?", ["ya"]),
                          ("Apakah kamu ingin mencoba hal baru hari ini?", ["ya"]),
                          ("Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?", ["ya"]),
                          ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["jalan-jalan"])],
            "Support": [("Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?", ["ya"]),
                        ("Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?", ["tidak"])]
        }

        scores = {need: sum(answers[q] in vals for q, vals in conds) for need, conds in rules.items()}
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
        def count_matching(facilities): return sum(any(kw in f.lower() for f in facilities) for kw in needed_keywords)

        filtered = df[(df['Min'] >= budget_min) & (df['Max'] <= budget_max)].copy()
        if filtered.empty:
            filtered = df.copy()
            st.warning("Menampilkan semua hotel karena tidak ada yang sesuai budget.")
        filtered['matching_score'] = filtered['list_fasilitas'].apply(count_matching)
        top_5 = filtered.sort_values(by='matching_score', ascending=False).head(5)
        st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'matching_score']])

        # Content-based filtering
        df['fasilitas_str'] = df['list_fasilitas'].apply(lambda x: ' '.join(x))
        vectorizer = CountVectorizer().fit_transform(df['fasilitas_str'])
        sim_fac = cosine_similarity(vectorizer)
        df['avg_price'] = (df['Min'] + df['Max']) / 2
        norm_price = MinMaxScaler().fit_transform(df[['avg_price']])
        sim_price = 1 - np.abs(norm_price - norm_price.T)
        sim_comb = 0.7 * sim_fac + 0.3 * sim_price
        sim_scores = np.mean(sim_comb[top_5.index], axis=0)
        idx_sorted = np.argsort(sim_scores)[::-1]
        idx_top10 = [i for i in idx_sorted if i not in top_5.index][:10]
        top_cb = df.iloc[idx_top10].copy()
        top_cb['similarity_score'] = sim_scores[idx_top10]
        st.subheader("🔁 Rekomendasi Hotel Serupa (Content-Based)")
        st.dataframe(top_cb[['Hotel Name', 'City', 'Min', 'Max', 'similarity_score']])

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- TAB 3 -----------------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    sim = cosine_similarity(encoded)
    idx = df.index[df['Hotel Name'] == hotel_name][0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)
    indices = [i[0] for i in scores if i[0] != idx][:top_n]
    return df.iloc[indices]

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.write("Silakan pilih hotel pada peta untuk melihat detail dan rekomendasi.")
    m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        if row['Hotel Rating'] != 'Belum ada rating':
            popup = f"""
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
        name = map_data["last_object_clicked_popup"]
        selected = df[df['Hotel Name'] == name].iloc[0]
        st.success(f"🏨 Hotel dipilih: *{name}*")
        if pd.notna(selected['Hotel Image']):
            st.image(selected['Hotel Image'], width=400)
        st.markdown(f"**Nama:** {selected['Hotel Name']}")
        st.markdown(f"**Lokasi:** {selected['City']}, {selected['Provinsi']}")
        st.markdown(f"**Rating:** {selected['Hotel Rating']}")
        st.markdown(f"**Harga:** Rp {int(selected['Min'])} - Rp {int(selected['Max'])}")
        st.markdown(f"**Fasilitas:** {', '.join(selected['list_fasilitas'])}")
        st.subheader("🔁 Rekomendasi Serupa")
        rekom = content_based_recommendation(df, name)
        for _, row in rekom.iterrows():
            st.markdown(f"### 🏨 {row['Hotel Name']}")
            if pd.notna(row['Hotel Image']):
                st.image(row['Hotel Image'], width=400)
            st.write(f"📍 {row['City']} - {row['Provinsi']}")
            st.write(f"💰 Rp {int(row['Min'])} - Rp {int(row['Max'])}")
            st.write(f"⭐ Rating: {row['Hotel Rating']}")
            st.write("Fasilitas:", ", ".join(row['list_fasilitas']))
            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)
