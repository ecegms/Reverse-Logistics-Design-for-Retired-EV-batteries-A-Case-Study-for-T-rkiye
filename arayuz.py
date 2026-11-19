
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pulp
import math

# Sayfa ayarları
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# GÖRSEL TASARIM (CSS INJECTION)
# ==========================================
custom_css = """
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #2F70AF !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1A3E62 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# TÜRKİYE 81 İL KOORDİNATLARI VE VARSAYILAN VERİ
# ==========================================
IL_KOORDINATLARI = {
    "Adana": (37.0000, 35.3213), "Adıyaman": (37.7648, 38.2786), "Afyonkarahisar": (38.7507, 30.5567),
    "Ağrı": (39.7191, 43.0503), "Amasya": (40.6499, 35.8353), "Ankara": (39.9334, 32.8597),
    "Antalya": (36.8969, 30.7133), "Artvin": (41.1828, 41.8183), "Aydın": (37.8380, 27.8456),
    "Balıkesir": (39.6484, 27.8826), "Bilecik": (40.1451, 29.9799), "Bingöl": (38.8847, 40.4939),
    "Bitlis": (38.4006, 42.1095), "Bolu": (40.7392, 31.6116), "Burdur": (37.7183, 30.2823),
    "Bursa": (40.1828, 29.0667), "Çanakkale": (40.1553, 26.4142), "Çankırı": (40.6013, 33.6134),
    "Çorum": (40.5506, 34.9556), "Denizli": (37.7765, 29.0864), "Diyarbakır": (37.9144, 40.2306),
    "Edirne": (41.6744, 26.5583), "Elazığ": (38.6810, 39.2264), "Erzincan": (39.7500, 39.5000),
    "Erzurum": (39.9000, 41.2700), "Eskişehir": (39.7767, 30.5206), "Gaziantep": (37.0662, 37.3833),
    "Giresun": (40.9128, 38.3895), "Gümüşhane": (40.4606, 39.4814), "Hakkari": (37.5744, 43.7408),
    "Hakkâri": (37.5744, 43.7408), "Hatay": (36.2000, 36.1667), "Isparta": (37.7648, 30.5566),
    "Mersin": (36.8000, 34.6333), "İstanbul": (41.0082, 28.9784), "İzmir": (38.4237, 27.1428),
    "Kars": (40.6013, 43.0975), "Kastamonu": (41.3766, 33.7765), "Kayseri": (38.7312, 35.4787),
    "Kırklareli": (41.7333, 27.2167), "Kırşehir": (39.1425, 34.1709), "Kocaeli": (40.7654, 29.9408),
    "Konya": (37.8667, 32.4833), "Kütahya": (39.4167, 29.9833), "Malatya": (38.3552, 38.3095),
    "Manisa": (38.6191, 27.4289), "Kahramanmaraş": (37.5858, 36.9371), "Mardin": (37.3131, 40.7436),
    "Muğla": (37.2153, 28.3636), "Muş": (38.7304, 41.4990), "Nevşehir": (38.6244, 34.7144),
    "Niğde": (37.9667, 34.6833), "Ordu": (40.9839, 37.8764), "Rize": (41.0201, 40.5234),
    "Sakarya": (40.7569, 30.3783), "Samsun": (41.2867, 36.3300), "Siirt": (37.9333, 41.9500),
    "Sinop": (42.0231, 35.1531), "Sivas": (39.7477, 37.0179), "Tekirdağ": (40.9780, 27.5085),
    "Tokat": (40.3167, 36.5500), "Trabzon": (41.0015, 39.7178), "Tunceli": (39.1079, 39.5401),
    "Şanlıurfa": (37.1591, 38.7969), "Uşak": (38.6823, 29.4082), "Van": (38.4891, 43.3889),
    "Yozgat": (39.8181, 34.8147), "Zonguldak": (41.4564, 31.7987), "Aksaray": (38.3687, 34.0370),
    "Bayburt": (40.2552, 40.2249), "Karaman": (37.1811, 33.2222), "Kırıkkale": (39.8453, 33.5063),
    "Batman": (37.8812, 41.1351), "Şırnak": (37.5228, 42.4594), "Bartın": (41.6344, 32.3375),
    "Ardahan": (41.1105, 42.7022), "Iğdır": (39.9237, 44.0450), "Yalova": (40.6500, 29.2667),
    "Karabük": (41.2061, 32.6228), "Kilis": (36.7184, 37.1147), "Osmaniye": (37.0742, 36.2475),
    "Düzce": (40.8438, 31.1565)
}

VARSAYILAN_EV_VERISI = {
    "İstanbul": 134054, "Ankara": 47965, "İzmir": 26992, "Antalya": 17436, "Bursa": 16673,
    "Kocaeli": 8927, "Konya": 7210, "Adana": 5877, "Kayseri": 5657, "Gaziantep": 5539,
    "Mersin": 5507, "Muğla": 5129, "Denizli": 3899, "Sakarya": 3727, "Samsun": 3711,
    "Eskişehir": 3262, "Aydın": 3203, "Balıkesir": 3142, "Tekirdağ": 3089, "Manisa": 2851,
    "Trabzon": 2426, "Hatay": 1954, "Kahramanmaraş": 1703, "Diyarbakır": 1667, "Afyonkarahisar": 1538,
    "Isparta": 1476, "Çanakkale": 1468, "Elazığ": 1442, "Yalova": 1228, "Sivas": 1088,
    "Ordu": 1086, "Kütahya": 1071, "Erzurum": 1046, "Malatya": 1039, "Şanlıurfa": 1016,
    "Düzce": 888, "Edirne": 874, "Bolu": 868, "Zonguldak": 815, "Uşak": 778,
    "Rize": 694, "Nevşehir": 691, "Aksaray": 681, "Kırklareli": 642, "Çorum": 613,
    "Giresun": 613, "Tokat": 592, "Osmaniye": 566, "Burdur": 538, "Van": 516,
    "Amasya": 475, "Mardin": 460, "Bilecik": 422, "Karabük": 388, "Karaman": 379,
    "Batman": 371, "Yozgat": 370, "Niğde": 347, "Kastamonu": 333, "Adıyaman": 310,
    "Kırıkkale": 284, "Kırşehir": 275, "Erzincan": 256, "Çankırı": 213, "Bartın": 201,
    "Sinop": 201, "Artvin": 154, "Bingöl": 129, "Gümüşhane": 110, "Siirt": 105,
    "Bayburt": 94, "Kilis": 85, "Şırnak": 82, "Kars": 81, "Bitlis": 80,
    "Muş": 66, "Ağrı": 65, "Iğdır": 65, "Tunceli": 54, "Ardahan": 38, "Hakkâri": 14
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==========================================
# 1. SESSION STATE YÖNETİMİ
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'mod1_data' not in st.session_state:
    st.session_state.mod1_data = None
if 'mod1_step' not in st.session_state:
    st.session_state.mod1_step = 0
if 'senaryo_gecmisi' not in st.session_state:
    st.session_state.senaryo_gecmisi = []

def reset_system():
    st.session_state.mod1_data = None
    st.session_state.mod1_step = 0
    st.session_state.senaryo_gecmisi = []
    for k in ['pareto_sonuclari', 'kmeans_sonuclari', 'kmeans_calisti', 'milp_sonuc']:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Sistem hafızası sıfırlandı. Yeni veri yüklenebilir.")

def go_to_dashboard():
    st.session_state.current_view = 'dashboard'

# ==========================================
# 2. GİRİŞ EKRANI (Login)
# ==========================================
def show_login():
    st.markdown("<h1 style='text-align: center;'>EV Batarya Tersine Lojistik Ağı KDS</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; opacity: 0.7;'>Sisteme Giriş Yapınız</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "proje2026":
                    st.session_state.logged_in = True
                    st.session_state.current_view = 'dashboard'
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

# ==========================================
# 3. ANA PANEL (Dashboard)
# ==========================================
def show_dashboard():
    dashboard_css = """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #F2F4F7 !important;
        color: #1E293B !important;
    }
    [data-testid="stHeader"] {
        background-color: #F2F4F7 !important;
    }
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 1080px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    /* Modül 1 & Modül 2: 3D Dikey Dikdörtgen Kurumsal Mavi Kartlar */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        width: 320px !important;
        max-width: 100% !important;
        min-height: 400px !important;
        padding: 36px 28px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        background: linear-gradient(180deg, #3B82C4 0%, #245B93 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-bottom: 6px solid #154360 !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 24px rgba(21, 67, 96, 0.24), 0 4px 10px rgba(21, 67, 96, 0.14) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, border-bottom-width 0.1s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(180deg, #448DD1 0%, #2966A3 100%) !important;
        transform: translateY(-5px) !important;
        border-bottom: 6px solid #123850 !important;
        box-shadow: 0 20px 34px rgba(21, 67, 96, 0.32), 0 8px 14px rgba(21, 67, 96, 0.18) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:active,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:active,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:active {
        transform: translateY(3px) !important;
        border-bottom: 2px solid #154360 !important;
        box-shadow: 0 4px 10px rgba(21, 67, 96, 0.22) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] [data-testid="stMarkdownContainer"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        text-align: center !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] p {
        margin: 0 !important;
        font-size: 0.96rem !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        color: rgba(255, 255, 255, 0.92) !important;
        white-space: normal !important;
        text-align: center !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] strong,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] strong,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] strong {
        display: block !important;
        font-size: 1.28rem !important;
        font-weight: 800 !important;
        line-height: 1.35 !important;
        letter-spacing: 0.4px !important;
        color: #FFFFFF !important;
        margin-bottom: 18px !important;
    }

    /* Güvenli Çıkış Yap Butonu */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 2px solid #334155 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(30, 41, 59, 0.08) !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] p {
        color: #1E293B !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #1E293B !important;
        box-shadow: 0 6px 14px rgba(30, 41, 59, 0.18) !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover p {
        color: #FFFFFF !important;
    }
    </style>
    """
    st.markdown(dashboard_css, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #1E293B; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 12px;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #475569; font-size: 17px; max-width: 780px; margin: 0 auto 36px auto; line-height: 1.65;'>
    Bu Karar Destek Sistemi, ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde toplanması ve geri kazanılması için tersine lojistik ağını tasarlamak amacıyla geliştirilmiştir.
    </p>
    """, unsafe_allow_html=True)

    _, col1, col2, _ = st.columns([1, 2.2, 2.2, 1], gap="large")
    with col1:
        if st.button("**MODÜL 1: ADAY MERKEZ BELİRLEME**\n\nPareto, K-Means ve Minimax algoritmaları ile aday tesis konumlarını belirleyin.", key="btn_mod1", type="primary"):
            st.session_state.current_view = 'module1'
            st.rerun()
    with col2:
        if st.button("**MODÜL 2: OPTİMİZASYON VE SENARYO ANALİZİ**\n\nParametreleri değiştirerek MILP modelini çalıştırın ve tüm karar değişkenlerini analiz edin.", key="btn_mod2", type="primary"):
            st.session_state.current_view = 'module2'
            st.rerun()

    st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
    _, col_logout, _ = st.columns([4, 2.5, 4])
    with col_logout:
        st.button("Güvenli Çıkış Yap", on_click=lambda: st.session_state.update({'logged_in': False, 'current_view': 'login'}), use_container_width=True)

# ==========================================
# 4. MODÜL 1 İÇERİĞİ
# ==========================================
def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme")
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    st.divider()
    
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: EV Verisi Yükleme")
        st.write("Sütun başlıkları 'İl' ve 'EV_Sayısı' olan Excel dosyasını yükleyin (veya boş bırakıp varsayılan 81 il verisini kullanın):")
        col_up, col_def = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("Excel Yükle", type=['xlsx', 'xls'], label_visibility="collapsed")
        with col_def:
            if st.button("Varsayılan 81 İl Verisini Kullan", use_container_width=True):
                df_def = pd.DataFrame(list(VARSAYILAN_EV_VERISI.items()), columns=['İl', 'EV_Sayısı'])
                st.session_state.mod1_data = df_def
                st.session_state.mod1_step = 1
                st.rerun()

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if 'İl' not in df.columns or 'EV_Sayısı' not in df.columns:
                    st.error("Excel dosyasında 'İl' ve 'EV_Sayısı' sütunları bulunamadı.")
                else:
                    st.session_state.mod1_data = df
                    st.success(f"Veri başarıyla işlendi. Toplam {len(df)} il tespit edildi.")
                    if st.button("Pareto Analizini Başlat", type="primary"):
                        st.session_state.mod1_step = 1
                        st.rerun()
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")

    if st.session_state.mod1_step >= 1 and st.session_state.mod1_data is not None:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi Sonuçları")
        df = st.session_state.mod1_data
        df_sorted = df.sort_values(by='EV_Sayısı', ascending=False).reset_index(drop=True)
        toplam_ev = df_sorted['EV_Sayısı'].sum()
        df_sorted['Kümülatif_Toplam'] = df_sorted['EV_Sayısı'].cumsum()
        df_sorted['Kümülatif_Yüzde'] = (df_sorted['Kümülatif_Toplam'] / toplam_ev) * 100
        
        idx_80 = df_sorted[df_sorted['Kümülatif_Yüzde'] >= 80].index.min()
        pareto_df = df_sorted.iloc[:idx_80 + 1].copy()
        st.info(f"Toplam talebin %80'ini karşılayan en yüksek yoğunluklu {len(pareto_df)} il, Aday Toplama Merkezi (j) olarak belirlendi.")
        
        m_pareto = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        for index, row in pareto_df.iterrows():
            il_adi = str(row['İl']).strip().title()
            if il_adi in IL_KOORDINATLARI:
                lat, lon = IL_KOORDINATLARI[il_adi]
                tooltip_html = f"<b>Tesis:</b> Aday Toplama Merkezi<br><b>İl:</b> {il_adi}<br><b>EV:</b> {row['EV_Sayısı']:,.0f}"
                folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_pareto)
                
        st_folium(m_pareto, width=900, height=420, key="map_pareto")
        st.session_state.pareto_sonuclari = pareto_df

        if st.button("K-Means Kümeleme Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 2
            st.rerun()

    if st.session_state.mod1_step >= 2:
        st.divider()
        st.subheader("Adım 3: Dirsek Metodu ve SPSS Tabanlı K-Means Kümeleme")
        if st.button("Dirsek Metodu ve K-Means Algoritmasını Çalıştır", type="primary", key="btn_kmeans"):
            pareto_df = st.session_state.pareto_sonuclari
            coords, sehirler = [], []
            for _, row in pareto_df.iterrows():
                il_adi = str(row['İl']).strip().title()
                if il_adi in IL_KOORDINATLARI:
                    coords.append(IL_KOORDINATLARI[il_adi])
                    sehirler.append(il_adi)
                    
            X = np.array(coords)
            max_k = min(10, len(coords))
            k_degerleri = list(range(2, max_k + 1))
            wcss = []
            
            for k in k_degerleri:
                kmeans = KMeans(n_clusters=k, init=X[:k], n_init=1, max_iter=10, random_state=None)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
                
            x1, y1 = k_degerleri[0], wcss[0]
            x2, y2 = k_degerleri[-1], wcss[-1]
            uzakliklar = [abs((y2 - y1)*k_degerleri[i] - (x2 - x1)*wcss[i] + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2) for i in range(len(k_degerleri))]
            optimum_k = k_degerleri[np.argmax(uzakliklar)]
            
            st.session_state.optimum_k = optimum_k
            st.session_state.wcss = wcss
            st.session_state.k_degerleri = k_degerleri
            
            kmeans_final = KMeans(n_clusters=optimum_k, init=X[:optimum_k], n_init=1, max_iter=10, random_state=None)
            kmeans_final.fit(X)
            centroids = kmeans_final.cluster_centers_
            
            aday_isleme_merkezleri = []
            for i, centroid in enumerate(centroids):
                min_dist = float('inf')
                closest_city = None
                for sehir in sehirler:
                    lat, lon = IL_KOORDINATLARI[sehir]
                    dist = np.sqrt((centroid[0] - lat)**2 + (centroid[1] - lon)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_city = sehir
                if not any(d['İl'] == closest_city for d in aday_isleme_merkezleri):
                    aday_isleme_merkezleri.append({
                        "İl": closest_city,
                        "Lat": IL_KOORDINATLARI[closest_city][0],
                        "Lon": IL_KOORDINATLARI[closest_city][1],
                        "Kod": f"İM-{i+1}"
                    })
            st.session_state.kmeans_sonuclari = aday_isleme_merkezleri
            st.session_state.kmeans_calisti = True

        if st.session_state.get('kmeans_calisti', False):
            opt_k = st.session_state.optimum_k
            col_grafik, col_table = st.columns([2, 1])
            with col_grafik:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(st.session_state.k_degerleri, st.session_state.wcss, marker='o', color='#2F70AF', linewidth=2)
                opt_wcss = st.session_state.wcss[opt_k - 2]
                ax.plot(opt_k, opt_wcss, marker='o', markersize=10, color='red')
                ax.set_title(f"Dirsek Grafiği (Optimum k={opt_k})")
                ax.set_xlabel("Küme Sayısı (k)")
                ax.set_ylabel("SSE")
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
            with col_table:
                sse_df = pd.DataFrame({"Küme Sayısı (k)": st.session_state.k_degerleri, "SSE": [f"{v:,.2f}" for v in st.session_state.wcss]})
                st.dataframe(sse_df, use_container_width=True, hide_index=True)

            if st.button("Minimax Algoritmasını Çalıştır", type="primary"):
                st.session_state.mod1_step = 3
                st.rerun()

    if st.session_state.mod1_step >= 3:
        st.divider()
        st.subheader("Adım 4: Minimax Algoritması Sonuçları")
        st.success("Tüm aday merkez konumları başarıyla tespit edildi! Aday Toplama, İşleme, Geri Dönüşüm ve İkinci Ömür Merkezleri haritada gösterilmektedir.")
        
        m_minimax = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        if 'pareto_sonuclari' in st.session_state:
            for _, row in st.session_state.pareto_sonuclari.iterrows():
                il_adi = str(row['İl']).strip().title()
                if il_adi in IL_KOORDINATLARI:
                    lat, lon = IL_KOORDINATLARI[il_adi]
                    folium.Marker([lat, lon], tooltip=f"Aday Toplama Merkezi: {il_adi}", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_minimax)

        if 'kmeans_sonuclari' in st.session_state:
            for merkez in st.session_state.kmeans_sonuclari:
                folium.Marker([merkez['Lat'], merkez['Lon']], tooltip=f"Aday İşleme Merkezi: {merkez['İl']} ({merkez['Kod']})", icon=folium.Icon(color="red", icon="cog")).add_to(m_minimax)

        geri_donusum_merkezleri = [("Adana", 37.0000, 35.3213, "GDM-01")]
        ikinci_omur_merkezleri = [("Konya", 37.8667, 32.4833, "İÖM-01")]
        
        for il, lat, lon, kod in geri_donusum_merkezleri:
            folium.Marker([lat, lon], tooltip=f"Aday Geri Dönüşüm Merkezi: {il} ({kod})", icon=folium.Icon(color="green", icon="recycle")).add_to(m_minimax)
            
        for il, lat, lon, kod in ikinci_omur_merkezleri:
            folium.Marker([lat, lon], tooltip=f"Aday İkinci Ömür Merkezi: {il} ({kod})", icon=folium.Icon(color="orange", icon="flash")).add_to(m_minimax)

        st_folium(m_minimax, width=900, height=450, key="map_minimax")

        st.write("<br>", unsafe_allow_html=True)
        if st.button("Modül 2: Optimizasyon Modelini Çöz ➡", type="primary"):
            st.session_state.current_view = 'module2'
            st.rerun()

# ==========================================
# 5. MODÜL 2: MILP OPTİMİZASYON VE SENARYO ANALİZİ
# ==========================================
def solve_milp_model(params):
    ev_dict = params['ev_data']
    I = list(ev_dict.keys())
    J = params['aday_J']
    K = params['aday_K']
    L = params['aday_L']
    M = params['aday_M']
    B = ['Binek', 'Agir_Vasita']
    N = ['Grafit', 'Aluminyum', 'Bakir', 'Kobalt', 'Lityum', 'Manganez', 'Nikel']

    binek_orani = params['binek_payi'] / 100.0
    gamma = params['gamma'] / 100.0

    DE = {}
    for i in I:
        toplam_arac = ev_dict[i]
        DE[(i, 'Binek')] = toplam_arac * binek_orani
        DE[(i, 'Agir_Vasita')] = toplam_arac * (1.0 - binek_orani)

    H = {'Binek': 96, 'Agir_Vasita': 1008}
    C = {'Binek': 3.5, 'Agir_Vasita': 0.0395}

    R1 = params['R1'] / 100.0
    R2 = params['R2'] / 100.0
    R3 = params['R3'] / 100.0

    R21, R22, R23 = 0.2047, 0.7019, 0.0934
    CR = {'Grafit': 0.16, 'Aluminyum': 0.15, 'Bakir': 0.10, 'Kobalt': 0.07, 'Lityum': 0.07, 'Manganez': 0.05, 'Nikel': 0.04}
    RR = {'Grafit': 0.945, 'Aluminyum': 0.955, 'Bakir': 0.98, 'Kobalt': 0.965, 'Lityum': 0.98, 'Manganez': 0.99, 'Nikel': 0.985}

    REV = {
        'Grafit': 1.2, 'Aluminyum': 2.4, 'Bakir': 8.5,
        'Kobalt': params['rev_kobalt'], 'Lityum': params['rev_lityum'],
        'Manganez': 1.8, 'Nikel': params['rev_nikel']
    }

    CC = {'Binek': 8.3 * 96, 'Agir_Vasita': 8.3 * 1008} if params['hucre_bazli_cc'] else {'Binek': 120.0, 'Agir_Vasita': 250.0}
    C1 = {'Binek': params['c1_binek'], 'Agir_Vasita': params['c1_agir']}
    C21, C22, C23 = 62.3, 20.5, 0.215
    C3 = params['c3_tasima']
    C4 = params['c4_tasima']
    C5 = params['c5_bertaraf']
    REC = {'Binek': 5.085, 'Agir_Vasita': 450.0}

    fj, fk, fl, fm = params['fj'], params['fk'], params['fl'], params['fm']
    Cap_jb = {'Binek': params['cap_j'], 'Agir_Vasita': params['cap_j']}
    Cap_kb = {'Binek': params['cap_k'], 'Agir_Vasita': params['cap_k']}
    Cap_l = params['cap_l'] * 1000.0
    Cap_mb = {'Binek': params['cap_m'] * 1000.0, 'Agir_Vasita': params['cap_m'] * 1000.0}

    d_ij = {(i, j): haversine_km(*IL_KOORDINATLARI[i], *IL_KOORDINATLARI[j]) for i in I for j in J}
    d_jk = {(j, k): haversine_km(*IL_KOORDINATLARI[j], *IL_KOORDINATLARI[k]) for j in J for k in K}
    d_kl = {(k, l): haversine_km(*IL_KOORDINATLARI[k], *IL_KOORDINATLARI[l]) for k in K for l in L}
    d_km = {(k, m): haversine_km(*IL_KOORDINATLARI[k], *IL_KOORDINATLARI[m]) for k in K for m in M}

    model = pulp.LpProblem("EV_Batarya_Tersine_Lojistik_MILP", pulp.LpMinimize)

    # Python 3.13+ ve PuLP uyumsuzluğunu aşmak için Binary yerine lowBound=0, upBound=1, cat='Integer' kullanımı
    x_ij = { (i, j): pulp.LpVariable(f"x_ij_{i}_{j}", lowBound=0, upBound=1, cat='Integer') for i in I for j in J }
    x_j = { j: pulp.LpVariable(f"x_j_{j}", lowBound=0, upBound=1, cat='Integer') for j in J }
    x_k = { k: pulp.LpVariable(f"x_k_{k}", lowBound=0, upBound=1, cat='Integer') for k in K }
    x_l = { l: pulp.LpVariable(f"x_l_{l}", lowBound=0, upBound=1, cat='Integer') for l in L }
    x_m = { m: pulp.LpVariable(f"x_m_{m}", lowBound=0, upBound=1, cat='Integer') for m in M }

    z_ijb = { (i, j, b): pulp.LpVariable(f"z_ijb_{i}_{j}_{b}", lowBound=0) for i in I for j in J for b in B }
    W_jkb = { (j, k, b): pulp.LpVariable(f"W_jkb_{j}_{k}_{b}", lowBound=0) for j in J for k in K for b in B }
    beta_kb = { (k, b): pulp.LpVariable(f"beta_kb_{k}_{b}", lowBound=0) for k in K for b in B }
    W_kmb = { (k, m, b): pulp.LpVariable(f"W_kmb_{k}_{m}_{b}", lowBound=0) for k in K for m in M for b in B }
    W_klb = { (k, l, b): pulp.LpVariable(f"W_klb_{k}_{l}_{b}", lowBound=0) for k in K for l in L for b in B }
    W_kb = { (k, b): pulp.LpVariable(f"W_kb_{k}_{b}", lowBound=0) for k in K for b in B }

    Wrep_lb = { (l, b): pulp.LpVariable(f"Wrep_lb_{l}_{b}", lowBound=0) for l in L for b in B }
    Wrem_lb = { (l, b): pulp.LpVariable(f"Wrem_lb_{l}_{b}", lowBound=0) for l in L for b in B }
    Wreu_lb = { (l, b): pulp.LpVariable(f"Wreu_lb_{l}_{b}", lowBound=0) for l in L for b in B }

    theta_lb = { (l, b): pulp.LpVariable(f"theta_lb_{l}_{b}", lowBound=0) for l in L for b in B }
    theta_mb = { (m, b): pulp.LpVariable(f"theta_mb_{m}_{b}", lowBound=0) for m in M for b in B }
    theta_mn = { (m, n): pulp.LpVariable(f"theta_mn_{m}_{n}", lowBound=0) for m in M for n in N }

    # Amaç Fonksiyonu Bileşenleri
    maliyet_toplama = pulp.lpSum([z_ijb[(i, j, b)] * CC[b] + z_ijb[(i, j, b)] * d_ij[(i, j)] * C3 for i in I for j in J for b in B])
    maliyet_isleme = pulp.lpSum([W_jkb[(j, k, b)] * C1[b] + W_jkb[(j, k, b)] * d_jk[(j, k)] * C3 for j in J for k in K for b in B])
    maliyet_ikinci_omur = pulp.lpSum([(C21 * Wrep_lb[(l, b)] + C22 * Wrem_lb[(l, b)] + C23 * Wreu_lb[(l, b)]) + pulp.lpSum([W_klb[(k, l, b)] * d_kl[(k, l)] * C4 for k in K]) for l in L for b in B])
    maliyet_geri_donusum = pulp.lpSum([W_kmb[(k, m, b)] * d_km[(k, m)] * C4 for k in K for m in M for b in B]) + pulp.lpSum([theta_mb[(m, b)] * REC[b] for m in M for b in B])
    gelir_geri_donusum = pulp.lpSum([REV[n] * theta_mn[(m, n)] for m in M for n in N])
    maliyet_bertaraf = pulp.lpSum([W_kb[(k, b)] * C5 for k in K for b in B])
    maliyet_insa = (
        pulp.lpSum([fj * x_j[j] for j in J]) +
        pulp.lpSum([fk * x_k[k] for k in K]) +
        pulp.lpSum([fl * x_l[l] for l in L]) +
        pulp.lpSum([fm * x_m[m] for m in M])
    )

    model += (maliyet_toplama + maliyet_isleme + maliyet_ikinci_omur + maliyet_geri_donusum - gelir_geri_donusum + maliyet_bertaraf + maliyet_insa), "Toplam_Net_Maliyet"

    # Kısıtlar
    for i in I:
        model += pulp.lpSum([x_ij[(i, j)] for j in J]) == 1, f"Atama_{i}"

    for i in I:
        for j in J:
            for b in B:
                model += z_ijb[(i, j, b)] == DE[(i, b)] * gamma * x_ij[(i, j)], f"Akis_z_{i}_{j}_{b}"

    for j in J:
        for b in B:
            model += pulp.lpSum([z_ijb[(i, j, b)] for i in I]) <= Cap_jb[b] * x_j[j], f"Cap_j_{j}_{b}"
            model += pulp.lpSum([z_ijb[(i, j, b)] for i in I]) == pulp.lpSum([W_jkb[(j, k, b)] for k in K]), f"Denge_j_{j}_{b}"

    for k in K:
        for b in B:
            model += pulp.lpSum([W_jkb[(j, k, b)] for j in J]) == beta_kb[(k, b)], f"Beta_{k}_{b}"
            model += Cap_kb[b] * x_k[k] >= beta_kb[(k, b)], f"Cap_k_{k}_{b}"
            model += beta_kb[(k, b)] * R1 * H[b] == pulp.lpSum([W_kmb[(k, m, b)] for m in M]), f"GD_Akis_{k}_{b}"
            model += beta_kb[(k, b)] * R2 * H[b] == pulp.lpSum([W_klb[(k, l, b)] for l in L]), f"IO_Akis_{k}_{b}"
            model += beta_kb[(k, b)] * R3 * H[b] == W_kb[(k, b)], f"Bertaraf_{k}_{b}"

    for m in M:
        for n in N:
            model += theta_mn[(m, n)] == pulp.lpSum([W_kmb[(k, m, b)] * C[b] for b in B for k in K]) * CR[n] * RR[n], f"Mat_{m}_{n}"
        for b in B:
            model += theta_mb[(m, b)] == pulp.lpSum([W_kmb[(k, m, b)] * C[b] for k in K]), f"Agirlik_m_{m}_{b}"
            model += Cap_mb[b] * x_m[m] >= theta_mb[(m, b)], f"Cap_m_{m}_{b}"

    for l in L:
        for b in B:
            model += pulp.lpSum([W_klb[(k, l, b)] for k in K]) * R21 == Wrep_lb[(l, b)], f"Wrep_{l}_{b}"
            model += pulp.lpSum([W_klb[(k, l, b)] for k in K]) * R22 == Wrem_lb[(l, b)], f"Wrem_{l}_{b}"
            model += pulp.lpSum([W_klb[(k, l, b)] for k in K]) * R23 == Wreu_lb[(l, b)], f"Wreu_{l}_{b}"
            model += pulp.lpSum([W_klb[(k, l, b)] * C[b] for k in K]) == theta_lb[(l, b)], f"Agirlik_l_{l}_{b}"
        model += pulp.lpSum([theta_lb[(l, b)] for b in B]) <= Cap_l * x_l[l], f"Cap_l_{l}"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "durum": pulp.LpStatus[status],
        "amac_degeri": pulp.value(model.objective),
        "maliyet_kalemleri": {
            "Toplama Merkezi (Taşıma + İşlem)": pulp.value(maliyet_toplama),
            "İşleme Merkezi (Taşıma + İşlem)": pulp.value(maliyet_isleme),
            "İkinci Ömür (Taşıma + İşlem)": pulp.value(maliyet_ikinci_omur),
            "Geri Dönüşüm (Taşıma + İşlem)": pulp.value(maliyet_geri_donusum),
            "Bertaraf Maliyeti": pulp.value(maliyet_bertaraf),
            "Sabit Tesis İnşa Maliyeti": pulp.value(maliyet_insa),
            "Materyal Satış Geliri (-)": -pulp.value(gelir_geri_donusum)
        },
        "x_j": {j: int(round(pulp.value(x_j[j]) or 0)) for j in J},
        "x_k": {k: int(round(pulp.value(x_k[k]) or 0)) for k in K},
        "x_l": {l: int(round(pulp.value(x_l[l]) or 0)) for l in L},
        "x_m": {m: int(round(pulp.value(x_m[m]) or 0)) for m in M},
        "x_ij": [(i, j) for i in I for j in J if (pulp.value(x_ij[(i, j)]) or 0) > 0.5],
        "z_ijb": [{"İl (i)": i, "Toplama Merkezi (j)": j, "Batarya Tipi (b)": b, "Miktar (Adet)": round(pulp.value(z_ijb[(i, j, b)]), 2)}
                  for i in I for j in J for b in B if (pulp.value(z_ijb[(i, j, b)]) or 0) > 0.01],
        "W_jkb": [{"Toplama Merkezi (j)": j, "İşleme Merkezi (k)": k, "Batarya Tipi (b)": b, "Miktar (Adet)": round(pulp.value(W_jkb[(j, k, b)]), 2)}
                  for j in J for k in K for b in B if (pulp.value(W_jkb[(j, k, b)]) or 0) > 0.01],
        "beta_kb": [{"İşleme Merkezi (k)": k, "Batarya Tipi (b)": b, "Toplam Batarya (Adet)": round(pulp.value(beta_kb[(k, b)]), 2)}
                    for k in K for b in B if (pulp.value(beta_kb[(k, b)]) or 0) > 0.01],
        "W_kmb": [{"İşleme Merkezi (k)": k, "Geri Dönüşüm Merkezi (m)": m, "Batarya Tipi (b)": b, "Hücre Sayısı": round(pulp.value(W_kmb[(k, m, b)]), 2)}
                  for k in K for m in M for b in B if (pulp.value(W_kmb[(k, m, b)]) or 0) > 0.01],
        "W_klb": [{"İşleme Merkezi (k)": k, "İkinci Ömür Merkezi (l)": l, "Batarya Tipi (b)": b,
                   "Toplam Hücre": round(pulp.value(W_klb[(k, l, b)]), 2),
                   "Wrep (Farklı Amaç)": round(pulp.value(Wrep_lb[(l, b)]), 2),
                   "Wrem (Yeniden İmalat)": round(pulp.value(Wrem_lb[(l, b)]), 2),
                   "Wreu (Doğrudan Kullanım)": round(pulp.value(Wreu_lb[(l, b)]), 2)}
                  for k in K for l in L for b in B if (pulp.value(W_klb[(k, l, b)]) or 0) > 0.01],
        "W_kb": [{"İşleme Merkezi (k)": k, "Batarya Tipi (b)": b, "Bertarafa Giden Hücre": round(pulp.value(W_kb[(k, b)]), 2)}
                 for k in K for b in B if (pulp.value(W_kb[(k, b)]) or 0) > 0.01],
        "theta_mb": [{"Geri Dönüşüm Merkezi (m)": m, "Batarya Tipi (b)": b, "Hücre Ağırlığı (kg)": round(pulp.value(theta_mb[(m, b)]), 2)}
                     for m in M for b in B if (pulp.value(theta_mb[(m, b)]) or 0) > 0.01],
        "theta_mn": [{"Geri Dönüşüm Merkezi (m)": m, "Materyal (n)": n, "Geri Kazanılan Miktar (kg)": round(pulp.value(theta_mn[(m, n)]), 2),
                      "Satış Geliri ($)": round(pulp.value(theta_mn[(m, n)]) * REV[n], 2)}
                     for m in M for n in N if (pulp.value(theta_mn[(m, n)]) or 0) > 0.01]
    }

def show_module2():
    st.header("Modül 2: MILP Ağ Optimizasyonu ve Senaryo Analizi")
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    st.divider()

    if 'pareto_sonuclari' in st.session_state:
        varsayilan_J = [str(x).strip().title() for x in st.session_state.pareto_sonuclari['İl'].tolist()]
    else:
        varsayilan_J = ["İstanbul", "Ankara", "İzmir", "Antalya", "Bursa", "Kocaeli", "Konya", "Adana", "Kayseri", "Gaziantep", "Mersin"]

    if 'kmeans_sonuclari' in st.session_state:
        varsayilan_K = [x['İl'] for x in st.session_state.kmeans_sonuclari]
    else:
        varsayilan_K = ["Bursa", "Adana", "Konya"]

    varsayilan_L = ["Konya", "Adana"]
    varsayilan_M = ["Adana", "Konya"]

    ev_data = VARSAYILAN_EV_VERISI.copy()
    if st.session_state.mod1_data is not None:
        for _, r in st.session_state.mod1_data.iterrows():
            il = str(r['İl']).strip().title()
            if il in IL_KOORDINATLARI:
                ev_data[il] = float(r['EV_Sayısı'])

    st.subheader("1. Senaryo Parametrelerini Yapılandırın (What-If Analizi)")
    senaryo_adi = st.text_input("Senaryo Adı", value=f"Senaryo {len(st.session_state.senaryo_gecmisi) + 1}")

    with st.expander("A. Talep, Geri Toplama ve Batarya Kalite (SoH) Oranları", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        gamma = c1.slider("Geri Toplama Oranı γ (%)", 5, 100, 40, 5)
        binek_payi = c2.slider("Binek Araç Payı (%)", 50, 99, 95, 1)
        R1 = c3.number_input("Geri Dönüşüm Oranı R1 (%)", 0.0, 100.0, 62.0, 1.0)
        R2 = c4.number_input("İkinci Ömür Oranı R2 (%)", 0.0, 100.0, 15.0, 1.0)
        R3 = round(100.0 - (R1 + R2), 2)
        c5.metric("Bertaraf Oranı R3 (%)", f"%{R3}", delta="Otomatik", delta_color="off")

    with st.expander("B. Tesis Kapasiteleri ve Sabit İnşa Maliyetleri", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        cap_j = k1.number_input("Toplama Kapasitesi", 1000, 500000, 100000, 5000)
        cap_k = k2.number_input("İşleme Kapasitesi", 1000, 500000, 100000, 5000)
        cap_l = k3.number_input("İkinci Ömür Kapasitesi (Ton)", 100, 100000, 15000, 500)
        cap_m = k4.number_input("Geri Dönüşüm Kapasitesi (Ton)", 100, 100000, 30000, 1000)

        m1, m2, m3, m4 = st.columns(4)
        fj = m1.number_input("Toplama İnşa ($)", 0, 5000000, 250000, 25000)
        fk = m2.number_input("İşleme İnşa ($)", 0, 5000000, 400000, 25000)
        fl = m3.number_input("İkinci Ömür İnşa ($)", 0, 5000000, 650000, 25000)
        fm = m4.number_input("Geri Dönüşüm İnşa ($)", 0, 5000000, 450000, 25000)

    with st.expander("C. Birim Taşıma, İşlem Maliyetleri ve Satış Fiyatları", expanded=False):
        t1, t2, t3, t4, t5 = st.columns(5)
        c3_tasima = t1.number_input("Batarya Taşıma C3", 0.0, 50.0, 8.0, 0.5)
        c4_tasima = t2.number_input("Hücre Taşıma C4", 0.0, 5.0, 0.08, 0.01)
        c1_binek = t3.number_input("İşleme Binek C1", 50.0, 5000.0, 350.0, 50.0)
        c1_agir = t4.number_input("İşleme Ağır C1", 100.0, 10000.0, 3200.0, 100.0)
        c5_bertaraf = t5.number_input("Bertaraf C5", 0.0, 100.0, 5.0, 1.0)

        g1, g2, g3, g4 = st.columns(4)
        rev_kobalt = g1.number_input("Kobalt ($/kg)", 1.0, 150.0, 50.0, 5.0)
        rev_lityum = g2.number_input("Lityum ($/kg)", 0.1, 50.0, 1.5, 0.2)
        rev_nikel = g3.number_input("Nikel ($/kg)", 1.0, 50.0, 12.0, 1.0)
        hucre_bazli_cc = g4.checkbox("Hücre Bazlı Toplama Maliyeti", value=False)

    with st.expander("D. Aday Tesis Lokasyon Kümeleri (J, K, L, M)", expanded=False):
        tum_iller = sorted(list(IL_KOORDINATLARI.keys()))
        aday_J = st.multiselect("Aday Toplama (J)", options=tum_iller, default=[x for x in varsayilan_J if x in tum_iller])
        aday_K = st.multiselect("Aday İşleme (K)", options=tum_iller, default=[x for x in varsayilan_K if x in tum_iller])
        aday_L = st.multiselect("Aday İkinci Ömür (L)", options=tum_iller, default=[x for x in varsayilan_L if x in tum_iller])
        aday_M = st.multiselect("Aday Geri Dönüşüm (M)", options=tum_iller, default=[x for x in varsayilan_M if x in tum_iller])

    if st.button("🚀 Optimizasyon Modelini (MILP) Çalıştır", type="primary", use_container_width=True):
        if R3 < 0 or not (aday_J and aday_K and aday_L and aday_M):
            st.error("Lütfen R1+R2 <= 100 olduğundan ve her tesis için en az 1 il seçildiğinden emin olun.")
        else:
            with st.spinner("MILP modeli CBC çözücüsü ile çözülüyor..."):
                params = {
                    'ev_data': ev_data, 'gamma': gamma, 'binek_payi': binek_payi,
                    'R1': R1, 'R2': R2, 'R3': R3,
                    'cap_j': cap_j, 'cap_k': cap_k, 'cap_l': cap_l, 'cap_m': cap_m,
                    'fj': fj, 'fk': fk, 'fl': fl, 'fm': fm,
                    'c3_tasima': c3_tasima, 'c4_tasima': c4_tasima,
                    'c1_binek': c1_binek, 'c1_agir': c1_agir, 'c5_bertaraf': c5_bertaraf,
                    'rev_kobalt': rev_kobalt, 'rev_lityum': rev_lityum, 'rev_nikel': rev_nikel,
                    'hucre_bazli_cc': hucre_bazli_cc,
                    'aday_J': aday_J, 'aday_K': aday_K, 'aday_L': aday_L, 'aday_M': aday_M
                }
                sonuc = solve_milp_model(params)
                st.session_state.milp_sonuc = sonuc

                if sonuc['durum'] == 'Optimal':
                    st.session_state.senaryo_gecmisi.append({
                        "Senaryo": senaryo_adi,
                        "Durum": sonuc['durum'],
                        "Net Toplam Maliyet ($)": round(sonuc['amac_degeri'], 2),
                        "Açılan Toplama": sum(sonuc['x_j'].values()),
                        "Açılan İşleme": sum(sonuc['x_k'].values()),
                        "Açılan İkinci Ömür": sum(sonuc['x_l'].values()),
                        "Açılan Geri Dönüşüm": sum(sonuc['x_m'].values())
                    })

    if 'milp_sonuc' in st.session_state:
        res = st.session_state.milp_sonuc
        st.divider()
        if res['durum'] != 'Optimal':
            st.error(f"Model uygun çözüm bulamadı! Durum: **{res['durum']}**.")
        else:
            st.subheader("2. Optimizasyon Sonuçları ve Maliyet Dağılımı")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Çözüm Durumu", res['durum'])
            k2.metric("Net Toplam Maliyet", f"${res['amac_degeri']:,.0f}")
            k3.metric("Açılan Toplama", f"{sum(res['x_j'].values())}")
            k4.metric("Açılan İşleme", f"{sum(res['x_k'].values())}")
            k5.metric("Açılan İÖM & GDM", f"{sum(res['x_l'].values())} & {sum(res['x_m'].values())}")

            col_g, col_t = st.columns([1.4, 1])
            with col_g:
                fig_cost, ax_cost = plt.subplots(figsize=(7, 3.8))
                kalemler = list(res['maliyet_kalemleri'].keys())
                degerler = list(res['maliyet_kalemleri'].values())
                renkler = ['#2F70AF' if v >= 0 else '#2E8B57' for v in degerler]
                ax_cost.barh(kalemler, degerler, color=renkler)
                ax_cost.axvline(0, color='black', linewidth=0.8)
                ax_cost.set_title("Amaç Fonksiyonu Bileşenleri ($)")
                ax_cost.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig_cost)
            with col_t:
                df_cost = pd.DataFrame({
                    "Kalem": list(res['maliyet_kalemleri'].keys()),
                    "Tutar": [f"${v:,.2f}" for v in res['maliyet_kalemleri'].values()]
                })
                st.dataframe(df_cost, use_container_width=True, hide_index=True)

            st.subheader("3. Karar Değişkenleri Detaylı Sonuçları")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "1️⃣ Tesis Açma (x)", "2️⃣ İl Atamaları (z)", "3️⃣ İşleme Yükü (β)",
                "4️⃣ Geri Dönüşüm (θ)", "5️⃣ İkinci Ömür & Bertaraf", "6️⃣ Harita & Senaryo"
            ])

            with tab1:
                c_j, c_k, c_l, c_m = st.columns(4)
                with c_j: st.dataframe(pd.DataFrame(list(res['x_j'].items()), columns=["Merkez", "x_j"]), use_container_width=True, hide_index=True)
                with c_k: st.dataframe(pd.DataFrame(list(res['x_k'].items()), columns=["Merkez", "x_k"]), use_container_width=True, hide_index=True)
                with c_l: st.dataframe(pd.DataFrame(list(res['x_l'].items()), columns=["Merkez", "x_l"]), use_container_width=True, hide_index=True)
                with c_m: st.dataframe(pd.DataFrame(list(res['x_m'].items()), columns=["Merkez", "x_m"]), use_container_width=True, hide_index=True)

            with tab2:
                if res['z_ijb']: st.dataframe(pd.DataFrame(res['z_ijb']), use_container_width=True, hide_index=True)

            with tab3:
                if res['beta_kb']: st.dataframe(pd.DataFrame(res['beta_kb']), use_container_width=True, hide_index=True)

            with tab4:
                if res['theta_mn']: st.dataframe(pd.DataFrame(res['theta_mn']), use_container_width=True, hide_index=True)

            with tab5:
                if res['W_klb']: st.dataframe(pd.DataFrame(res['W_klb']), use_container_width=True, hide_index=True)

            with tab6:
                m_opt = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
                for j, acik in res['x_j'].items():
                    if acik == 1: folium.Marker(IL_KOORDINATLARI[j], tooltip=f"Toplama: {j}", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_opt)
                for k, acik in res['x_k'].items():
                    if acik == 1: folium.Marker(IL_KOORDINATLARI[k], tooltip=f"İşleme: {k}", icon=folium.Icon(color="red", icon="cog")).add_to(m_opt)
                for l, acik in res['x_l'].items():
                    if acik == 1: folium.Marker(IL_KOORDINATLARI[l], tooltip=f"İkinci Ömür: {l}", icon=folium.Icon(color="orange", icon="flash")).add_to(m_opt)
                for m, acik in res['x_m'].items():
                    if acik == 1: folium.Marker(IL_KOORDINATLARI[m], tooltip=f"Geri Dönüşüm: {m}", icon=folium.Icon(color="green", icon="recycle")).add_to(m_opt)
                st_folium(m_opt, width=900, height=420, key="map_opt_final")
                if st.session_state.senaryo_gecmisi:
                    st.dataframe(pd.DataFrame(st.session_state.senaryo_gecmisi), use_container_width=True, hide_index=True)

# ==========================================
# ANA KONTROL MEKANİZMASI
# ==========================================
if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.current_view == 'dashboard':
        show_dashboard()
    elif st.session_state.current_view == 'module1':
        show_module1()
    elif st.session_state.current_view == 'module2':
        show_module2()
