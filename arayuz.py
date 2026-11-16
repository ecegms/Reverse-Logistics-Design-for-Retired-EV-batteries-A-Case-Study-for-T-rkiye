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
# TÜRKİYE 81 İL KOORDİNATLARI VE VARSAYILAN EK-1 VERİSİ
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

# Rapor EK-1 Tablosundaki Varsayılan 81 İl EV Verileri (Modül 1 çalıştırılmazsa doğrudan kullanılabilir)
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
    """İki koordinat arası kuş uçuşu mesafe (km)"""
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
    st.markdown("<h1 style='text-align: center;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 18px; max-width: 900px; margin: 0 auto; line-height: 1.6; opacity: 0.8;'>
    Bu Karar Destek Sistemi, ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde toplanması ve geri kazanılması için tersine lojistik ağını tasarlamak amacıyla geliştirilmiştir.
    </p><br><br>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("MODÜL 1: ADAY MERKEZ BELİRLEME\n\nPareto, K-Means ve Minimax algoritmaları ile aday tesis konumlarını belirleyin.", key="btn_mod1", type="primary", use_container_width=True):
            st.session_state.current_view = 'module1'
            st.rerun()
    with col2:
        if st.button("MODÜL 2: OPTİMİZASYON VE SENARYO ANALİZİ\n\nParametreleri değiştirerek MILP modelini çalıştırın ve tüm karar değişkenlerini analiz edin.", key="btn_mod2", type="primary", use_container_width=True):
            st.session_state.current_view = 'module2'
            st.rerun()
            
    st.write("<br>", unsafe_allow_html=True)
    _, col_logout, _ = st.columns([4, 2, 4])
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
        st.success("Aday Toplama (j), İşleme (k), İkinci Ömür (l) ve Geri Dönüşüm (m) merkezleri belirlendi. Modül 2'ye geçebilirsiniz.")
        if st.button("Modül 2: Optimizasyon Modelini Çöz ➡", type="primary"):
            st.session_state.current_view = 'module2'
            st.rerun()

# ==========================================
# 5. MODÜL 2: MILP OPTİMİZASYON VE SENARYO ANALİZİ
# ==========================================
def solve_milp_model(params):
    """
    Rapor Bölüm 4 (Eşitlik 1 - 37) MILP Matematiksel Modeli Çözücüsü
    """
    # 1. Kümeler
    ev_dict = params['ev_data']
    I = list(ev_dict.keys())  # 81 il
    J = params['aday_J']      # Aday Toplama Merkezleri
    K = params['aday_K']      # Aday İşleme Merkezleri
    L = params['aday_L']      # Aday İkinci Ömür Merkezleri
    M = params['aday_M']      # Aday Geri Dönüşüm Merkezleri
    B = ['Binek', 'Agir_Vasita']
    N = ['Grafit', 'Aluminyum', 'Bakir', 'Kobalt', 'Lityum', 'Manganez', 'Nikel']

    # 2. Parametreler
    binek_orani = params['binek_payi'] / 100.0
    agir_orani = 1.0 - binek_orani
    gamma = params['gamma'] / 100.0  # Geri toplama oranı

    # Talep DE_{ib}
    DE = {}
    for i in I:
        toplam_arac = ev_dict[i]
        DE[(i, 'Binek')] = toplam_arac * binek_orani
        DE[(i, 'Agir_Vasita')] = toplam_arac * agir_orani

    # Batarya teknik parametreleri (Tablo 2.6)
    H = {'Binek': 96, 'Agir_Vasita': 1008}
    C = {'Binek': 3.5, 'Agir_Vasita': 0.0395}

    # SoH Yönlendirme Oranları (Sayfa 28)
    R1 = params['R1'] / 100.0  # Geri dönüşüm (%62)
    R2 = params['R2'] / 100.0  # İkinci ömür (%15)
    R3 = params['R3'] / 100.0  # Bertaraf (%23)

    # İkinci ömür alt oranları (Sayfa 28)
    R21 = 0.2047  # Farklı amaçlarla yeniden kullanım
    R22 = 0.7019  # Yeniden imalat
    R23 = 0.0934  # Doğrudan yeniden kullanım

    # Materyal oranları ve geri kazanım verimlilikleri (Tablo 2.7)
    CR = {'Grafit': 0.16, 'Aluminyum': 0.15, 'Bakir': 0.10, 'Kobalt': 0.07, 'Lityum': 0.07, 'Manganez': 0.05, 'Nikel': 0.04}
    RR = {'Grafit': 0.945, 'Aluminyum': 0.955, 'Bakir': 0.98, 'Kobalt': 0.965, 'Lityum': 0.98, 'Manganez': 0.99, 'Nikel': 0.985}

    # Birim satış gelirleri REV_n ($/kg) (Tablo 2.8)
    REV = {
        'Grafit': 1.2, 'Aluminyum': 2.4, 'Bakir': 8.5,
        'Kobalt': params['rev_kobalt'], 'Lityum': params['rev_lityum'],
        'Manganez': 1.8, 'Nikel': params['rev_nikel']
    }

    # Maliyet Parametreleri (Sayfa 29 & Tablo 2.10)
    CC = {'Binek': 8.3 * 96, 'Agir_Vasita': 8.3 * 1008} if params['hucre_bazli_cc'] else {'Binek': 120.0, 'Agir_Vasita': 250.0}
    C1 = {'Binek': params['c1_binek'], 'Agir_Vasita': params['c1_agir']}
    C21, C22, C23 = 62.3, 20.5, 0.215
    C3 = params['c3_tasima']   # Batarya başına taşıma ($/km)
    C4 = params['c4_tasima']   # Hücre başına taşıma ($/km)
    C5 = params['c5_bertaraf'] # Bertaraf maliyeti ($/hücre)
    REC = {'Binek': 5.085, 'Agir_Vasita': 450.0}

    # Sabit inşa maliyetleri ($) (Tablo 2.10)
    fj = params['fj']
    fk = params['fk']
    fl = params['fl']
    fm = params['fm']

    # Kapasiteler (Tablo 2.9)
    Cap_jb = {'Binek': params['cap_j'], 'Agir_Vasita': params['cap_j']}
    Cap_kb = {'Binek': params['cap_k'], 'Agir_Vasita': params['cap_k']}
    Cap_l = params['cap_l'] * 1000.0   # Ton -> kg
    Cap_mb = {'Binek': params['cap_m'] * 1000.0, 'Agir_Vasita': params['cap_m'] * 1000.0}

    # Mesafe Matrisleri
    d_ij = {(i, j): haversine_km(*IL_KOORDINATLARI[i], *IL_KOORDINATLARI[j]) for i in I for j in J}
    d_jk = {(j, k): haversine_km(*IL_KOORDINATLARI[j], *IL_KOORDINATLARI[k]) for j in J for k in K}
    d_kl = {(k, l): haversine_km(*IL_KOORDINATLARI[k], *IL_KOORDINATLARI[l]) for k in K for l in L}
    d_km = {(k, m): haversine_km(*IL_KOORDINATLARI[k], *IL_KOORDINATLARI[m]) for k in K for m in M}

    # 3. MILP Modeli Kurulumu
    model = pulp.LpProblem("EV_Batarya_Tersine_Lojistik_MILP", pulp.LpMinimize)

    # Karar Değişkenleri (Eş. 28 - 37)
    x_ij = pulp.LpVariable.dicts("x_ij", [(i, j) for i in I for j in J], cat=pulp.LpBinary)
    x_j = pulp.LpVariable.dicts("x_j", J, cat=pulp.LpBinary)
    x_k = pulp.LpVariable.dicts("x_k", K, cat=pulp.LpBinary)
    x_l = pulp.LpVariable.dicts("x_l", L, cat=pulp.LpBinary)
    x_m = pulp.LpVariable.dicts("x_m", M, cat=pulp.LpBinary)

    z_ijb = pulp.LpVariable.dicts("z_ijb", [(i, j, b) for i in I for j in J for b in B], lowBound=0)
    W_jkb = pulp.LpVariable.dicts("W_jkb", [(j, k, b) for j in J for k in K for b in B], lowBound=0)
    beta_kb = pulp.LpVariable.dicts("beta_kb", [(k, b) for k in K for b in B], lowBound=0)
    W_kmb = pulp.LpVariable.dicts("W_kmb", [(k, m, b) for k in K for m in M for b in B], lowBound=0)
    W_klb = pulp.LpVariable.dicts("W_klb", [(k, l, b) for k in K for l in L for b in B], lowBound=0)
    W_kb = pulp.LpVariable.dicts("W_kb", [(k, b) for k in K for b in B], lowBound=0)

    Wrep_lb = pulp.LpVariable.dicts("Wrep_lb", [(l, b) for l in L for b in B], lowBound=0)
    Wrem_lb = pulp.LpVariable.dicts("Wrem_lb", [(l, b) for l in L for b in B], lowBound=0)
    Wreu_lb = pulp.LpVariable.dicts("Wreu_lb", [(l, b) for l in L for b in B], lowBound=0)

    theta_lb = pulp.LpVariable.dicts("theta_lb", [(l, b) for l in L for b in B], lowBound=0)
    theta_mb = pulp.LpVariable.dicts("theta_mb", [(m, b) for m in M for b in B], lowBound=0)
    theta_mn = pulp.LpVariable.dicts("theta_mn", [(m, n) for m in M for n in N], lowBound=0)

    # Amaç Fonksiyonu Bileşenleri (Eş. 1 - 10)
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

    # Kısıtlar (Eş. 11 - 27)
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

    # Çözüm
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

    # Modül 1'den gelen veya raporun varsayılan aday merkezlerini hazırla
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

    # ==========================================
    # SENARYO PARAMETRE KONTROL PANELİ
    # ==========================================
    st.subheader("1. Senaryo Parametrelerini Yapılandırın (What-If Analizi)")
    st.caption("Aşağıdaki parametreleri değiştirip 'Optimizasyon Modelini Çalıştır' butonuna basarak farklı senaryoları test edebilirsiniz.")

    senaryo_adi = st.text_input("Senaryo Adı (Karşılaştırma tablosunda görünür)", value=f"Senaryo {len(st.session_state.senaryo_gecmisi) + 1}")

    with st.expander("A. Talep, Geri Toplama ve Batarya Kalite (SoH) Oranları", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        gamma = c1.slider("Geri Toplama Oranı γ (%)", min_value=5, max_value=100, value=40, step=5)
        binek_payi = c2.slider("Binek Araç Payı (%)", min_value=50, max_value=99, value=95, step=1)
        R1 = c3.number_input("Geri Dönüşüm Oranı R1 (%)", min_value=0.0, max_value=100.0, value=62.0, step=1.0)
        R2 = c4.number_input("İkinci Ömür Oranı R2 (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        R3 = round(100.0 - (R1 + R2), 2)
        c5.metric("Bertaraf Oranı R3 (%)", f"%{R3}", delta="Otomatik (100 - R1 - R2)", delta_color="off")
        if R3 < 0:
            st.error("R1 + R2 toplamı %100'ü geçemez!")

    with st.expander("B. Tesis Kapasiteleri ve Sabit İnşa Maliyetleri", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        cap_j = k1.number_input("Toplama Merkezi Kapasitesi (Adet)", min_value=1000, max_value=500000, value=100000, step=5000)
        cap_k = k2.number_input("İşleme Merkezi Kapasitesi (Adet)", min_value=1000, max_value=500000, value=100000, step=5000)
        cap_l = k3.number_input("İkinci Ömür Kapasitesi (Ton)", min_value=100, max_value=100000, value=15000, step=500)
        cap_m = k4.number_input("Geri Dönüşüm Kapasitesi (Ton)", min_value=100, max_value=100000, value=30000, step=1000)

        m1, m2, m3, m4 = st.columns(4)
        fj = m1.number_input("Toplama Merkezi İnşa ($)", min_value=0, max_value=5000000, value=250000, step=25000)
        fk = m2.number_input("İşleme Merkezi İnşa ($)", min_value=0, max_value=5000000, value=400000, step=25000)
        fl = m3.number_input("İkinci Ömür İnşa ($)", min_value=0, max_value=5000000, value=650000, step=25000)
        fm = m4.number_input("Geri Dönüşüm İnşa ($)", min_value=0, max_value=5000000, value=450000, step=25000)

    with st.expander("C. Birim Taşıma, İşlem Maliyetleri ve Materyal Satış Fiyatları ($)", expanded=False):
        t1, t2, t3, t4, t5 = st.columns(5)
        c3_tasima = t1.number_input("Batarya Taşıma C3 ($/km)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
        c4_tasima = t2.number_input("Hücre Taşıma C4 ($/km)", min_value=0.0, max_value=5.0, value=0.08, step=0.01)
        c1_binek = t3.number_input("İşleme Binek C1 ($/batarya)", min_value=50.0, max_value=5000.0, value=350.0, step=50.0)
        c1_agir = t4.number_input("İşleme Ağır C1 ($/batarya)", min_value=100.0, max_value=10000.0, value=3200.0, step=100.0)
        c5_bertaraf = t5.number_input("Bertaraf C5 ($/hücre)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)

        g1, g2, g3, g4 = st.columns(4)
        rev_kobalt = g1.number_input("Kobalt Satış Fiyatı ($/kg)", min_value=1.0, max_value=150.0, value=50.0, step=5.0)
        rev_lityum = g2.number_input("Lityum Karbonat ($/kg)", min_value=0.1, max_value=50.0, value=1.5, step=0.2)
        rev_nikel = g3.number_input("Nikel Satış Fiyatı ($/kg)", min_value=1.0, max_value=50.0, value=12.0, step=1.0)
        hucre_bazli_cc = g4.checkbox("Toplama Maliyetini Hücre Bazlı (8.3$/hücre) Hesapla", value=False)

    with st.expander("D. Aday Tesis Lokasyon Kümeleri (J, K, L, M)", expanded=False):
        tum_iller = sorted(list(IL_KOORDINATLARI.keys()))
        aday_J = st.multiselect("Aday Toplama Merkezleri (J)", options=tum_iller, default=[x for x in varsayilan_J if x in tum_iller])
        aday_K = st.multiselect("Aday İşleme Merkezleri (K)", options=tum_iller, default=[x for x in varsayilan_K if x in tum_iller])
        aday_L = st.multiselect("Aday İkinci Ömür Merkezleri (L)", options=tum_iller, default=[x for x in varsayilan_L if x in tum_iller])
        aday_M = st.multiselect("Aday Geri Dönüşüm Merkezleri (M)", options=tum_iller, default=[x for x in varsayilan_M if x in tum_iller])

    # ÇÖZÜM BUTONU
    if st.button("🚀 Optimizasyon Modelini (MILP) Çalıştır", type="primary", use_container_width=True):
        if R3 < 0 or not (aday_J and aday_K and aday_L and aday_M):
            st.error("Lütfen R1+R2 <= 100 olduğundan ve her tesis tipi için en az 1 aday il seçildiğinden emin olun.")
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
                        "Toplama Oranı (γ)": f"%{gamma}",
                        "Net Toplam Maliyet ($)": round(sonuc['amac_degeri'], 2),
                        "Açılan Toplama (Σ x_j)": sum(sonuc['x_j'].values()),
                        "Açılan İşleme (Σ x_k)": sum(sonuc['x_k'].values()),
                        "Açılan İkinci Ömür (Σ x_l)": sum(sonuc['x_l'].values()),
                        "Açılan Geri Dönüşüm (Σ x_m)": sum(sonuc['x_m'].values()),
                        "Geri Kazanım Geliri ($)": round(-sonuc['maliyet_kalemleri']["Materyal Satış Geliri (-)"], 2)
                    })

    # ==========================================
    # SONUÇLAR VE KARAR DEĞİŞKENLERİ GÖRSELLEŞTİRME
    # ==========================================
    if 'milp_sonuc' in st.session_state:
        res = st.session_state.milp_sonuc
        st.divider()

        if res['durum'] != 'Optimal':
            st.error(f"Model uygun bir çözüm bulamadı! Çözüm Durumu: **{res['durum']}**. Kapasite değerlerini artırarak tekrar deneyin.")
        else:
            st.subheader("2. Optimizasyon Sonuçları ve Maliyet Dağılımı")

            # Üst Metrik Kartları
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Çözüm Durumu", res['durum'])
            k2.metric("Net Toplam Maliyet (Z)", f"${res['amac_degeri']:,.0f}")
            k3.metric("Açılan Toplama (x_j)", f"{sum(res['x_j'].values())} / {len(res['x_j'])}")
            k4.metric("Açılan İşleme (x_k)", f"{sum(res['x_k'].values())} / {len(res['x_k'])}")
            k5.metric("Açılan İÖM (x_l) & GDM (x_m)", f"{sum(res['x_l'].values())} & {sum(res['x_m'].values())}")

            # Maliyet Dağılım Grafiği ve Tablosu
            col_g, col_t = st.columns([1.4, 1])
            with col_g:
                fig_cost, ax_cost = plt.subplots(figsize=(7, 3.8))
                kalemler = list(res['maliyet_kalemleri'].keys())
                degerler = list(res['maliyet_kalemleri'].values())
                renkler = ['#2F70AF' if v >= 0 else '#2E8B57' for v in degerler]
                ax_cost.barh(kalemler, degerler, color=renkler)
                ax_cost.axvline(0, color='black', linewidth=0.8)
                ax_cost.set_title("Amaç Fonksiyonu Bileşenleri ($)")
                ax_cost.set_xlabel("Tutar ($)")
                ax_cost.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig_cost)
            with col_t:
                df_cost = pd.DataFrame({
                    "Maliyet / Gelir Kalemi": list(res['maliyet_kalemleri'].keys()),
                    "Tutar ($)": [f"${v:,.2f}" for v in res['maliyet_kalemleri'].values()]
                })
                st.dataframe(df_cost, use_container_width=True, hide_index=True)

            # ==========================================
            # HER BİR KARAR DEĞİŞKENİNİN SONUÇLARI (SEKMELİ)
            # ==========================================
            st.subheader("3. Karar Değişkenleri Detaylı Sonuçları")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "1️⃣ Tesis Açma (x_j, x_k, x_l, x_m)",
                "2️⃣ İl Atamaları (x_ij & z_ijb)",
                "3️⃣ Toplama → İşleme (W_jkb & β_kb)",
                "4️⃣ İşleme → Geri Dönüşüm (W_kmb & θ)",
                "5️⃣ İşleme → İkinci Ömür (W_klb) & Bertaraf (W_kb)",
                "6️⃣ Harita Görünümü & Senaryo Karşılaştırma"
            ])

            # SEKME 1: İKİLİ TESİS AÇMA KARAR DEĞİŞKENLERİ
            with tab1:
                st.write("**Tesis Açma İkili Karar Değişkenleri (1 = Açıldı, 0 = Açılmadı)**")
                c_j, c_k, c_l, c_m = st.columns(4)
                with c_j:
                    st.markdown("**Toplama Merkezleri ($x_j$)**")
                    st.dataframe(pd.DataFrame(list(res['x_j'].items()), columns=["Merkez (j)", "x_j"]), use_container_width=True, hide_index=True)
                with c_k:
                    st.markdown("**İşleme Merkezleri ($x_k$)**")
                    st.dataframe(pd.DataFrame(list(res['x_k'].items()), columns=["Merkez (k)", "x_k"]), use_container_width=True, hide_index=True)
                with c_l:
                    st.markdown("**İkinci Ömür Merkezleri ($x_l$)**")
                    st.dataframe(pd.DataFrame(list(res['x_l'].items()), columns=["Merkez (l)", "x_l"]), use_container_width=True, hide_index=True)
                with c_m:
                    st.markdown("**Geri Dönüşüm Merkezleri ($x_m$)**")
                    st.dataframe(pd.DataFrame(list(res['x_m'].items()), columns=["Merkez (m)", "x_m"]), use_container_width=True, hide_index=True)

            # SEKME 2: x_ij ve z_ijb
            with tab2:
                st.write("**İllerin Toplama Merkezlerine Atanması ($x_{ij}=1$) ve Batarya Akış Miktarları ($z_{ijb}$)**")
                df_z = pd.DataFrame(res['z_ijb'])
                if not df_z.empty:
                    col_z1, col_z2 = st.columns([1.3, 1])
                    with col_z1:
                        st.dataframe(df_z, use_container_width=True, hide_index=True)
                    with col_z2:
                        ozet_j = df_z.groupby(["Toplama Merkezi (j)", "Batarya Tipi (b)"])["Miktar (Adet)"].sum().unstack(fill_value=0)
                        fig_z, ax_z = plt.subplots(figsize=(5, 4))
                        ozet_j.plot(kind='bar', ax=ax_z, color=['#2F70AF', '#E67E22'])
                        ax_z.set_title("Toplama Merkezlerine Gelen Batarya ($z_{ijb}$)")
                        ax_z.set_ylabel("Batarya Adedi")
                        plt.xticks(rotation=45)
                        st.pyplot(fig_z)

            # SEKME 3: W_jkb ve beta_kb
            with tab3:
                st.write("**Toplama Merkezlerinden İşleme Merkezlerine Akış ($W_{jkb}$) ve İşleme Merkezi İş Yükü ($\\beta_{kb}$)**")
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    st.markdown("**$W_{jkb}$ Akış Tablosu**")
                    st.dataframe(pd.DataFrame(res['W_jkb']), use_container_width=True, hide_index=True)
                with col_w2:
                    st.markdown("**$\\beta_{kb}$ İşleme Merkezi Toplam Yükü**")
                    df_beta = pd.DataFrame(res['beta_kb'])
                    st.dataframe(df_beta, use_container_width=True, hide_index=True)
                    if not df_beta.empty:
                        fig_b, ax_b = plt.subplots(figsize=(5, 3))
                        df_beta.pivot(index="İşleme Merkezi (k)", columns="Batarya Tipi (b)", values="Toplam Batarya (Adet)").plot(kind='bar', ax=ax_b)
                        ax_b.set_title("İşleme Merkezleri Yükü ($\\beta_{kb}$)")
                        plt.xticks(rotation=0)
                        st.pyplot(fig_b)

            # SEKME 4: W_kmb, theta_mb, theta_mn
            with tab4:
                st.write("**Geri Dönüşüme Giden Hücreler ($W_{kmb}$), Hücre Ağırlıkları ($\\theta_{mb}$) ve Kazanılan Materyaller ($\\theta_{mn}$)**")
                c_gd1, c_gd2 = st.columns(2)
                with c_gd1:
                    st.markdown("**$W_{kmb}$ Hücre Akışları**")
                    st.dataframe(pd.DataFrame(res['W_kmb']), use_container_width=True, hide_index=True)
                    st.markdown("**$\\theta_{mb}$ Geri Dönüşüm Merkezi Giriş Ağırlıkları (kg)**")
                    st.dataframe(pd.DataFrame(res['theta_mb']), use_container_width=True, hide_index=True)
                with c_gd2:
                    st.markdown("**$\\theta_{mn}$ Geri Kazanılan Materyal Miktarları (kg) ve Gelir**")
                    df_mn = pd.DataFrame(res['theta_mn'])
                    st.dataframe(df_mn, use_container_width=True, hide_index=True)
                    if not df_mn.empty:
                        fig_mn, ax_mn = plt.subplots(figsize=(5, 3.2))
                        df_mn.groupby("Materyal (n)")["Geri Kazanılan Miktar (kg)"].sum().plot(kind='bar', ax=ax_mn, color='#2E8B57')
                        ax_mn.set_title("Geri Kazanılan Materyal Miktarları ($\\theta_{mn}$)")
                        ax_mn.set_ylabel("Ağırlık (kg)")
                        plt.xticks(rotation=45)
                        st.pyplot(fig_mn)

            # SEKME 5: W_klb, Wrep, Wrem, Wreu ve W_kb (Bertaraf)
            with tab5:
                st.write("**İkinci Ömür Merkezi Akışları ($W_{klb}, Wrep_{lb}, Wrem_{lb}, Wreu_{lb}$) ve Bertaraf ($W_{kb}$)**")
                c_io1, c_io2 = st.columns([1.5, 1])
                with c_io1:
                    st.markdown("**İkinci Ömür Alt Süreçleri**")
                    st.dataframe(pd.DataFrame(res['W_klb']), use_container_width=True, hide_index=True)
                with c_io2:
                    st.markdown("**Bertarafa Giden Hücreler ($W_{kb}$)**")
                    st.dataframe(pd.DataFrame(res['W_kb']), use_container_width=True, hide_index=True)

            # SEKME 6: HARİTA VE SENARYO KARŞILAŞTIRMA
            with tab6:
                st.markdown("**Optimize Edilen Tersine Lojistik Ağı Haritası**")
                m_opt = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
                for j, acik in res['x_j'].items():
                    if acik == 1:
                        folium.Marker(IL_KOORDINATLARI[j], tooltip=f"Açılan Toplama Merkezi: {j}", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_opt)
                for k, acik in res['x_k'].items():
                    if acik == 1:
                        folium.Marker(IL_KOORDINATLARI[k], tooltip=f"Açılan İşleme Merkezi: {k}", icon=folium.Icon(color="red", icon="cog")).add_to(m_opt)
                for l, acik in res['x_l'].items():
                    if acik == 1:
                        folium.Marker(IL_KOORDINATLARI[l], tooltip=f"Açılan İkinci Ömür Merkezi: {l}", icon=folium.Icon(color="orange", icon="flash")).add_to(m_opt)
                for m, acik in res['x_m'].items():
                    if acik == 1:
                        folium.Marker(IL_KOORDINATLARI[m], tooltip=f"Açılan Geri Dönüşüm Merkezi: {m}", icon=folium.Icon(color="green", icon="recycle")).add_to(m_opt)
                st_folium(m_opt, width=900, height=420, key="map_opt_final")

                st.markdown("**Çalıştırılan Senaryoların Karşılaştırma Tablosu**")
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
