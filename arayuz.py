# KODUN EN ÜSTÜNE EKLENECEK YENİ IMPORTLAR:
from sklearn.cluster import KMeans
import numpy as np
import folium
from streamlit_folium import st_folium
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt # Dirsek Grafiği için eklendi

# Sayfa ayarları (Her zaman en üstte olmalı)
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# GÖRSEL TASARIM (CSS INJECTION)
# ==========================================
custom_css = """
<style>
/* 
Kullanıcının ilettiği paletteki güvenli ve kurumsal renkler kullanıldı. 
Primary butonlar devasa, uzun dikdörtgenlere çevrildi.
*/
button[data-testid="baseButton-primary"] {
    height: 400px !important; /* Boyu aşağıya doğru iyice uzatıldı */
    width: 100% !important;
    background-color: #2F70AF !important; /* Paletteki Orta Mavi */
    color: #ffffff !important; 
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    padding: 30px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Farenin modül butonunun üzerine gelme (Hover) efekti */
button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 15px 30px rgba(47, 112, 175, 0.4) !important;
    background-color: #1A3E62 !important; /* Paletteki Koyu Lacivert/Mavi */
    border: none !important;
}

/* DİKKAT: Metinlerin yan yana uzayıp 3 nokta (...) olmasını kesin olarak engelleyen kısım */
button[data-testid="baseButton-primary"] div, 
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] * {
    white-space: pre-wrap !important; /* Kesinlikle alt satıra inmeye zorlar */
    word-wrap: break-word !important;
    text-align: center !important;
    font-size: 20px !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    font-family: 'Georgia', serif !important;
    color: #ffffff !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 1. SESSION STATE (Hafıza) YÖNETİMİ
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'mod1_data' not in st.session_state:
    st.session_state.mod1_data = None
if 'mod1_step' not in st.session_state:
    st.session_state.mod1_step = 0 

def reset_system():
    st.session_state.mod1_data = None
    st.session_state.mod1_step = 0
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
# 3. ANA PANEL (Dashboard - Mavi Paletli Dev Modüller)
# ==========================================
def show_dashboard():
    st.markdown("<h1 style='text-align: center;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; font-size: 18px; max-width: 900px; margin: 0 auto; line-height: 1.6; opacity: 0.8;'>
    Bu Karar Destek Sistemi, ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde toplanması ve geri kazanılması için tersine lojistik ağını tasarlamak amacıyla geliştirilmiştir.
    </p>
    <br><br>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satır atlamaları \n\n ile korundu, yeni CSS sayesinde artık tek satırda kalmayacak
        btn_mod1_text = "MODÜL 1: ADAY MERKEZ BELİRLEME\n\nBu modülde veri yüklemesi yapıp, Pareto, K-Means ve Minimax algoritmaları ile toplama merkezlerinin stratejik konumlarını adım adım belirleyebilirsiniz."
        if st.button(btn_mod1_text, key="btn_mod1", type="primary", use_container_width=True):
            st.session_state.current_view = 'module1'
            st.rerun()

    with col2:
        btn_mod2_text = "MODÜL 2: OPTİMİZASYON VE SENARYO ANALİZİ\n\nBu modülde kapasite ve maliyet parametrelerini ayarlayarak MILP ağ optimizasyon modelini çalıştırabilir, finansal raporları inceleyebilirsiniz."
        if st.button(btn_mod2_text, key="btn_mod2", type="primary", use_container_width=True):
            st.session_state.current_view = 'module2'
            st.rerun()
            
    st.write("<br><br>", unsafe_allow_html=True)
    col_empty1, col_logout, col_empty2 = st.columns([4, 2, 4])
    with col_logout:
        st.button("Güvenli Çıkış Yap", on_click=lambda: st.session_state.update({'logged_in': False, 'current_view': 'login'}), use_container_width=True)

import folium
from streamlit_folium import st_folium

import folium
import pandas as pd
from streamlit_folium import st_folium
import streamlit as st

# Türkiye'nin 81 ilinin koordinat veri tabanı (Sistem bu sözlükten koordinatları otomatik çeker)
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
    "Hatay": (36.2000, 36.1667), "Isparta": (37.7648, 30.5566), "Mersin": (36.8000, 34.6333),
    "İstanbul": (41.0082, 28.9784), "İzmir": (38.4237, 27.1428), "Kars": (40.6013, 43.0975),
    "Kastamonu": (41.3766, 33.7765), "Kayseri": (38.7312, 35.4787), "Kırklareli": (41.7333, 27.2167),
    "Kırşehir": (39.1425, 34.1709), "Kocaeli": (40.7654, 29.9408), "Konya": (37.8667, 32.4833),
    "Kütahya": (39.4167, 29.9833), "Malatya": (38.3552, 38.3095), "Manisa": (38.6191, 27.4289),
    "Kahramanmaraş": (37.5858, 36.9371), "Mardin": (37.3131, 40.7436), "Muğla": (37.2153, 28.3636),
    "Muş": (38.7304, 41.4990), "Nevşehir": (38.6244, 34.7144), "Niğde": (37.9667, 34.6833),
    "Ordu": (40.9839, 37.8764), "Rize": (41.0201, 40.5234), "Sakarya": (40.7569, 30.3783),
    "Samsun": (41.2867, 36.3300), "Siirt": (37.9333, 41.9500), "Sinop": (42.0231, 35.1531),
    "Sivas": (39.7477, 37.0179), "Tekirdağ": (40.9780, 27.5085), "Tokat": (40.3167, 36.5500),
    "Trabzon": (41.0015, 39.7178), "Tunceli": (39.1079, 39.5401), "Şanlıurfa": (37.1591, 38.7969),
    "Uşak": (38.6823, 29.4082), "Van": (38.4891, 43.3889), "Yozgat": (39.8181, 34.8147),
    "Zonguldak": (41.4564, 31.7987), "Aksaray": (38.3687, 34.0370), "Bayburt": (40.2552, 40.2249),
    "Karaman": (37.1811, 33.2222), "Kırıkkale": (39.8453, 33.5063), "Batman": (37.8812, 41.1351),
    "Şırnak": (37.5228, 42.4594), "Bartın": (41.6344, 32.3375), "Ardahan": (41.1105, 42.7022),
    "Iğdır": (39.9237, 44.0450), "Yalova": (40.6500, 29.2667), "Karabük": (41.2061, 32.6228),
    "Kilis": (36.7184, 37.1147), "Osmaniye": (37.0742, 36.2475), "Düzce": (40.8438, 31.1565)
}

def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()
    
    # ADIM 1: GERÇEK VERİ YÜKLEME VE İŞLEME
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: EV Verisi Yükleme")
        st.write("Sütun başlıkları 'İl' ve 'EV_Sayısı' olan Excel dosyasını yükleyin:")
        uploaded_file = st.file_uploader("", type=['xlsx', 'xls'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                # Excel verisini gerçekten okuyoruz
                df = pd.read_excel(uploaded_file)
                
                if 'İl' not in df.columns or 'EV_Sayısı' not in df.columns:
                    st.error("Excel dosyasında 'İl' ve 'EV_Sayısı' sütunları bulunamadı. Lütfen sütun isimlerini kontrol edin.")
                else:
                    st.session_state.mod1_data = df
                    st.success(f"Veri başarıyla işlendi. Toplam {len(df)} il tespit edildi.")
                    
                    if st.button("Pareto Analizini Başlat", type="primary"):
                        st.session_state.mod1_step = 1
                        st.rerun()
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")

    # ADIM 2: GERÇEK MATEMATİKSEL PARETO ANALİZİ VE HARİTALAMA
    if st.session_state.mod1_step >= 1:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi Sonuçları")
        
        df = st.session_state.mod1_data
        
        # 1. EV Sayısına göre büyükten küçüğe sırala
        df_sorted = df.sort_values(by='EV_Sayısı', ascending=False).reset_index(drop=True)
        
        # 2. Kümülatif toplam ve kümülatif yüzde hesapla
        toplam_ev = df_sorted['EV_Sayısı'].sum()
        df_sorted['Kümülatif_Toplam'] = df_sorted['EV_Sayısı'].cumsum()
        df_sorted['Kümülatif_Yüzde'] = (df_sorted['Kümülatif_Toplam'] / toplam_ev) * 100
        
        # 3. Kümülatif yüzdesi %80'e ulaşana kadar olan (ve %80'i ilk aşan) illeri filtrele
        idx_80 = df_sorted[df_sorted['Kümülatif_Yüzde'] >= 80].index.min()
        pareto_df = df_sorted.iloc[:idx_80 + 1].copy()
        
        secilen_il_sayisi = len(pareto_df)
        st.info(f"Matematiksel analiz tamamlandı. Toplam talebin %80'ini karşılayan en yüksek yoğunluklu {secilen_il_sayisi} il, Aday Toplama Merkezi olarak belirlendi.")
        
        # Haritayı oluştur ve pinleri ekle
        m_pareto = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        for index, row in pareto_df.iterrows():
            il_adi = str(row['İl']).strip().title()
            ev_sayisi = row['EV_Sayısı']
            merkez_kodu = f"TM-{index+1:02d}"
            
            # İli koordinat sözlüğünde ara
            if il_adi in IL_KOORDINATLARI:
                lat, lon = IL_KOORDINATLARI[il_adi]
                tooltip_html = f"""
                <div style='font-family: Arial; font-size: 14px;'>
                    <b>Tesis Tipi:</b> Aday Toplama Merkezi<br>
                    <b>İl Adı:</b> {il_adi}<br>
                    <b>EV Sayısı:</b> {ev_sayisi:,.0f}<br>
                    <b>Merkez Kodu:</b> {merkez_kodu}
                </div>
                """
                folium.Marker(
                    [lat, lon], 
                    tooltip=tooltip_html, 
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m_pareto)
            else:
                st.warning(f"Koordinat bulunamadı: {il_adi}. (Haritaya eklenemedi)")
                
        st_folium(m_pareto, width=900, height=450, key="map_pareto")
        
        # Pareto sonucunu ileriki adımlar için session_state'e kaydet
        st.session_state.pareto_sonuclari = pareto_df

        if st.button("K-Means Kümeleme Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 2
            st.rerun()

   # ==========================================
# ADIM 3: DİRSEK METODU DESTEKLİ GERÇEK K-MEANS 
# ==========================================
if st.session_state.mod1_step >= 2:
    st.divider()
    st.subheader("Adım 3: Dirsek Metodu (Elbow Method) ve K-Means Kümeleme")
    st.write("Sistem, Pareto analizi sonucunda elde edilen aday merkezleri kullanarak optimum Aday İşleme Merkezi sayısını Dirsek Metodu ile matematiksel olarak hesaplayacaktır.")
    
    if st.button("Dirsek Metodu ve K-Means Algoritmasını Çalıştır", type="primary", key="btn_kmeans"):
        # 1. Pareto'dan gelen illerin koordinatlarını al
        pareto_df = st.session_state.pareto_sonuclari
        coords = []
        sehirler = []
        
        for index, row in pareto_df.iterrows():
            il_adi = str(row['İl']).strip().title()
            if il_adi in IL_KOORDINATLARI:
                coords.append(IL_KOORDINATLARI[il_adi])
                sehirler.append(il_adi)
                
        X = np.array(coords)
        
        # 2. DİRSEK METODU (ELBOW METHOD) HESAPLAMASI
        # Aday il sayısına kadar k değerlerini dene (maksimum 10)
        max_k = min(10, len(coords))
        k_degerleri = list(range(1, max_k + 1))
        wcss = []
        
        for k in k_degerleri:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            
        # Optimum k noktasını geometrik olarak bulma (Doğruya en uzak nokta)
        p1 = np.array([k_degerleri[0], wcss[0]])
        p2 = np.array([k_degerleri[-1], wcss[-1]])
        uzakliklar = []
        
        for i in range(len(k_degerleri)):
            p0 = np.array([k_degerleri[i], wcss[i]])
            # Noktanın doğruya olan dik uzaklığı
            uzaklik = np.linalg.norm(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
            uzakliklar.append(uzaklik)
            
        optimum_k = k_degerleri[np.argmax(uzakliklar)]
        
        st.session_state.optimum_k = optimum_k
        st.session_state.wcss = wcss
        st.session_state.k_degerleri = k_degerleri
        
        # 3. Bulunan OPTİMUM K değeri ile Nihai K-Means Modelini Çalıştır
        kmeans_final = KMeans(n_clusters=optimum_k, random_state=42, n_init=10)
        kmeans_final.fit(X)
        centroids = kmeans_final.cluster_centers_
        
        # 4. Matematiksel merkezlere en yakın gerçek illeri atama
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
            
            # Aynı şehrin tekrar eklenmesini önlemek için küçük bir kontrol
            if not any(d['İl'] == closest_city for d in aday_isleme_merkezleri):
                aday_isleme_merkezleri.append({
                    "İl": closest_city,
                    "Lat": IL_KOORDINATLARI[closest_city][0],
                    "Lon": IL_KOORDINATLARI[closest_city][1],
                    "Kod": f"İM-{i+1}"
                })
        
        st.session_state.kmeans_sonuclari = aday_isleme_merkezleri
        st.session_state.kmeans_calisti = True

    # EKRAN ÇIKTILARI (GRAFİK VE HARİTA)
    if st.session_state.get('kmeans_calisti', False):
        opt_k = st.session_state.optimum_k
        
        # 2 Kolon açıyoruz: Solda Dirsek Grafiği, Sağda Açıklama
        col_grafik, col_bilgi = st.columns([1.5, 1])
        
        with col_grafik:
            # Dirsek (Elbow) Grafiğinin Çizdirilmesi
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(st.session_state.k_degerleri, st.session_state.wcss, marker='o', linestyle='-', color='#2F70AF', linewidth=2)
            # Optimum noktayı kırmızı ile işaretle
            opt_wcss = st.session_state.wcss[opt_k - 1]
            ax.plot(opt_k, opt_wcss, marker='o', markersize=10, color='red')
            ax.annotate(f'Optimum k={opt_k}', xy=(opt_k, opt_wcss), xytext=(opt_k, opt_wcss + (max(st.session_state.wcss)*0.1)),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
            
            ax.set_title("K-Means Dirsek (Elbow) Grafiği")
            ax.set_xlabel("Küme Sayısı (k)")
            ax.set_ylabel("WCSS (Küme İçi Hata)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        with col_bilgi:
            st.success(f"**Optimum Küme Sayısı: {opt_k}**")
            st.write("Sistem, WCSS değerlerindeki kırılım noktasına (dirsek) olan en uzak mesafeyi hesaplayarak, veri setiniz için açılması gereken en uygun işleme merkezi sayısını otomatik olarak tespit etmiştir.")
            st.info("Aşağıdaki haritada Pareto (Mavi) adayları, tespit edilen bu optimum küme merkezlerine (Kırmızı) göre atanmıştır.")

        # 1. ve 2. Katmanların Birleştiği Harita
        m_kmeans = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # Katman 1: Mavi Pinler (Pareto)
        pareto_df = st.session_state.pareto_sonuclari
        for index, row in pareto_df.iterrows():
            il_adi = str(row['İl']).strip().title()
            if il_adi in IL_KOORDINATLARI:
                lat, lon = IL_KOORDINATLARI[il_adi]
                tooltip_html = f"<div style='font-family: Arial; font-size: 14px;'><b>Tesis Tipi:</b> Aday Toplama Merkezi<br><b>İl Adı:</b> {il_adi}</div>"
                folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_kmeans)

        # Katman 2: Kırmızı Pinler (Optimum K-Means)
        for merkez in st.session_state.kmeans_sonuclari:
            tooltip_html = f"<div style='font-family: Arial; font-size: 14px;'><b>Tesis Tipi:</b> Aday İşleme Merkezi<br><b>İl Adı:</b> {merkez['İl']}<br><b>Merkez Kodu:</b> {merkez['Kod']}</div>"
            folium.Marker([merkez['Lat'], merkez['Lon']], tooltip=tooltip_html, icon=folium.Icon(color="red", icon="cog")).add_to(m_kmeans)
            
        st_folium(m_kmeans, width=900, height=450, key="map_kmeans_real")

        if st.button("Minimax Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 3
            st.rerun()

    # ADIM 4: MINIMAX VE HARİTASI
    if st.session_state.mod1_step >= 3:
        st.divider()
        st.subheader("Adım 4: Minimax Algoritması Sonuçları")
        st.success("Tüm aday merkez konumları başarıyla tespit edildi! Aday Geri Dönüşüm ve İkinci Ömür Merkezleri eklendi.")
        
        m_minimax = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya Minimax sonuçlarını bağlayacaksın.
        # Temsili Minimax Çıktıları
        geri_donusum_merkezleri = [("Eskişehir", 39.7767, 30.5206, "GDM-26")]
        ikinci_omur_merkezleri = [("Konya", 37.8667, 32.4833, "İÖM-42")]
        
        for il, lat, lon, kod in geri_donusum_merkezleri:
            tooltip_html = f"<b>Tesis Tipi:</b> Aday Geri Dönüşüm Merkezi<br><b>İl:</b> {il}<br><b>Kod:</b> {kod}"
            folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="green", icon="recycle")).add_to(m_minimax)
            
        for il, lat, lon, kod in ikinci_omur_merkezleri:
            tooltip_html = f"<b>Tesis Tipi:</b> Aday İkinci Ömür Merkezi<br><b>İl:</b> {il}<br><b>Kod:</b> {kod}"
            folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="orange", icon="flash")).add_to(m_minimax)

        st_folium(m_minimax, width=900, height=450, key="map_minimax")

# ==========================================
# 5. MODÜL 2 İÇERİĞİ
# ==========================================
def show_module2():
    st.header("Modül 2: Optimizasyon ve Senaryo Analizi")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()
    
    st.write("Bu alana senaryo parametreleri ve model çalıştırma butonları eklenecektir.")

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
