import streamlit as st
import pandas as pd
import time

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

# ==========================================
# 4. MODÜL 1 İÇERİĞİ (GÜNCELLENMİŞ HARİTALI VERSİYON)
# ==========================================
def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()
    
    # ADIM 1: VERİ YÜKLEME
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: EV Verisi Yükleme")
        st.write("İllere göre elektrikli araç sayılarını içeren Excel dosyasını yükleyin:")
        uploaded_file = st.file_uploader("", type=['xlsx', 'xls'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.session_state.mod1_data = "Veri Yüklendi" # Burada pd.read_excel(uploaded_file) yapacaksın
            st.success("Dosya başarıyla sisteme aktarıldı.")
            if st.button("Pareto Analizini Başlat", type="primary"):
                st.session_state.mod1_step = 1
                st.rerun()

    # ADIM 2: PARETO ANALİZİ VE HARİTASI
    if st.session_state.mod1_step >= 1:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi Sonuçları")
        st.info("Toplam talebin %80'ini karşılayan iller belirlendi ve haritaya Aday Toplama Merkezi olarak işlendi.")
        
        # Türkiye merkezli Folium haritası oluşturuluyor
        m_pareto = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya kendi Pareto algoritmanın çıktılarını bağlayacaksın. 
        # Temsili Pareto Çıktıları (İl Adı, Enlem, Boylam, Merkez Kodu)
        pareto_sonuclari = [
            ("İstanbul", 41.0082, 28.9784, "TM-34"),
            ("Ankara", 39.9334, 32.8597, "TM-06"),
            ("İzmir", 38.4237, 27.1428, "TM-35")
        ]
        
        for il, lat, lon, kod in pareto_sonuclari:
            # HTML destekli gelişmiş Tooltip (Fare ile üzerine gelince çıkan bilgi kutusu)
            tooltip_html = f"""
            <div style='font-family: Arial; font-size: 14px;'>
                <b>Tesis Tipi:</b> Aday Toplama Merkezi<br>
                <b>İl Adı:</b> {il}<br>
                <b>Merkez Kodu:</b> {kod}
            </div>
            """
            folium.Marker(
                [lat, lon], 
                tooltip=tooltip_html, 
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m_pareto)
            
        # Haritayı Streamlit ekrana bas
        st_folium(m_pareto, width=900, height=450, key="map_pareto")

        if st.button("K-Means Kümeleme Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 2
            st.rerun()

    # ADIM 3: K-MEANS VE HARİTASI
    if st.session_state.mod1_step >= 2:
        st.divider()
        st.subheader("Adım 3: K-Means Kümeleme Sonuçları")
        st.info("Pareto çıktıları kullanılarak Aday İşleme Merkezleri belirlendi.")
        
        m_kmeans = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya K-Means sonuçlarını bağlayacaksın.
        # Temsili K-Means Çıktıları
        kmeans_sonuclari = [
            ("Bursa", 40.1828, 29.0667, "İM-16"),
            ("Kayseri", 38.7312, 35.4787, "İM-38")
        ]
        
        for il, lat, lon, kod in kmeans_sonuclari:
            tooltip_html = f"""
            <div style='font-family: Arial; font-size: 14px;'>
                <b>Tesis Tipi:</b> Aday İşleme Merkezi<br>
                <b>İl Adı:</b> {il}<br>
                <b>Merkez Kodu:</b> {kod}
            </div>
            """
            # İşleme merkezleri için kırmızı renk kullanıyoruz
            folium.Marker(
                [lat, lon], 
                tooltip=tooltip_html, 
                icon=folium.Icon(color="red", icon="cog")
            ).add_to(m_kmeans)
            
        st_folium(m_kmeans, width=900, height=450, key="map_kmeans")

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
