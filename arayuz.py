import streamlit as st
import pandas as pd
import time

# Sayfa ayarları - Geniş ekran (Bu her zaman en üstte olmalı)
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# GÖRSEL TASARIM (CSS INJECTION)
# ==========================================
# Sayfanın tamamını antrasit yapar ve buton modüllerine gölge/derinlik kazandırır
custom_css = """
<style>
/* Ana Arka Planı Antrasit Yapma */
[data-testid="stAppViewContainer"] {
    background-color: #2b2e33 !important; /* Premium Antrasit Rengi */
    color: #f0f2f6 !important;
}
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Dashboard'daki Yan Yana 2 Dev Buton Modülünün Tasarımı */
/* Boyutları uzatıldı, kart görünümü verildi */
div[data-testid="column"] button {
    height: 320px !important; /* Modül boyu uzatıldı */
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    background-color: #1f2226 !important; /* Buton içi daha koyu antrasit */
}

/* Modül 1 Tasarımı - Kırmızı Ağırlıklı (Hocanın iyi görebilmesi için yüksek kontrast) */
div[data-testid="column"]:nth-of-type(1) button {
    border-left: 8px solid #E53935 !important; /* Güçlü Kırmızı çizgi */
    box-shadow: 0 4px 10px rgba(229, 57, 53, 0.15) !important;
}
div[data-testid="column"]:nth-of-type(1) button:hover {
    box-shadow: 0 12px 24px rgba(229, 57, 53, 0.4) !important;
    transform: translateY(-6px);
    border-color: #E53935 !important;
}

/* Modül 2 Tasarımı - Mavi Ağırlıklı (Kırmızı ile asla karışmaz, renk körü dostu) */
div[data-testid="column"]:nth-of-type(2) button {
    border-left: 8px solid #1E88E5 !important; /* Güçlü Mavi çizgi */
    box-shadow: 0 4px 10px rgba(30, 136, 229, 0.15) !important;
}
div[data-testid="column"]:nth-of-type(2) button:hover {
    box-shadow: 0 12px 24px rgba(30, 136, 229, 0.4) !important;
    transform: translateY(-6px);
    border-color: #1E88E5 !important;
}

/* Buton İçindeki Yazıların Ayarı */
div[data-testid="column"] button p {
    font-size: 20px !important;
    font-weight: 500 !important;
    white-space: pre-wrap !important; /* Yazının alt satıra inmesi için */
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
    st.markdown("<h1 style='text-align: center; color: white;'>EV Batarya Tersine Lojistik Ağı KDS</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #a9a9a9;'>Sisteme Giriş Yapınız</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap", use_container_width=True)
            
            if submit:
                # Chrome uyarı vermesin diye şifreyi proje2026 olarak güncelledik
                if username == "admin" and password == "proje2026":
                    st.session_state.logged_in = True
                    st.session_state.current_view = 'dashboard'
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

# ==========================================
# 3. ANA PANEL (Dashboard - Tıklanabilir Dikdörtgen Modüller)
# ==========================================
def show_dashboard():
    st.markdown("<h1 style='text-align: center; color: white;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    
    # İstenen açıklama paragrafı (Ortalanmış)
    st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #b3b8c2; max-width: 900px; margin: 0 auto; line-height: 1.6;'>
    Bu Karar Destek Sistemi (KDS), ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde sürdürülebilir ve maliyet-etkin bir şekilde toplanarak geri kazanılması için tersine tedarik zinciri ağını tasarlamak ve senaryo bazlı optimize etmek amacıyla geliştirilmiştir. Lütfen çalışmak istediğiniz modüle tıklayınız.
    </p>
    <br><br>
    """, unsafe_allow_html=True)
    
    # 2 Modülü yan yana koyan kolonlar
    col1, col2 = st.columns(2)
    
    with col1:
        # Modül 1'in kendisi artık devasa bir buton. Satır atlamaları \n\n ile yapıyoruz.
        btn_mod1_text = "🔴 MODÜL 1\n\nADAY TOPLAMA MERKEZİ BELİRLEME\n\n----------------------------\n\nVeri Yükleme\nPareto Analizi (%80 Talep)\nK-Means Kümeleme\nMinimax Algoritması"
        if st.button(btn_mod1_text, key="btn_mod1", use_container_width=True):
            st.session_state.current_view = 'module1'
            st.rerun()

    with col2:
        # Modül 2'nin kendisi de buton
        btn_mod2_text = "🔵 MODÜL 2\n\nOPTİMİZASYON & SENARYO ANALİZİ\n\n----------------------------\n\nParametre ve Maliyet Ayarları\nMILP Ağ Modeli Çözümü\nAkış Miktarları ve Finansal Raporlar"
        if st.button(btn_mod2_text, key="btn_mod2", use_container_width=True):
            st.session_state.current_view = 'module2'
            st.rerun()
            
    st.write("<br><br>", unsafe_allow_html=True)
    col_empty1, col_logout, col_empty2 = st.columns([4, 2, 4])
    with col_logout:
        st.button("Güvenli Çıkış Yap", on_click=lambda: st.session_state.update({'logged_in': False, 'current_view': 'login'}), use_container_width=True)

# Geri kalan Modül 1 ve Modül 2 fonksiyonları önceki kodla aynı kalabilir...
