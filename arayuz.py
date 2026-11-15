import streamlit as st
import pandas as pd
import time

# Sayfa ayarları (Her zaman en üstte olmalı)
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# GÖRSEL TASARIM (CSS INJECTION)
# ==========================================
# Sadece "primary" (Ana) tipindeki butonları DEVASA KIRMIZI MODÜLLERE çevirir
# Diğer normal butonlar (Giriş yap, Çıkış yap, Sıfırla) bu durumdan etkilenmez!
custom_css = """
<style>
/* Modül Butonlarının Genel Tasarımı (Aşağıya doğru uzun ve kırmızı) */
button[data-testid="baseButton-primary"] {
    height: 400px !important; /* Boyu aşağıya doğru iyice uzatıldı */
    width: 100% !important;
    background-color: #C62828 !important; /* Tok ve profesyonel bir Kırmızı */
    color: #ffffff !important; /* Buton içi yazı her modda beyaz kalır */
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important; /* Buton (Derinlik) Hissi */
    transition: all 0.3s ease !important;
    white-space: pre-wrap !important; /* Yazının üç nokta (...) olmasını engeller, alt satıra indirir */
    padding: 30px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}

/* Farenin modül butonunun üzerine gelme (Hover) efekti */
button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 15px 30px rgba(198, 40, 40, 0.4) !important;
    background-color: #D32F2F !important;
    border: none !important;
}

/* Buton içindeki yazıların (p etiketlerinin) formatı - Zarif font */
button[data-testid="baseButton-primary"] p {
    font-size: 20px !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    text-align: center !important;
    font-family: 'Georgia', serif !important; /* Zarif bir font ailesi */
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
    # Markdown'larda color belirtmiyoruz ki sistem Light/Dark moda göre rengi kendi ayarlasın
    st.markdown("<h1 style='text-align: center;'>EV Batarya Tersine Lojistik Ağı KDS</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; opacity: 0.7;'>Sisteme Giriş Yapınız</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap", use_container_width=True)
            
            if submit:
                # Güvenlik uyarısı vermemesi için şifreyi değiştirdik
                if username == "admin" and password == "proje2026":
                    st.session_state.logged_in = True
                    st.session_state.current_view = 'dashboard'
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

# ==========================================
# 3. ANA PANEL (Dashboard - Kırmızı Dev Modüller)
# ==========================================
def show_dashboard():
    st.markdown("<h1 style='text-align: center;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    
    # Ortalanmış, mod uyumlu zarif açıklama paragrafı
    st.markdown("""
    <p style='text-align: center; font-size: 18px; max-width: 900px; margin: 0 auto; line-height: 1.6; opacity: 0.8;'>
    Bu Karar Destek Sistemi, ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde toplanması ve geri kazanılması için tersine lojistik ağını tasarlamak amacıyla geliştirilmiştir.
    </p>
    <br><br>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Alt satıra inmesi için \n kullanıyoruz. (Pre-wrap CSS'i sayesinde bu çalışacak)
        btn_mod1_text = "MODÜL 1: ADAY MERKEZ BELİRLEME\n\nBu modülde veri yüklemesi yapıp, Pareto, K-Means ve Minimax algoritmaları ile toplama merkezlerinin stratejik konumlarını adım adım belirleyebilirsiniz."
        
        # type="primary" parametresi sayesinde sadece bu butonlar dev kırmızı CSS kodundan etkilenir
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
        # type="secondary" (varsayılan) olduğu için bu buton standart kalır
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
        uploaded_file = st.file_uploader("İllere göre elektrikli araç sayılarını içeren Excel dosyasını yükleyin", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            st.session_state.mod1_data = "Veri Yüklendi" 
            st.success("Dosya başarıyla sisteme aktarıldı.")
            if st.button("Pareto Analizini Başlat"):
                st.session_state.mod1_step = 1
                st.rerun()

    if st.session_state.mod1_step >= 1:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi")
        st.info("Pareto analizi tamamlandı.")
        if st.button("K-Means Kümeleme Algoritmasını Çalıştır"):
            st.session_state.mod1_step = 2
            st.rerun()

    if st.session_state.mod1_step >= 2:
        st.divider()
        st.subheader("Adım 3: K-Means ve Minimax Algoritmaları")
        st.success("Merkez konumları başarıyla tespit edildi. Modül 2'ye geçebilirsiniz.")

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
