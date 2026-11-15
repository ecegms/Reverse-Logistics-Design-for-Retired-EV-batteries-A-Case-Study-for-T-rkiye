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
# ==========================================
# 4. MODÜL 1: ADAY MERKEZ BELİRLEME
# ==========================================
def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme Aşamaları")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()

    # ADIM 0: Veri Yükleme
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: İl Bazlı EV Verisi Yükleme")
        uploaded_file = st.file_uploader("İllere göre elektrikli araç sayılarını içeren Excel dosyasını yükleyin", type=['xlsx', 'xls'])
        
        if uploaded_file is not None and st.session_state.mod1_data is None:
            # Gerçekte pd.read_excel(uploaded_file) yapacaksın
            st.session_state.mod1_data = "Veri Yüklendi" 
            st.success("Dosya başarıyla sisteme aktarıldı.")
            
        if st.session_state.mod1_data is not None:
            if st.button("Pareto Analizini Başlat"):
                st.session_state.mod1_step = 1
                st.rerun()

    st.divider()

    # ADIM 1: Pareto Analizi
    if st.session_state.mod1_step >= 1:
        st.subheader("Adım 2: Pareto Analizi (%80 Talep Karşılama)")
        with st.spinner("Pareto uygulanıyor..."):
            time.sleep(1) # Simülasyon
        st.info("Pareto analizi tamamlandı. Toplam talebin %80'ini oluşturan iller haritada işaretlendi.")
        # Buraya folium veya st.map haritasını koyacaksın
        st.write("[ HARİTA / GRAFİK BURAYA GELECEK ]")
        
        if st.button("K-Means Kümeleme Algoritmasını Çalıştır"):
            st.session_state.mod1_step = 2
            st.rerun()

    st.divider()

    # ADIM 2: K-Means
    if st.session_state.mod1_step >= 2:
        st.subheader("Adım 3: K-Means ile Merkez Seçimi")
        st.info("K-Means uygulandı. Küme merkezleri belirlendi.")
        st.write("[ K-MEANS SONUÇ HARİTASI BURAYA GELECEK ]")
        
        if st.button("Minimax Algoritmasını Çalıştır"):
            st.session_state.mod1_step = 3
            st.rerun()

    st.divider()
    
    # ADIM 3: Minimax ve Nihai Sonuç
    if st.session_state.mod1_step >= 3:
        st.subheader("Adım 4: Minimax ve Nihai Aday Merkezler")
        st.success("Tüm adımlar tamamlandı. Nihai aday tesis konumları belirlendi.")
        st.write("[ NİHAİ MERKEZLER TABLOSU VE HARİTASI BURAYA GELECEK ]")
        st.write("Artık Modül 2'ye geçerek optimizasyon modelini çalıştırabilirsiniz.")

# ==========================================
# 5. MODÜL 2: OPTİMİZASYON VE SENARYO ANALİZİ
# ==========================================
def show_module2():
    st.header("Modül 2: Matematiksel Model (MILP) ve Senaryo Analizi")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()
    
    col_params, col_results = st.columns([1, 2])
    
    with col_params:
        st.subheader("Senaryo Parametreleri")
        st.write("Modeli çalıştırmadan önce parametreleri ayarlayabilirsiniz.")
        
        # Senaryo analizi için parametre girişleri
        kapasite_tasima = st.number_input("Tır Taşıma Kapasitesi (Adet)", value=50, step=5)
        maliyet_km = st.number_input("Birim Nakliye Maliyeti (TL/km)", value=25.0, step=1.0)
        geri_kazanim_orani = st.slider("Geri Kazanım Başarı Oranı (%)", min_value=50, max_value=100, value=85)
        
        st.write("")
        if st.button("🚀 Modeli Çöz (Gurobi Arka Planda Çalışır)", type="primary", use_container_width=True):
            st.session_state.model_solved = True
            with st.spinner("Model çözülüyor, lütfen bekleyiniz..."):
                time.sleep(2) # Gurobi çözümü simülasyonu
            st.rerun()

    with col_results:
        st.subheader("Optimizasyon Sonuçları")
        if st.session_state.get('model_solved', False):
            st.success("Model Optimal Çözüme Ulaştı!")
            st.metric(label="Toplam Ağ Maliyeti", value="₺ 4,250,000", delta="-₺ 150,000 (Bir önceki senaryoya göre)")
            
            # Sonuçları sekmelerle göstermek çok profesyonel durur
            tab1, tab2, tab3 = st.tabs(["Tesis Açılış Kararları", "Akış Miktarları", "Finansal Özet"])
            with tab1:
                st.write("Hangi ile geri dönüşüm tesisi açılacak tablosu...")
            with tab2:
                st.write("Toplama merkezinden tesislere giden akış rotaları...")
            with tab3:
                st.write("Taşıma, kurulum ve operasyon maliyetleri kırılımı...")
        else:
            st.info("Lütfen sol taraftan parametreleri ayarlayıp modeli çalıştırın.")

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
