import streamlit as st
import pandas as pd
import time

# Sayfa ayarları - Geniş ekran
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# 1. SESSION STATE (Hafıza) YÖNETİMİ
# ==========================================
# Kullanıcı giriş yaptı mı?
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Hangi ekrandayız? (login, dashboard, module1, module2)
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'

# Modül 1 verileri
if 'mod1_data' not in st.session_state:
    st.session_state.mod1_data = None
if 'mod1_step' not in st.session_state:
    st.session_state.mod1_step = 0 # 0: Yükleme, 1: Pareto, 2: KMeans, 3: Minimax

# Genel sıfırlama fonksiyonu
def reset_system():
    st.session_state.mod1_data = None
    st.session_state.mod1_step = 0
    # Modül 2 verilerini de buraya ekleyebilirsin
    st.success("Sistem hafızası sıfırlandı. Yeni veri yüklenebilir.")

def go_to_dashboard():
    st.session_state.current_view = 'dashboard'

# ==========================================
# 2. GİRİŞ EKRANI (Login)
# ==========================================
def show_login():
    st.markdown("<h1 style='text-align: center;'>EV Batarya Tersine Lojistik Ağı KDS</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Sisteme Giriş Yapınız</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap", use_container_width=True)
            
            if submit:
                # Hocanın beklediği basit kontrol (Bunu değiştirebilirsin)
                if username == "admin" and password == "1234":
                    st.session_state.logged_in = True
                    st.session_state.current_view = 'dashboard'
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

# ==========================================
# 3. ANA PANEL (Dashboard - 2 Modül Yan Yana)
# ==========================================
def show_dashboard():
    st.title("Sistem Kontrol Paneli")
    st.write("Lütfen işlem yapmak istediğiniz modülü seçin.")
    
    # İki modülü yan yana ve uzun göstermek için CSS destekli butonlar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='border: 2px solid #2e4053; border-radius: 10px; padding: 30px; text-align: center; height: 250px; background-color: #1b2631;'>
            <h2 style='color: #85c1e9;'>Modül 1</h2>
            <h4 style='color: white;'>Aday Toplama Merkezi Belirleme</h4>
            <p style='color: gray;'>Veri Yükleme, Pareto, K-Means ve Minimax Algoritmaları</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Boşluk
        if st.button("Modül 1'e Giriş Yap", key="btn_mod1", use_container_width=True):
            st.session_state.current_view = 'module1'
            st.rerun()

    with col2:
        st.markdown("""
        <div style='border: 2px solid #2e4053; border-radius: 10px; padding: 30px; text-align: center; height: 250px; background-color: #1b2631;'>
            <h2 style='color: #f5b041;'>Modül 2</h2>
            <h4 style='color: white;'>Ağ Optimizasyonu & Senaryo Analizi</h4>
            <p style='color: gray;'>Parametre Ayarları ve MILP Modelinin Çalıştırılması</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Boşluk
        if st.button("Modül 2'ye Giriş Yap", key="btn_mod2", use_container_width=True):
            st.session_state.current_view = 'module2'
            st.rerun()
            
    st.divider()
    st.button("Çıkış Yap", on_click=lambda: st.session_state.update({'logged_in': False, 'current_view': 'login'}))

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
