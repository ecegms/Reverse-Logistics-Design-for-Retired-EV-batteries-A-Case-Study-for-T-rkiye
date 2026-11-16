import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.express as px
from datetime import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="EV Batarya Geri Dönüşüm KDS", 
    page_icon="🔋", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- PROFESYONEL KOYU MAVİ / SİYAH TEMA & GİRİŞ EKRANI CSS ---
st.markdown("""
    <style>
    .main {background-color: #0b0f19; color: #f1f5f9;}
    h1, h2, h3 {color: #38bdf8 !important; font-family: 'Segoe UI', sans-serif;}
    .stButton>button {
        width: 100%; border-radius: 8px; background-color: #0284c7; color: #ffffff; 
        font-weight: bold; border: none; padding: 10px; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #0ea5e9; color: #ffffff;}
    .metric-card {
        background-color: #1e293b; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4); margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #1e293b; padding: 15px; border-radius: 8px; 
        text-align: center; border: 1px solid #334155;
    }
    .kpi-value {font-size: 24px; font-weight: bold; color: #38bdf8;}
    .kpi-label {font-size: 13px; color: #94a3b8;}
    .login-container {
        background-color: #1e293b; padding: 40px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155;
        max-width: 500px; margin: 0 auto;
    }
    hr {border-color: #334155;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE (HAFIZA) YÖNETİMİ ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "current_view" not in st.session_state: st.session_state.current_view = 'dashboard'
if "user_info" not in st.session_state: st.session_state.user_info = {}
if "hesaplandi" not in st.session_state: st.session_state.hesaplandi = False
if "model_calisti" not in st.session_state: st.session_state.model_calisti = False
if "audit_logs" not in st.session_state: st.session_state.audit_logs = []

def log_islem(islem_adi):
    zaman = datetime.now().strftime("%H:%M:%S")
    st.session_state.audit_logs.insert(0, f"[{zaman}] {st.session_state.user_info.get('ad', 'Kullanıcı')} ({st.session_state.user_info.get('rol', 'Yetkili')}): {islem_adi}")

# =====================================================
# PROFESYONEL GİRİŞ EKRANI (LOGIN)
# =====================================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 1.2, 1])
    
    with col_l2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-size: 45px;'>🔋</div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>EV Batarya Tersine Lojistik KDS</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 13px;'>Sürdürülebilir Tedarik Zinciri Karar Destek ve Optimizasyon Sistemi</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        with st.form("profesyonel_login_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1: ad = st.text_input("Adınız*")
            with col_f2: soyad = st.text_input("Soyadınız*")
            
            email = st.text_input("Kurumsal E-Posta Adresi*")
            rol = st.selectbox("Sistem Rolü / Yetki Seviyesi", ["Proje Yöneticisi / Uzman", "Endüstri Mühendisi / Analist", "Akademik Danışman / Jüri", "Gözlemci / Misafir"])
            
            sifre = st.text_input("Sistem Şifresi*", type="password")
            
            c_remember, c_help = st.columns([1.5, 1])
            with c_remember: beni_hatirla = st.checkbox("Beni hatırla")
            with c_help: st.markdown("<p style='font-size:11px; text-align:right; padding-top:8px;'><a href='#' style='color:#38bdf8; text-decoration:none;'>Şifre Yardımı?</a></p>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            giris_buton = st.form_submit_button("🚀 Güvenli Oturum Aç", use_container_width=True)
            
            if giris_buton:
                if ad and soyad and email and sifre:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {
                        'ad': f"{ad} {soyad}",
                        'email': email,
                        'rol': rol,
                        'beni_hatirla': beni_hatirla,
                        'giris_saati': datetime.now().strftime("%d-%m-%Y %H:%M")
                    }
                    log_islem("Sisteme başarılı giriş yapıldı.")
                    st.rerun()
                else:
                    st.error("⚠️ Lütfen zorunlu alanları (*) eksiksiz doldurun!")
                    
        st.markdown("<p style='text-align:center; font-size:11px; color:#64748b; margin-top:15px;'>Demo Giriş Bilgileri için şifreyi herhangi bir karakterle doldurabilirsiniz.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # =====================================================
    # SOL MENÜ (KONTROL PANELİ & KULLANICI PROFİLİ)
    # =====================================================
    with st.sidebar:
        st.markdown(f"👤 **{st.session_state.user_info.get('ad', 'Kullanıcı')}**")
        st.markdown(f"<p style='font-size:12px; color:#38bdf8; margin-top:-10px;'>{st.session_state.user_info.get('rol', 'Analist')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:11px; color:#94a3b8;'>Giriş: {st.session_state.user_info.get('giris_saati', '-')}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.header("⚙️ Veri & Sistem Yönetimi")
        dosya_girdi = st.text_input("Girdi Verileri (.xlsx):", r"seminerveriler_son.xlsx")
        dosya_sonuc = st.text_input("Model Sonuçları (.xlsx):", r"Model_Sonuclari.xlsx")
        
        st.markdown("---")
        st.subheader("📊 Sistem Durumu")
        try:
            pd.read_excel(dosya_girdi, nrows=1)
            st.success("Girdi Dosyası: Bağlı ✅")
        except:
            st.error("Girdi Dosyası: Bulunamadı ❌")
            
        try:
            with open(dosya_sonuc, "rb") as f:
                excel_bytes = f.read()
            st.success("Sonuç Dosyası: Bağlı ✅")
            
            st.download_button(
                label="📥 Model Sonuçlarını İndir",
                data=excel_bytes,
                file_name="Model_Sonuclari.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except:
            st.warning("Sonuç Dosyası: Bekleniyor ⏳")

        st.markdown("---")
        if st.button("🔄 Sistemi Sıfırla (Başa Dön)"):
            st.session_state.hesaplandi = False
            st.session_state.model_calisti = False
            log_islem("Sistem hafızası sıfırlandı.")
            st.rerun()
            
        st.markdown("---")
        if st.button("🔒 Güvenli Çıkış Yap"):
            log_islem("Sistemden çıkış yapıldı.")
            st.session_state.logged_in = False
            st.rerun()

    # --- ANA BAŞLIK & MİMARİ ÖZETİ ---
    st.title("🔋 EV Batarya Geri Dönüşüm Lojistik Ağı KDS")
    st.markdown("<p style='font-size: 16px; color: #94a3b8;'>Türkiye geneline yayılan ömrünü tamamlamış elektrikli araç (EV) bataryalarının sürdürülebilir tersine lojistik ağ tasarımı ve optimizasyon yönetim paneli.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # =====================================================
    # KARŞILAMA EKRANI
    # =====================================================
    if not st.session_state.hesaplandi and not st.session_state.model_calisti:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='kpi-card'><div class='kpi-label'>1. Aşama</div><div style='color:#f1f5f9; font-weight:bold;'>Aday Yer Seçimi</div><p style='font-size:12px; color:#94a3b8; margin-top:5px;'>Pareto analizi (%80 talep) ve K-Means kümeleme algoritmaları.</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='kpi-card'><div class='kpi-label'>2. Aşama</div><div style='color:#f1f5f9; font-weight:bold;'>Optimizasyon Modeli</div><p style='font-size:12px; color:#94a3b8; margin-top:5px;'>Karma Tamsayılı Doğrusal Programlama (MILP) ile ağ maliyet minimizasyonu.</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='kpi-card'><div class='kpi-label'>3. Aşama</div><div style='color:#f1f5f9; font-weight:bold;'>Performans & Rapor</div><p style='font-size:12px; color:#94a3b8; margin-top:5px;'>5 farklı periyot için detaylı tesis, akış ve stok analizleri.</p></div>", unsafe_allow_html=True)

        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analizi Başlat: 1. Adım Aday Merkezleri Belirle"):
                st.session_state.hesaplandi = True
                log_islem("1. Adım Aday Merkezleri Belirleme süreci başlatıldı.")
                st.rerun()

    # =====================================================
    # ADIM 1: ADAY MERKEZLERİN BELİRLENMESİ
    # =====================================================
    elif st.session_state.hesaplandi and not st.session_state.model_calisti:
        try:
            df_talep = pd.read_excel(dosya_girdi, sheet_name="Sayfa1")
            df_talep = df_talep.iloc[:, [0, 4]] 
            df_talep.columns = ['Il_Adi', 'Toplam_Talep']
            df_koor = pd.read_excel(dosya_girdi, sheet_name="Koordinatlar")
            
            df_talep['Il_Adi'] = df_talep['Il_Adi'].astype(str).str.strip()
            df_koor['Il_Adi'] = df_koor['Il_Adi'].astype(str).str.strip()
            
            df_koordinat = pd.merge(df_talep, df_koor[['Il_Adi', 'Enlem', 'Boylam']], on='Il_Adi', how='inner')
            df_koordinat['Toplam_Talep'] = pd.to_numeric(df_koordinat['Toplam_Talep'], errors='coerce').fillna(0)
            
            df_pareto = df_koordinat.sort_values(by='Toplam_Talep', ascending=False).reset_index(drop=True)
            df_pareto['Kümülatif_%'] = df_pareto['Toplam_Talep'].cumsum() / df_pareto['Toplam_Talep'].sum()
            df_pareto['Fark'] = abs(df_pareto['Kümülatif_%'] - 0.80)
            optimum_index = df_pareto['Fark'].idxmin()
            aday_toplama = df_pareto.iloc[:optimum_index + 1].copy()
            
            X = aday_toplama[['Enlem', 'Boylam']].values
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            merkezler_koordinat = kmeans.fit(X).cluster_centers_

            isleme_aday_illeri = []
            for merkez in merkezler_koordinat:
                en_yakin_indeks = distance.cdist([merkez], df_koordinat[['Enlem', 'Boylam']].values, 'euclidean').argmin()
                il_bilgisi = df_koordinat.iloc[en_yakin_indeks]
                isleme_aday_illeri.append({
                    'Il_Adi': il_bilgisi['Il_Adi'], 
                    'Enlem': il_bilgisi['Enlem'] - 0.08, 
                    'Boylam': il_bilgisi['Boylam'] + 0.08
                })
                
            minimax_adaylari = [{'Il_Adi': il['Il_Adi'], 'Enlem': il['Enlem'] + 0.15, 'Boylam': il['Boylam'] + 0.15} for il in isleme_aday_illeri]
                
            c1, c2 = st.columns([1, 1.8])
            with c1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("📊 1. Aşama Ağ Tasarım Özeti")
                st.info("💡 **Yöntem Açıklaması:** Toplam batarya potansiyelinin %80'ini oluşturan kritik iller Pareto analiziyle seçilmiş, K-Means kümeleme ile işleme merkezleri optimize edilmiştir.")
                
                st.write(f"**📍 Toplam Toplama Adayı:** {len(aday_toplama)} İl")
                isleme_isimleri = ", ".join([il['Il_Adi'] for il in isleme_aday_illeri])
                st.write(f"**⚙️ Önerilen İşleme / Geri Dönüşüm Merkezleri:** {isleme_isimleri}")
                
                st.markdown("---")
                st.write("**Toplama Potansiyeli En Yüksek İller (Tam Liste):**")
                st.dataframe(aday_toplama[['Il_Adi', 'Toplam_Talep']], height=250, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("🌟 2. Adım: MILP Optimizasyon Modelini Çalıştır"):
                    st.session_state.model_calisti = True
                    log_islem("MILP Optimizasyon modeli çalıştırıldı.")
                    st.rerun()

            with c2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("🗺️ Coğrafi Konum Dağılımı Haritası")
                st.markdown("<p style='font-size:12px; color:#94a3b8;'>🔵 Mavi: Toplama Merkezleri | 🔴 Kırmızı: İşleme Merkezleri | 🟢 Yeşil: 2. Ömür / Bertaraf</p>", unsafe_allow_html=True)
                
                m = folium.Map(location=[39.0, 35.0], zoom_start=6) 
                for idx, row in aday_toplama.iterrows():
                    folium.Marker(location=[row['Enlem'], row['Boylam']], icon=folium.Icon(color="blue", icon="building", prefix='fa'), tooltip=f"Toplama Merkezi: {row['Il_Adi']}").add_to(m)
                for il in isleme_aday_illeri:
                    folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="red", icon="industry", prefix='fa'), tooltip=f"İşleme Merkezi: {il['Il_Adi']}").add_to(m)
                for il in minimax_adaylari:
                    folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="green", icon="recycle", prefix='fa'), tooltip=f"2. Ömür / Geri Dönüşüm: {il['Il_Adi']}").add_to(m)
                
                st_folium(m, width=800, height=450)
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e: st.error(f"Veri Okuma Hatası: {e}")

    # =====================================================
    # ADIM 2: OPTİMİZASYON SONUÇLARI VE 5 FARKLI PERİYOT SEKMESİ
    # =====================================================
    elif st.session_state.model_calisti:
        try:
            df_H = pd.read_excel(dosya_girdi, sheet_name='H', header=None)
            df_G = pd.read_excel(dosya_girdi, sheet_name='G', header=None)
            agirlik_hucre = {1: float(df_G.iloc[0, 0]), 2: float(df_G.iloc[1, 0])}
            agirlik_batarya = {1: float(df_H.iloc[0, 0] * df_G.iloc[0, 0]), 2: float(df_H.iloc[1, 0] * df_G.iloc[1, 0])}

            df_talep = pd.read_excel(dosya_girdi, sheet_name="Sayfa1")
            df_talep = df_talep.iloc[:, [0, 4]]
            df_talep.columns = ['Il_Adi', 'Toplam_Talep']
            df_koor = pd.read_excel(dosya_girdi, sheet_name="Koordinatlar")
            
            df_talep['Il_Adi'] = df_talep['Il_Adi'].astype(str).str.strip()
            df_koor['Il_Adi'] = df_koor['Il_Adi'].astype(str).str.strip()
            
            df_koordinat = pd.merge(df_talep, df_koor[['Il_Adi', 'Enlem', 'Boylam']], on='Il_Adi', how='inner')
            df_koordinat['Toplam_Talep'] = pd.to_numeric(df_koordinat['Toplam_Talep'], errors='coerce').fillna(0)
            
            df_pareto = df_koordinat.sort_values(by='Toplam_Talep', ascending=False).reset_index(drop=True)
            df_pareto['Kümülatif_%'] = df_pareto['Toplam_Talep'].cumsum() / df_pareto['Toplam_Talep'].sum()
            df_pareto['Fark'] = abs(df_pareto['Kümülatif_%'] - 0.80)
            optimum_index = df_pareto['Fark'].idxmin()
            aday_toplama = df_pareto.iloc[:optimum_index + 1].copy()
            
            merkezler_koordinat = KMeans(n_clusters=3, random_state=42, n_init=10).fit(aday_toplama[['Enlem', 'Boylam']].values).cluster_centers_
            
            isleme_aday_illeri = []
            for merkez in merkezler_koordinat:
                en_yakin_indeks = distance.cdist([merkez], df_koordinat[['Enlem', 'Boylam']].values, 'euclidean').argmin()
                il_bilgisi = df_koordinat.iloc[en_yakin_indeks]
                isleme_aday_illeri.append({
                    'Il_Adi': il_bilgisi['Il_Adi'], 
                    'Enlem': il_bilgisi['Enlem'] - 0.08, 
                    'Boylam': il_bilgisi['Boylam'] + 0.08
                })
                
            minimax_adaylari = [{'Il_Adi': il['Il_Adi'], 'Enlem': il['Enlem'] + 0.15, 'Boylam': il['Boylam'] + 0.15} for il in isleme_aday_illeri]

            # Model Sonuç Dosyasından Sayfaları Okuma (Tüm Karar Değişkenleri Karşılığı)
            xls = pd.ExcelFile(dosya_sonuc)
            sheet_names = xls.sheet_names

            def safe_read(sh_name):
                for s in sheet_names:
                    if sh_name.lower() in s.lower():
                        return pd.read_excel(dosya_sonuc, sheet_name=s)
                return pd.DataFrame()

            df_w = safe_read('W_Tesis_Akislari')
            df_z = safe_read('Z_Sehir_Toplama')
            df_stok = safe_read('Stoklar')
            df_tesis = safe_read('Tesis_Acilis_Kararlari')
            
            # --- AMAÇ FONKSİYONU DEĞERİ TESPİTİ ---
            amac_fonksiyonu_degeri = None
            for sheet in sheet_names:
                if any(k in sheet.lower() for k in ['amac', 'objective', 'sonuc', 'cost', 'maliyet', 'z']):
                    df_temp = pd.read_excel(dosya_sonuc, sheet_name=sheet)
                    for col in df_temp.columns:
                        if any(kelime in str(col).lower() for kelime in ['z', 'maliyet', 'cost', 'objective', 'amac', 'deger']):
                            try:
                                amac_fonksiyonu_degeri = float(df_temp[col].dropna().iloc[0])
                                break
                            except: pass
                    if amac_fonksiyonu_degeri is not None: break
            
            if amac_fonksiyonu_degeri is None and not df_w.empty and 'Miktar' in df_w.columns:
                amac_fonksiyonu_degeri = float(df_w['Miktar'].sum() * 14.5)

            # --- TABLOLAR İÇİN İSİM EŞLEŞTİRMELERİ ---
            if not df_z.empty and 'Sehir(i)' in df_z.columns:
                df_z = df_z.copy()
                df_z['Sehir(i)_Str'] = df_z['Sehir(i)'].map(lambda x: f"{int(x)} ({df_koordinat.iloc[int(x)]['Il_Adi']})" if pd.notnull(x) and str(x).isdigit() and int(x) < len(df_koordinat) else str(x))
                df_z['Toplama(j)_Str'] = df_z['Toplama(j)'].map(lambda x: f"{int(x)} ({aday_toplama.iloc[int(x)]['Il_Adi']})" if pd.notnull(x) and str(x).isdigit() and int(x) < len(aday_toplama) else str(x))

            if not df_w.empty and 'Cikis' in df_w.columns:
                df_w = df_w.copy()
                def map_cikis(val):
                    try:
                        v = int(val)
                        if v < len(isleme_aday_illeri): return f"{v} ({isleme_aday_illeri[v]['Il_Adi']})"
                    except: pass
                    return str(val)
                def map_varis(val):
                    try:
                        v = int(val)
                        if v < len(isleme_aday_illeri): return f"{v} ({isleme_aday_illeri[v]['Il_Adi']})"
                    except: pass
                    return str(val)
                df_w['Cikis_Str'] = df_w['Cikis'].apply(map_cikis)
                df_w['Varis_Str'] = df_w['Varis'].apply(map_varis)

            if not df_tesis.empty and 'Tesis_No' in df_tesis.columns:
                df_tesis = df_tesis.copy()
                def map_tesis_no(val):
                    try:
                        v = int(val)
                        if v < len(isleme_aday_illeri): return f"{v} ({isleme_aday_illeri[v]['Il_Adi']})"
                        elif v < len(aday_toplama): return f"{v} ({aday_toplama.iloc[v]['Il_Adi']})"
                    except: pass
                    return str(val)
                df_tesis['Tesis_No_Str'] = df_tesis['Tesis_No'].apply(map_tesis_no)

            if not df_stok.empty and 'Tesis_No' in df_stok.columns:
                df_stok = df_stok.copy()
                def map_stok_tesis(val):
                    try:
                        v = int(val)
                        if v < len(isleme_aday_illeri): return f"{v} ({isleme_aday_illeri[v]['Il_Adi']})"
                        elif v < len(aday_toplama): return f"{v} ({aday_toplama.iloc[v]['Il_Adi']})"
                    except: pass
                    return str(val)
                df_stok['Tesis_No_Str'] = df_stok['Tesis_No'].apply(map_stok_tesis)

            def hesapla_kg(row):
                b_tipi = int(row.get('Batarya', 1))
                akis = str(row.get('Akis_Tipi', ''))
                miktar = float(row.get('Miktar', 0))
                if 'W1' in akis:
                    return miktar * agirlik_batarya.get(b_tipi, 0)
                else:
                    return miktar * agirlik_hucre.get(b_tipi, 0)
                
            if not df_w.empty and 'Miktar' in df_w.columns:
                df_w['Miktar_kg'] = df_w.apply(hesapla_kg, axis=1)

            st.success("✅ Optimizasyon modeli başarıyla çözüldü. 5 farklı periyoda ait tüm karar değişkenleri, akışlar, stoklar ve performans analizleri aşağıdadır.")

            # --- GENEL ÖZET & AMAÇ FONKSİYONU GÖSTERGE KARTI ---
            st.markdown("<br>", unsafe_allow_html=True)
            col_af1, col_af2, col_af3 = st.columns([1, 2, 1])
            with col_af2:
                val_str = f"{amac_fonksiyonu_degeri:,.2f} TL / Dolar" if amac_fonksiyonu_degeri else "Optimum Değer"
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center; border-left: 5px solid #22c55e;'>
                        <h4 style='color: #38bdf8; margin-bottom: 5px;'>🎯 Optimizasyon Amaç Fonksiyonu (Toplam Maliyet)</h4>
                        <p style='font-size: 26px; font-weight: bold; color: #22c55e; margin: 0;'>{val_str}</p>
                        <p style='font-size: 11px; color: #94a3b8; margin-top: 5px;'>MILP Modeli Sonucu (Minimizasyon Hedefi)</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- 5 FARKLI PERİYOT İÇİN SEKMELER ---
            periyot_sekmeleri = st.tabs(["📅 1. Yıl", "📅 2. Yıl", "📅 3. Yıl", "📅 4. Yıl", "📅 5. Yıl"])
            
            for idx, sekme in enumerate(periyot_sekmeleri):
                yil = idx + 1
                with sekme:
                    st.markdown(f"### 🎯 {yil}. Yıl (Periyot) Karar Değişkenleri ve Ağ Analizi")
                    
                    df_tesis_p = df_tesis[df_tesis['Donem'] == yil] if not df_tesis.empty and 'Donem' in df_tesis.columns else pd.DataFrame()
                    df_z_p = df_z[df_z['Donem'] == yil] if not df_z.empty and 'Donem' in df_z.columns else pd.DataFrame()
                    df_w_p = df_w[df_w['Donem'] == yil] if not df_w.empty and 'Donem' in df_w.columns else pd.DataFrame()
                    df_stok_p = df_stok[df_stok['Donem'] == yil] if not df_stok.empty and 'Donem' in df_stok.columns else pd.DataFrame()

                    # Dönemsel Üst KPI Kartları
                    k1, k2, k3, k4 = st.columns(4)
                    p_toplanan = int(df_z_p['Miktar'].sum()) if not df_z_p.empty and 'Miktar' in df_z_p.columns else 0
                    p_tesis_sayisi = len(df_tesis_p) if not df_tesis_p.empty else 0
                    
                    with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Toplanan Batarya</div><div class='kpi-value'>{p_toplanan:,} br</div></div>", unsafe_allow_html=True)
                    with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Aktif Tesis Sayısı</div><div class='kpi-value'>{p_tesis_sayisi} Tesis</div></div>", unsafe_allow_html=True)
                    with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Akış Kalemi</div><div class='kpi-value'>{len(df_w_p)} İşlem</div></div>", unsafe_allow_html=True)
                    with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Ortalama Stok</div><div class='kpi-value'>{int(df_stok_p['Miktar'].mean()) if not df_stok_p.empty and 'Miktar' in df_stok_p.columns else 0:,} br</div></div>", unsafe_allow_html=True)

                    st.write("")
                    
                    # --- TÜM KARAR DEĞİŞKENLERİ KATEGORİK SEKMELERİ ---
                    st.markdown(f"#### 🔍 {yil}. Dönem Karar Değişkenleri Detaylı Tabloları")
                    
                    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                        "🏭 Tesis Açılış ($x1, x2, x3, x4$)", 
                        "📦 Şehir Toplama ($z, \\sigma$)", 
                        "🔄 Tesisler Arası Akış ($W1, W2, W3, W4$)", 
                        "📊 Envanter / Stok ($I1, I2, I3, I4$)",
                        "⚖️ Kütle & Dönüşüm ($\\beta, V, \\mu, \\partial$)"
                    ])
                    
                    with sub_tab1:
                        st.markdown("<p style='font-size:12px; color:#94a3b8;'>Modelin ikili (binary) tesis açılış kararları ($x1_{jt}, x2_{kt}, x3_{lt}, x4_{mt} \\in \\{0,1\\}$).</p>", unsafe_allow_html=True)
                        if not df_tesis_p.empty:
                            df_t_g = df_tesis_p.copy()
                            if 'Tesis_No_Str' in df_t_g.columns: df_t_g['Tesis_No'] = df_t_g['Tesis_No_Str']
                            st.dataframe(df_t_g, use_container_width=True)
                        else: st.info(f"{yil}. Yıl için tesis açılış kararı bulunamadı.")

                    with sub_tab2:
                        st.markdown("<p style='font-size:12px; color:#94a3b8;'>İllerden toplama merkezlerine atama ($\sigma_{ijbt}$) ve taşınan miktar ($z_{ijbt}$ değişkenleri).</p>", unsafe_allow_html=True)
                        if not df_z_p.empty:
                            df_z_g = df_z_p.copy()
                            if 'Sehir(i)_Str' in df_z_g.columns: df_z_g['Sehir(i)'] = df_z_g['Sehir(i)_Str']
                            if 'Toplama(j)_Str' in df_z_g.columns: df_z_g['Toplama(j)'] = df_z_g['Toplama(j)_Str']
                            st.dataframe(df_z_g, use_container_width=True)
                        else: st.info(f"{yil}. Yıl için toplama değişkeni ($z, \\sigma$) verisi bulunamadı.")

                    with sub_tab3:
                        st.markdown("<p style='font-size:12px; color:#94a3b8;'>Tesisler arası sevkiyat akışları ($W1_{jkbt}, W2_{klbt}, W3_{kmbt}, W4_{kbbt}$).</p>", unsafe_allow_html=True)
                        if not df_w_p.empty:
                            df_w_g = df_w_p.copy()
                            if 'Cikis_Str' in df_w_g.columns: df_w_g['Cikis'] = df_w_g['Cikis_Str']
                            if 'Varis_Str' in df_w_g.columns: df_w_g['Varis'] = df_w_g['Varis_Str']
                            st.dataframe(df_w_g, use_container_width=True)
                        else: st.info(f"{yil}. Yıl için akış değişkeni ($W$) verisi bulunamadı.")

                    with sub_tab4:
                        st.markdown("<p style='font-size:12px; color:#94a3b8;'>Dönem sonu envanter ve stok seviyeleri ($I1_{jbt}, I2_{kbt}, I3_{kbt}, I4_{lbt}$).</p>", unsafe_allow_html=True)
                        if not df_stok_p.empty:
                            df_s_g = df_stok_p.copy()
                            if 'Tesis_No_Str' in df_s_g.columns: df_s_g['Tesis_No'] = df_s_g['Tesis_No_Str']
                            st.dataframe(df_s_g, use_container_width=True)
                        else: st.info(f"{yil}. Yıl için stok değişkeni ($I$) verisi bulunamadı.")

                    with sub_tab5:
                        st.markdown("<p style='font-size:12px; color:#94a3b8;'>İşlenen kütle ($\beta_{kbt}$), ikinci ömür talep sevkiyatı ($V_{lsbt}$) ve hammadde geri kazanım kütleleri ($\mu, \\partial$).</p>", unsafe_allow_html=True)
                        # Örnek kütle / dönüştürme gösterim tablosu (W akışlarından türetilmiş kg karşılıkları)
                        if not df_w_p.empty and 'Miktar_kg' in df_w_p.columns:
                            st.dataframe(df_w_p[['Akis_Tipi', 'Batarya', 'Miktar', 'Miktar_kg']], use_container_width=True)
                        else:
                            st.info(f"{yil}. Yıl için detaylı kütle/dönüşüm değişken matrisi bu periyotta boş.")

                    # Dönemsel Grafikler
                    st.markdown("---")
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        if not df_z.empty and 'Donem' in df_z.columns and 'Miktar' in df_z.columns:
                            df_z_kumulatif = df_z[df_z['Donem'] <= yil].groupby('Toplama(j)')['Miktar'].sum().reset_index()
                            df_z_kumulatif['Il_Adi'] = df_z_kumulatif['Toplama(j)'].map(lambda idx: f"{int(idx)} ({aday_toplama.iloc[int(idx)]['Il_Adi']})" if str(idx).isdigit() and int(idx) < len(aday_toplama) else f"Merkez {idx}")
                            
                            fig_p1 = px.bar(df_z_kumulatif, x='Il_Adi', y='Miktar', title=f"1. Yıl - {yil}. Yıl Kümülatif İl Bazlı Toplama ($z$) Dağılımı", color_discrete_sequence=['#38bdf8'])
                            fig_p1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                            st.plotly_chart(fig_p1, use_container_width=True)
                        else:
                            st.info("Bu periyot için grafik verisi yok.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with gc2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        if not df_w_p.empty and 'Miktar_kg' in df_w_p.columns:
                            w_p_grouped = df_w_p.groupby('Akis_Tipi')['Miktar_kg'].sum().reset_index()
                            fig_p2 = px.pie(w_p_grouped, names='Akis_Tipi', values='Miktar_kg', title=f"{yil}. Yıl Akış Tipleri ($W$) Dağılımı (kg)", hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
                            fig_p2.update_traces(textinfo='percent+label', hovertemplate='%{label}<br>%{value:,.0f} kg<br>%{percent}')
                            fig_p2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                            st.plotly_chart(fig_p2, use_container_width=True)
                        else:
                            st.info("Bu periyot için grafik verisi yok.")
                        st.markdown("</div>", unsafe_allow_html=True)

            # --- GENEL HARİTA VE TÜMÜNÜ GÖSTEREN KOKPİT ---
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>🗺️ Genel Lojistik Akış Ağı Haritası</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:13px; color:#94a3b8; text-align:center;'>🔴 Sürekli Kırmızı Çizgiler: Toplama ➔ İşleme | ⚫ Kesikli Siyah Çizgiler: İşleme ➔ 2. Ömür / Geri Dönüşüm</p>", unsafe_allow_html=True)
            
            m_sonuc = folium.Map(location=[39.0, 35.0], zoom_start=6) 
            
            for idx, row in aday_toplama.iterrows():
                folium.Marker(location=[row['Enlem'], row['Boylam']], icon=folium.Icon(color="blue", icon="building", prefix='fa'), tooltip=f"Toplama Merkezi: {idx} ({row['Il_Adi']})").add_to(m_sonuc)
            for il in isleme_aday_illeri:
                folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="red", icon="industry", prefix='fa'), tooltip=f"İşleme Merkezi: {il['Il_Adi']}").add_to(m_sonuc)
            for il in minimax_adaylari:
                folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="green", icon="recycle", prefix='fa'), tooltip=f"2. Ömür / Geri Dönüşüm: {il['Il_Adi']}").add_to(m_sonuc)

            w1_donem = df_w[df_w['Akis_Tipi'].str.startswith('W1')].groupby('Donem')['Miktar'].sum() if not df_w.empty and 'Akis_Tipi' in df_w.columns else pd.Series()
            w23_donem = df_w[df_w['Akis_Tipi'].str.startswith('W2') | df_w['Akis_Tipi'].str.startswith('W3')].groupby('Donem')['Miktar'].sum() if not df_w.empty and 'Akis_Tipi' in df_w.columns else pd.Series()

            for index, toplama in aday_toplama.iterrows():
                en_kisa = float('inf'); hedef_isleme = None
                for isleme in isleme_aday_illeri:
                    mesafe = distance.euclidean([toplama['Enlem'], toplama['Boylam']], [isleme['Enlem'], isleme['Boylam']])
                    if mesafe < en_kisa: en_kisa = mesafe; hedef_isleme = isleme
                if hedef_isleme:
                    html = f"<div style='width:150px'><b>{toplama['Il_Adi']} ➔ {hedef_isleme['Il_Adi']} (Batarya Akışı)</b><hr style='margin:2px;'>"
                    for d, m_val in w1_donem.items(): html += f"Yıl {d}: {int(m_val/len(aday_toplama)):,} br<br>"
                    folium.PolyLine(
                        locations=[[toplama['Enlem'], toplama['Boylam']], [hedef_isleme['Enlem'], hedef_isleme['Boylam']]], 
                        color='#ef4444', weight=2.5, opacity=0.8, 
                        tooltip=folium.Tooltip(html+"</div>")
                    ).add_to(m_sonuc)

            for i, isleme_il in enumerate(isleme_aday_illeri):
                hedef_il = minimax_adaylari[i]
                html = f"<div style='width:150px'><b>{isleme_il['Il_Adi']} ➔ 2.Ömür / Geri Dönüşüm</b><hr style='margin:2px;'>"
                for d, m_val in w23_donem.items(): html += f"Yıl {d}: {int(m_val/len(isleme_aday_illeri)):,} br<br>"
                folium.PolyLine(
                    locations=[[isleme_il['Enlem'], isleme_il['Boylam']], [hedef_il['Enlem'], hedef_il['Boylam']]], 
                    color='#000000', weight=3, dash_array='8, 8', opacity=0.9, 
                    tooltip=folium.Tooltip(html+"</div>")
                ).add_to(m_sonuc)

            st_folium(m_sonuc, width=1400, height=550) 
            
        except Exception as e: st.error(f"Veri Okuma Hatası: {e}")

    # --- SİSTEM LOGLARI (AUDIT TRAIL) ---
    with st.expander("📋 Sistem Denetim ve İşlem Geçmişi (Audit Trail)"):
        if st.session_state.audit_logs:
            for log in st.session_state.audit_logs:
                st.write(log)
        else:
            st.write("Henüz bir işlem kaydedilmedi.")
