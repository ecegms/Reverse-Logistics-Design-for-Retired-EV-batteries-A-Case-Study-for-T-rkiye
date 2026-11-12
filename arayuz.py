import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.express as px

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="EV Batarya Geri Dönüşüm KDS", 
    page_icon="🔋", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- PROFESYONEL KOYU MAVİ / SİYAH TEMA (CSS) ---
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
    hr {border-color: #334155;}
    </style>
""", unsafe_allow_html=True)

# --- HAFIZA (SESSION STATE) ---
if "hesaplandi" not in st.session_state: st.session_state.hesaplandi = False
if "model_calisti" not in st.session_state: st.session_state.model_calisti = False

# --- SOL MENÜ (KONTROL PANELİ) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933991.png", width=50)
    st.header("⚙️ Veri & Sistem Yönetimi")
    st.markdown("<p style='font-size:12px; color:#94a3b8;'>Model girdilerini ve optimizasyon çıktı dosyalarını buradan yönetin.</p>", unsafe_allow_html=True)
    
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
        
        # MODEL SONUÇLARI İNDİRME BUTONU
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
            st.rerun()

# =====================================================
# ADIM 1: ADAY MERKEZLERİN BELİRLENMESİ
# =====================================================
if st.session_state.hesaplandi and not st.session_state.model_calisti:
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
if st.session_state.model_calisti:
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

        df_w = pd.read_excel(dosya_sonuc, sheet_name='W_Tesis_Akislari')
        df_z = pd.read_excel(dosya_sonuc, sheet_name='Z_Sehir_Toplama')
        df_stok = pd.read_excel(dosya_sonuc, sheet_name='Stoklar_I1_I2_I3_I4')
        df_tesis = pd.read_excel(dosya_sonuc, sheet_name='Tesis_Acilis_Kararlari')
        
        # --- TABLOLAR İÇİN İSİM EŞLEŞTİRMELERİ (KOD + ŞEHİR ADI) ---
        if not df_z.empty:
            df_z = df_z.copy()
            df_z['Sehir(i)_Str'] = df_z['Sehir(i)'].map(lambda x: f"{int(x)} ({df_koordinat.iloc[int(x)]['Il_Adi']})" if pd.notnull(x) and int(x) < len(df_koordinat) else str(x))
            df_z['Toplama(j)_Str'] = df_z['Toplama(j)'].map(lambda x: f"{int(x)} ({aday_toplama.iloc[int(x)]['Il_Adi']})" if pd.notnull(x) and int(x) < len(aday_toplama) else str(x))

        if not df_w.empty:
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

        if not df_tesis.empty:
            df_tesis = df_tesis.copy()
            def map_tesis_no(val):
                try:
                    v = int(val)
                    if v < len(isleme_aday_illeri): return f"{v} ({isleme_aday_illeri[v]['Il_Adi']})"
                    elif v < len(aday_toplama): return f"{v} ({aday_toplama.iloc[v]['Il_Adi']})"
                except: pass
                return str(val)
            df_tesis['Tesis_No_Str'] = df_tesis['Tesis_No'].apply(map_tesis_no)

        if not df_stok.empty:
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
            akis = str(row['Akis_Tipi'])
            miktar = float(row['Miktar'])
            if 'W1' in akis:
                return miktar * agirlik_batarya.get(b_tipi, 0)
            else:
                return miktar * agirlik_hucre.get(b_tipi, 0)
                
        if not df_w.empty:
            df_w['Miktar_kg'] = df_w.apply(hesapla_kg, axis=1)

        st.success("✅ Optimizasyon modeli başarıyla çözüldü. 5 farklı periyoda ait dönemsel ağ kararları ve performans analizleri aşağıdadır.")

        # --- YENİ EKLENEN: FİNANSAL ÖZET KARTI ---
        try:
            df_finans = pd.read_excel(dosya_sonuc, sheet_name='Finansal_Ozet')
            toplam_kar = df_finans['Toplam_Kar'].iloc[0]
            toplam_gelir = df_finans['Toplam_Gelir'].iloc[0]
            toplam_maliyet = df_finans['Toplam_Maliyet'].iloc[0]
        except:
            toplam_kar = 0
            toplam_gelir = 0
            toplam_maliyet = 0
            st.warning("⚠️ Model_Sonuclari.xlsx dosyasında 'Finansal_Ozet' sayfası bulunamadı. Değerler 0 gösterilmektedir.")

        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: #1e293b; border-radius: 10px; border-left: 5px solid #10b981; margin-bottom: 25px;">
            <h3 style="color: #10b981; margin-bottom: 5px; margin-top: 0;">Maksimum Kar (Amaç Fonksiyonu Z): ${toplam_kar:,.2f}</h3>
            <p style="color: #94a3b8; font-size: 13px; margin: 0;">
                📈 Toplam Beklenen Gelir: <b>${toplam_gelir:,.2f}</b> &nbsp; | &nbsp; 📉 Toplam Sistem Maliyeti: <b>${toplam_maliyet:,.2f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        # ----------------------------------------

        # --- 5 FARKLI PERİYOT İÇİN SEKMELER ---
        periyot_sekmeleri = st.tabs(["📅 1. Yıl", "📅 2. Yıl", "📅 3. Yıl", "📅 4. Yıl", "📅 5. Yıl"])
        
        for idx, sekme in enumerate(periyot_sekmeleri):
            yil = idx + 1
            with sekme:
                st.markdown(f"### 🎯 {yil}. Yıl (Periyot) Analiz ve Karar Kokpiti")
                
                df_tesis_p = df_tesis[df_tesis['Donem'] == yil] if 'Donem' in df_tesis.columns else pd.DataFrame()
                df_z_p = df_z[df_z['Donem'] == yil] if 'Donem' in df_z.columns else pd.DataFrame()
                df_w_p = df_w[df_w['Donem'] == yil] if 'Donem' in df_w.columns else pd.DataFrame()
                df_stok_p = df_stok[df_stok['Donem'] == yil] if 'Donem' in df_stok.columns else pd.DataFrame()

                k1, k2, k3, k4 = st.columns(4)
                p_toplanan = int(df_z_p['Miktar'].sum()) if not df_z_p.empty else 0
                p_tesis_sayisi = len(df_tesis_p) if not df_tesis_p.empty else 0
                
                with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Toplanan Batarya</div><div class='kpi-value'>{p_toplanan:,} br</div></div>", unsafe_allow_html=True)
                with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Aktif Tesis Sayısı</div><div class='kpi-value'>{p_tesis_sayisi} Tesis</div></div>", unsafe_allow_html=True)
                with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Akış Kalemi</div><div class='kpi-value'>{len(df_w_p)} İşlem</div></div>", unsafe_allow_html=True)
                with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{yil}. Yıl Ortalama Stok</div><div class='kpi-value'>{int(df_stok_p['Miktar'].mean()) if not df_stok_p.empty else 0:,} br</div></div>", unsafe_allow_html=True)

                st.write("")
                
                t1, t2, t3, t4 = st.tabs([f"🏭 {yil}. Yıl Açılan Tesisler", f"📦 {yil}. Yıl Şehir Toplamaları (Z)", f"🔄 {yil}. Yıl Akışlar (W)", f"📊 {yil}. Yıl Stoklar"])
                
                with t1:
                    if not df_tesis_p.empty:
                        df_tesis_gosterim = df_tesis_p.copy()
                        df_tesis_gosterim['Tesis_No'] = df_tesis_gosterim['Tesis_No_Str']
                        st.dataframe(df_tesis_gosterim[['Donem', 'Tesis_Tipi', 'Tesis_No']], use_container_width=True)
                    else: st.info(f"{yil}. Yıl için aktif tesis yatırım kaydı bulunamadı.")
                with t2:
                    if not df_z_p.empty:
                        df_z_gosterim = df_z_p.copy()
                        df_z_gosterim['Sehir(i)'] = df_z_gosterim['Sehir(i)_Str']
                        df_z_gosterim['Toplama(j)'] = df_z_gosterim['Toplama(j)_Str']
                        st.dataframe(df_z_gosterim[['Donem', 'Sehir(i)', 'Toplama(j)', 'Batarya(b)', 'Miktar']], use_container_width=True)
                    else: 
                        st.info(f"{yil}. Yıl için şehir toplama verisi bulunamadı.")
                with t3:
                    if not df_w_p.empty:
                        df_w_gosterim = df_w_p.copy()
                        df_w_gosterim['Cikis'] = df_w_gosterim['Cikis_Str']
                        df_w_gosterim['Varis'] = df_w_gosterim['Varis_Str']
                        cols_to_show = [c for c in ['Donem', 'Akis_Tipi', 'Cikis', 'Varis', 'Batarya', 'Miktar', 'Miktar_kg'] if c in df_w_gosterim.columns]
                        st.dataframe(df_w_gosterim[cols_to_show], use_container_width=True)
                    else: st.info(f"{yil}. Yıl için akış verisi bulunamadı.")
                with t4:
                    if not df_stok_p.empty:
                        df_stok_gosterim = df_stok_p.copy()
                        df_stok_gosterim['Tesis_No'] = df_stok_gosterim['Tesis_No_Str']
                        st.dataframe(df_stok_gosterim[['Donem', 'Stok_Tipi', 'Tesis_No', 'Batarya', 'Miktar']], use_container_width=True)
                    else: st.info(f"{yil}. Yıl için stok verisi bulunamadı.")

                # Dönemsel Grafikler
                st.markdown("---")
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    if not df_z.empty and 'Donem' in df_z.columns:
                        df_z_kumulatif = df_z[df_z['Donem'] <= yil].groupby('Toplama(j)')['Miktar'].sum().reset_index()
                        df_z_kumulatif['Il_Adi'] = df_z_kumulatif['Toplama(j)'].map(lambda idx: f"{int(idx)} ({aday_toplama.iloc[int(idx)]['Il_Adi']})" if int(idx) < len(aday_toplama) else f"Merkez {idx}")
                        
                        fig_p1 = px.bar(df_z_kumulatif, x='Il_Adi', y='Miktar', title=f"1. Yıl - {yil}. Yıl Kümülatif İl Bazlı Toplama Dağılımı", color_discrete_sequence=['#38bdf8'])
                        fig_p1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_p1, use_container_width=True)
                    else:
                        st.info("Bu periyot için grafik verisi yok.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with gc2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    if not df_w_p.empty:
                        w_p_grouped = df_w_p.groupby('Akis_Tipi')['Miktar_kg'].sum().reset_index()
                        fig_p2 = px.pie(w_p_grouped, names='Akis_Tipi', values='Miktar_kg', title=f"{yil}. Yıl Akış Tipi Dağılımı (kg)", hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
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

        w1_donem = df_w[df_w['Akis_Tipi'].str.startswith('W1')].groupby('Donem')['Miktar'].sum() if not df_w.empty else pd.Series()
        w23_donem = df_w[df_w['Akis_Tipi'].str.startswith('W2') | df_w['Akis_Tipi'].str.startswith('W3')].groupby('Donem')['Miktar'].sum() if not df_w.empty else pd.Series()

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
