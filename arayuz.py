import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- SAYFA AYARLARI VE CSS ---
st.set_page_config(page_title="EV Batarya KDS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #fca311 !important;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #fca311; color: #000; font-weight: bold;}
    .stDataFrame {border-radius: 10px; overflow: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- SOL MENÜ (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063050.png", width=80)
st.sidebar.title("KDS Kontrol Paneli")
st.sidebar.markdown("---")

dosya_veri = st.sidebar.text_input("Girdi Verileri (.xlsx):", "seminerveriler_son.xlsx")
# DOSYA ADI BURADA KALICI OLARAK DÜZELTİLDİ:
dosya_sonuc = st.sidebar.text_input("Model Sonuçları (.xlsx):", "Model_Sonuclari.xlsx")

st.sidebar.markdown("---")

# Session State ile adımları kontrol etme
if 'adim' not in st.session_state:
    st.session_state.adim = 0

if st.sidebar.button("1. Adımı Çalıştır (Lokasyon Seçimi)"):
    st.session_state.adim = 1
if st.sidebar.button("2. Adımı Çalıştır (Gurobi Entegrasyonu)"):
    st.session_state.adim = 2
if st.sidebar.button("Sıfırla 🔄"):
    st.session_state.adim = 0
    st.rerun()

# --- KARŞILAMA EKRANI ---
if st.session_state.adim == 0:
    st.title("🔋 Elektrikli Araç Bataryaları Tersine Lojistik Ağı")
    st.markdown("""
    ### Karar Destek Sistemine (KDS) Hoş Geldiniz
    Bu sistem, Türkiye'deki emekliye ayrılmış lityum-iyon EV bataryalarının toplanması, işlenmesi ve geri dönüştürülmesi için optimum tedarik zinciri ağını tasarlar.
    
    * **Adım 1:** Güncel araç taleplerini analiz eder, dinamik Pareto (%80) mantığıyla toplama merkezlerini belirler ve K-Means ile işleme adaylarını seçer.
    * **Adım 2:** Gurobi matematiksel optimizasyon modelinin sonuçlarını okuyarak tesis açılış kararlarını ve lojistik akış ağını interaktif haritada görselleştirir.
    
    👈 *Başlamak için sol menüdeki butonları kullanın.*
    """)

# --- ORTAK FONKSİYON: DİNAMİK PARETO ---
def dinamik_pareto_hesapla(df, talep_sutunu='Toplam_Talep', hedef=0.80):
    """
    Veriyi büyükten küçüğe sıralar, kümülatif toplamı hesaplar 
    ve %80'e mutlak olarak en yakın sonucu veren il sayısını otomatik seçer.
    """
    df_sirali = df.sort_values(by=talep_sutunu, ascending=False).reset_index(drop=True)
    df_sirali['Kumulatif_Yuzde'] = df_sirali[talep_sutunu].cumsum() / df_sirali[talep_sutunu].sum()
    # 0.80'e olan mutlak farkı hesapla
    df_sirali['Fark'] = abs(df_sirali['Kumulatif_Yuzde'] - hedef)
    # Farkın en küçük olduğu satırın indeksini (optimum kesim noktasını) bul
    optimum_index = df_sirali['Fark'].idxmin()
    # O indekse kadar olan tüm illeri al
    return df_sirali.iloc[:optimum_index + 1].copy()

# --- ADIM 1 VE 2 İÇİN ANA GÖVDE ---
if st.session_state.adim in [1, 2]:
    try:
        # 1. Excel Verilerini Okuma
        df_koordinat = pd.read_excel(dosya_veri, sheet_name="Koordinatlar")
        
        # --- DİNAMİK PARETO UYGULAMASI ---
        aday_toplama = dinamik_pareto_hesapla(df_koordinat)
        
        # --- K-MEANS İŞLEME MERKEZLERİ (3 ADET) ---
        X = aday_toplama[['Enlem', 'Boylam']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        aday_toplama['Küme'] = kmeans.fit_predict(X)
        isleme_merkezleri = kmeans.cluster_centers_

        st.title(f"📊 {'1. Aşama: Aday Lokasyonlar' if st.session_state.adim == 1 else '2. Aşama: Optimize Edilmiş Lojistik Ağ (Gurobi)'}")
        
        # --- 2x2 DASHBOARD ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Talep (EV Batarya)", f"{df_koordinat['Toplam_Talep'].sum():,.0f}")
        with col2:
            # Artık dinamik hesaplandığı için kaç il çıkarsa onu yazar
            st.metric("Aday Toplama Merkezi (Dinamik Pareto)", f"{len(aday_toplama)} İl")

        # --- HARİTA OLUŞTURMA (CartoDB Dark Matter) ---
        m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB dark_matter")

        # Toplama Merkezlerini Haritaya Ekle (Bina İkonu)
        for _, row in aday_toplama.iterrows():
            folium.Marker(
                location=[row['Enlem'], row['Boylam']],
                popup=f"{row['Il_Adi']} (Talep: {row['Toplam_Talep']})",
                tooltip=row['Il_Adi'],
                icon=folium.Icon(color="lightblue", icon="building", prefix="fa")
            ).add_to(m)

        # İşleme Merkezlerini Haritaya Ekle (Fabrika İkonu)
        for i, center in enumerate(isleme_merkezleri):
            folium.Marker(
                location=[center[0], center[1]],
                popup=f"İşleme Adayı {i+1}",
                tooltip="İşleme Merkezi",
                icon=folium.Icon(color="red", icon="industry", prefix="fa")
            ).add_to(m)

        # --- ADIM 2: GUROBİ AKIŞLARI (ÖRÜMCEK AĞI) ---
        if st.session_state.adim == 2:
            st.success("Gurobi Optimizasyon Sonuçları Haritaya İşlendi!")
            try:
                # Gurobi sonuç dosyasını okuma (W1 akışları örneği)
                df_w1 = pd.read_excel(dosya_sonuc, sheet_name="W1")
                # Eğer Excel'inde akışlar (Başlangıç_Enlem, Başlangıç_Boylam, Bitiş_Enlem, Bitiş_Boylam, Miktar) formatındaysa:
                for _, row in df_w1.iterrows():
                    if row['Miktar'] > 0:  # Sadece akış olan rotaları çiz
                        baslangic = [row['Baslangic_Enlem'], row['Baslangic_Boylam']]
                        bitis = [row['Bitis_Enlem'], row['Bitis_Boylam']]
                        
                        # Örümcek Ağı Çizgisi ve Tooltip
                        folium.PolyLine(
                            locations=[baslangic, bitis],
                            color='#fca311',
                            weight=2,
                            opacity=0.7,
                            tooltip=f"Akış Miktarı: {row['Miktar']} Birim"
                        ).add_to(m)
            except Exception as e:
                st.warning(f"Akış çizgileri çizilirken bir hata oluştu veya W1 sayfası bulunamadı. Lütfen Excel yapısını kontrol edin. Detay: {e}")

        # Haritayı ve Pareto Tablosunu Ekrana Basma
        col_map, col_table = st.columns([3, 1])
        with col_map:
            st_folium(m, width=900, height=600)
        with col_table:
            st.markdown("### Dinamik Pareto Listesi")
            st.caption("%80'e en yakın optimum kesim")
            # Sınırlama olmadan tüm tespit edilen illeri göster
            st.dataframe(aday_toplama[['Il_Adi', 'Toplam_Talep', 'Kumulatif_Yuzde']], use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"❌ Dosya bulunamadı! Lütfen sol menüden dosya adlarının doğruluğunu kontrol et. Hata Detayı: {e}")
    except Exception as e:
        st.error(f"❌ Beklenmeyen bir hata oluştu: {e}")
