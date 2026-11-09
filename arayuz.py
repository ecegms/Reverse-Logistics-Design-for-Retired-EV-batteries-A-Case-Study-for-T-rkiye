import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

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
dosya_sonuc = st.sidebar.text_input("Model Sonuçları (.xlsx):", "Model_Sonuclari.xlsx")

st.sidebar.markdown("---")

if 'adim' not in st.session_state:
    st.session_state.adim = 0

if st.sidebar.button("1. Adımı Çalıştır (Lokasyon Seçimi)"):
    st.session_state.adim = 1
if st.sidebar.button("2. Adımı Çalıştır (Gurobi Entegrasyonu)"):
    st.session_state.adim = 2
if st.sidebar.button("Sıfırla 🔄"):
    st.session_state.adim = 0
    st.rerun()

# --- ORTAK FONKSİYON: DİNAMİK PARETO ---
def dinamik_pareto_hesapla(df, talep_sutunu='Toplam_Talep', hedef=0.80):
    df = df.dropna(subset=[talep_sutunu]).copy()
    df[talep_sutunu] = pd.to_numeric(df[talep_sutunu], errors='coerce').fillna(0)
    df_sirali = df.sort_values(by=talep_sutunu, ascending=False).reset_index(drop=True)
    df_sirali['Kumulatif_Yuzde'] = df_sirali[talep_sutunu].cumsum() / df_sirali[talep_sutunu].sum()
    df_sirali['Fark'] = abs(df_sirali['Kumulatif_Yuzde'] - hedef)
    optimum_index = df_sirali['Fark'].idxmin()
    return df_sirali.iloc[:optimum_index + 1].copy()

# --- ADIM 1 VE 2 İÇİN ANA GÖVDE ---
if st.session_state.adim in [1, 2]:
    try:
        # 1. Excel Verilerini Oku
        df_talep = pd.read_excel(dosya_veri, sheet_name="Sayfa1")
        df_koor = pd.read_excel(dosya_veri, sheet_name="Koordinatlar")
        
        df_talep = df_talep.rename(columns={'Şehir (İl)': 'Il_Adi', 'İldeki Elektrikli Araç Sayısı': 'Toplam_Talep'})
        df_talep['Il_Adi'] = df_talep['Il_Adi'].astype(str).str.strip()
        df_koor['Il_Adi'] = df_koor['Il_Adi'].astype(str).str.strip()
        
        df_koordinat = pd.merge(df_talep, df_koor[['Il_Adi', 'Enlem', 'Boylam']], on='Il_Adi', how='inner')
            
        # --- DİNAMİK PARETO VE K-MEANS ---
        aday_toplama = dinamik_pareto_hesapla(df_koordinat)
        X = aday_toplama[['Enlem', 'Boylam']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        aday_toplama['Küme'] = kmeans.fit_predict(X)
        isleme_merkezleri = kmeans.cluster_centers_

        st.title(f"📊 {'1. Aşama: Aday Lokasyonlar' if st.session_state.adim == 1 else '2. Aşama: Optimize Edilmiş Lojistik Ağ (Gurobi)'}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Talep (EV Batarya)", f"{df_koordinat['Toplam_Talep'].sum():,.0f}")
        with col2:
            st.metric("Aday Toplama Merkezi (Dinamik Pareto)", f"{len(aday_toplama)} İl")

        # --- HARİTA OLUŞTURMA ---
        m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="OpenStreetMap")

        # Toplama Merkezi Pinleri (Mavi)
        for _, row in aday_toplama.iterrows():
            folium.Marker(
                location=[row['Enlem'], row['Boylam']],
                tooltip=row['Il_Adi'] + " (Toplama)",
                icon=folium.Icon(color="lightblue", icon="building", prefix="fa")
            ).add_to(m)

        # İşleme Merkezi Pinleri (Kırmızı)
        for i, center in enumerate(isleme_merkezleri):
            folium.Marker(
                location=[center[0], center[1]],
                tooltip=f"İşleme Tesisi {i}",
                icon=folium.Icon(color="red", icon="industry", prefix="fa")
            ).add_to(m)

        # --- ADIM 2: GUROBİ AKIŞLARI (DÜZ, KESİKLİ TURKUAZ ÇİZGİLER VE DETAYLI TOOLTIP) ---
        if st.session_state.adim == 2:
            st.success("Gurobi Optimizasyon Sonuçları Haritaya İşlendi! (Tesisler Arası Akışlar)")
            
            # Not: Z (Şehir -> Toplama) akışlarının haritayı karmaşıklaştırmaması için çizimi kaldırıldı.

            # Sadece W Akışları (Toplama->İşleme, İşleme->İkinci Ömür/Geri Dönüşüm)
            try:
                df_w = pd.read_excel(dosya_sonuc, sheet_name="W_Tesis_Akislari")
                df_w = df_w[df_w['Miktar'] > 0]
                
                w_grouped = df_w.groupby(['Cikis', 'Varis', 'Akis_Tipi'])
                
                for (c_idx, v_val, akis_tipi), group in w_grouped:
                    c_idx = int(c_idx)
                    akis_tipi = str(akis_tipi)
                    
                    if 'W1' in akis_tipi:
                        # Toplama -> İşleme
                        v_idx = int(v_val)
                        bas_isim = aday_toplama.iloc[c_idx]['Il_Adi'] + " (Toplama)"
                        bitis_isim = f"İşleme {v_idx}"
                        bas_k = [aday_toplama.iloc[c_idx]['Enlem'], aday_toplama.iloc[c_idx]['Boylam']]
                        bitis_k = [isleme_merkezleri[v_idx][0], isleme_merkezleri[v_idx][1]]
                        
                        tooltip_html = f"<div style='font-family: sans-serif; font-size: 13px; min-width: 120px;'>"
                        tooltip_html += f"<b style='font-size: 14px;'>{bas_isim} ➔ {bitis_isim}</b><hr style='margin: 4px 0;'>"
                        for _, r in group.iterrows():
                            donem = int(r.get('Donem', 1))
                            tooltip_html += f"Dönem {donem}: {float(r['Miktar']):,.0f} br<br>"
                        tooltip_html += "</div>"
                        
                        folium.PolyLine(locations=[bas_k, bitis_k], color='teal', weight=2.5, dash_array='8, 8', opacity=0.8, tooltip=tooltip_html).add_to(m)
                        
                    elif 'W2' in akis_tipi or 'W3' in akis_tipi:
                        # İşleme -> İkinci Ömür / Geri Dönüşüm
                        bas_isim = f"İşleme {c_idx}"
                        bas_k = [isleme_merkezleri[c_idx][0], isleme_merkezleri[c_idx][1]]
                        
                        if str(v_val).isdigit():
                            v_idx = int(v_val)
                            bitis_k = [df_koordinat.iloc[v_idx]['Enlem'], df_koordinat.iloc[v_idx]['Boylam']]
                            il_adi = df_koordinat.iloc[v_idx]['Il_Adi']
                            
                            if 'W2' in akis_tipi:
                                bitis_isim = il_adi + " (2. Ömür)"
                                folium.Marker(location=bitis_k, tooltip=bitis_isim, icon=folium.Icon(color="green", icon="leaf", prefix="fa")).add_to(m)
                            else:
                                bitis_isim = il_adi + " (G. Dönüşüm)"
                                folium.Marker(location=bitis_k, tooltip=bitis_isim, icon=folium.Icon(color="purple", icon="recycle", prefix="fa")).add_to(m)
                        else:
                            bitis_k = [bas_k[0] + 0.3, bas_k[1] + 0.3]
                            bitis_isim = str(v_val)
                            
                        tooltip_html = f"<div style='font-family: sans-serif; font-size: 13px; min-width: 120px;'>"
                        tooltip_html += f"<b style='font-size: 14px;'>{bas_isim} ➔ {bitis_isim}</b><hr style='margin: 4px 0;'>"
                        for _, r in group.iterrows():
                            donem = int(r.get('Donem', 1))
                            tooltip_html += f"Dönem {donem}: {float(r['Miktar']):,.0f} br<br>"
                        tooltip_html += "</div>"
                        
                        folium.PolyLine(locations=[bas_k, bitis_k], color='teal', weight=2.5, dash_array='8, 8', opacity=0.8, tooltip=tooltip_html).add_to(m)
            except Exception as e:
                st.warning(f"W Akışları çizilemedi. Hata: {e}")

        # Haritayı ve Tabloyu Ekrana Basma
        col_map, col_table = st.columns([3, 1])
        with col_map:
            st_folium(m, width=900, height=600)
        with col_table:
            st.markdown("### Dinamik Pareto Listesi")
            st.caption("%80'e en yakın optimum kesim")
            tablo_gosterim = aday_toplama[['Il_Adi', 'Toplam_Talep', 'Kumulatif_Yuzde']].copy()
            tablo_gosterim['Kumulatif_Yuzde'] = (tablo_gosterim['Kumulatif_Yuzde'] * 100).round(2).astype(str) + '%'
            st.dataframe(tablo_gosterim, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Beklenmeyen bir hata oluştu: {e}")
