import streamlit as st
import pandas as pd
import numpy as np
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

# --- YAY ÇİZGİSİ (BEZIER CURVE) ALGORİTMASI ---
def kavisli_yay_hesapla(p1, p2, offset=0.15, num_points=30):
    """İki nokta arasına kalın ve kavisli (yay şeklinde) çizgi çekmek için noktalar dizisi oluşturur"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0: return [p1.tolist(), p2.tolist()]
    mid = (p1 + p2) / 2
    direction = p2 - p1
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)
    control = mid + normal * (dist * offset)
    t = np.linspace(0, 1, num_points)
    curve = np.outer((1-t)**2, p1) + np.outer(2*(1-t)*t, control) + np.outer(t**2, p2)
    return curve.tolist()

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

        for _, row in aday_toplama.iterrows():
            folium.Marker(
                location=[row['Enlem'], row['Boylam']],
                popup=f"{row['Il_Adi']} (Talep: {row['Toplam_Talep']:,.0f})",
                tooltip=row['Il_Adi'] + " (Toplama Merkezi)",
                icon=folium.Icon(color="lightblue", icon="building", prefix="fa")
            ).add_to(m)

        for i, center in enumerate(isleme_merkezleri):
            folium.Marker(
                location=[center[0], center[1]],
                popup=f"İşleme Adayı {i}",
                tooltip=f"İşleme Merkezi {i}",
                icon=folium.Icon(color="red", icon="industry", prefix="fa")
            ).add_to(m)

        # --- ADIM 2: GUROBİ AKIŞLARI (KAVİSLİ EĞRİLER) ---
        if st.session_state.adim == 2:
            st.success("Gurobi Optimizasyon Sonuçları Haritaya İşlendi! (Tüm Batarya ve Hücre Akışları)")
            
            try:
                # 1. Z Akışları (İllerden -> Toplama Merkezlerine | BATARYA)
                df_z = pd.read_excel(dosya_sonuc, sheet_name="Z_Sehir_Toplama")
                for _, row in df_z.iterrows():
                    miktar = float(row.get('Miktar', 0))
                    if miktar > 0:
                        c_idx, v_idx = int(row['Cikis']), int(row['Varis'])
                        
                        bas_isim = df_koordinat.iloc[c_idx]['Il_Adi'] + " (İl)"
                        bas_k = [df_koordinat.iloc[c_idx]['Enlem'], df_koordinat.iloc[c_idx]['Boylam']]
                        
                        bitis_isim = aday_toplama.iloc[v_idx]['Il_Adi'] + " (Toplama)"
                        bitis_k = [aday_toplama.iloc[v_idx]['Enlem'], aday_toplama.iloc[v_idx]['Boylam']]
                        
                        yay = kavisli_yay_hesapla(bas_k, bitis_k, offset=0.1)
                        bilgi = f"<div style='font-family: Arial; font-size: 12px;'><b>Tip:</b> Batarya (Sürekli)<br><b>Nereden:</b> {bas_isim}<br><b>Nereye:</b> {bitis_isim}<br><b>Miktar:</b> {miktar:,.0f} Batarya</div>"
                        
                        # Batarya -> Kalın, Sürekli Siyah Çizgi
                        folium.PolyLine(locations=yay, color='black', weight=4, opacity=0.7, tooltip=bilgi).add_to(m)
            except Exception as e:
                pass # Z sayfası yoksa diğer akışlara geç

            try:
                # 2. W Akışları (Toplama->İşleme, İşleme->İkinci Ömür/Geri Dönüşüm)
                df_w = pd.read_excel(dosya_sonuc, sheet_name="W_Tesis_Akislari")
                for _, row in df_w.iterrows():
                    miktar = float(row.get('Miktar', 0))
                    if miktar <= 0: continue
                    
                    akis_tipi = str(row['Akis_Tipi'])
                    c_idx = int(row['Cikis'])
                    v_val = row['Varis']
                    
                    if 'W1' in akis_tipi:
                        # Toplama -> İşleme | BATARYA
                        v_idx = int(v_val)
                        bas_isim = aday_toplama.iloc[c_idx]['Il_Adi'] + " (Toplama)"
                        bas_k = [aday_toplama.iloc[c_idx]['Enlem'], aday_toplama.iloc[c_idx]['Boylam']]
                        
                        bitis_isim = f"İşleme Adayı {v_idx}"
                        bitis_k = [isleme_merkezleri[v_idx][0], isleme_merkezleri[v_idx][1]]
                        
                        yay = kavisli_yay_hesapla(bas_k, bitis_k, offset=0.15)
                        bilgi = f"<div style='font-family: Arial; font-size: 12px;'><b>Tip:</b> Batarya (Sürekli)<br><b>Nereden:</b> {bas_isim}<br><b>Nereye:</b> {bitis_isim}<br><b>Miktar:</b> {miktar:,.0f} Batarya</div>"
                        
                        # Batarya -> Kalın, Sürekli Siyah Çizgi
                        folium.PolyLine(locations=yay, color='black', weight=4, opacity=0.8, tooltip=bilgi).add_to(m)
                        
                    elif 'W2' in akis_tipi or 'W3' in akis_tipi:
                        # İşleme -> İkinci Ömür / Geri Dönüşüm | HÜCRE
                        bas_isim = f"İşleme Adayı {c_idx}"
                        bas_k = [isleme_merkezleri[c_idx][0], isleme_merkezleri[c_idx][1]]
                        
                        try:
                            v_idx = int(v_val)
                            if v_idx < len(aday_toplama):
                                bitis_k = [aday_toplama.iloc[v_idx]['Enlem'], aday_toplama.iloc[v_idx]['Boylam']]
                                il_adi = aday_toplama.iloc[v_idx]['Il_Adi']
                            else:
                                bitis_k = [df_koordinat.iloc[v_idx]['Enlem'], df_koordinat.iloc[v_idx]['Boylam']]
                                il_adi = df_koordinat.iloc[v_idx]['Il_Adi']
                            bitis_isim = il_adi + (" (2. Ömür)" if 'W2' in akis_tipi else " (Geri Dönüşüm)")
                        except:
                            # 'Bertaraf' vb. string ifadeler gelirse sembolik çıkış
                            bitis_k = [bas_k[0] + 0.3, bas_k[1] + 0.3]
                            bitis_isim = str(v_val)
                            
                        yay = kavisli_yay_hesapla(bas_k, bitis_k, offset=0.15)
                        bilgi = f"<div style='font-family: Arial; font-size: 12px;'><b>Tip:</b> Hücre (Kesikli)<br><b>Nereden:</b> {bas_isim}<br><b>Nereye:</b> {bitis_isim}<br><b>Miktar:</b> {miktar:,.0f} Hücre</div>"
                        
                        # Hücre -> Kalın, Kesikli (Dashed) Siyah Çizgi
                        folium.PolyLine(locations=yay, color='black', weight=4, dash_array='10, 10', opacity=0.9, tooltip=bilgi).add_to(m)
            except Exception as e:
                st.warning(f"Akış çizgileri çizilirken bir hata oluştu. Detay: {e}")

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

    except ValueError as e:
        st.error(f"❌ Excel okuma hatası! Lütfen sayfaların doğruluğunu kontrol edin. Detay: {e}")
    except Exception as e:
        st.error(f"❌ Beklenmeyen bir hata oluştu: {e}")
