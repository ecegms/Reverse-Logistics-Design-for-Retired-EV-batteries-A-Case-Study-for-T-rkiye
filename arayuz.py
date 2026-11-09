import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.express as px

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Tedarik Zinciri KDS", layout="wide", initial_sidebar_state="expanded")

# --- ÖZEL CSS TASARIMI ---
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #fca311 !important; font-family: 'Segoe UI', sans-serif;}
    .stButton>button {
        width: 100%; border-radius: 8px; background-color: #fca311; color: #14213d; 
        font-weight: bold; border: none; padding: 10px; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #e5e5e5; color: #14213d;}
    .css-1d391kg {background-color: #14213d;} /* Sol menü arkaplanı */
    .metric-card {
        background-color: #1e2430; padding: 20px; border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    hr {border-color: #333;}
    </style>
""", unsafe_allow_html=True)

# --- HAFIZA (SESSION STATE) ---
if "hesaplandi" not in st.session_state: st.session_state.hesaplandi = False
if "model_calisti" not in st.session_state: st.session_state.model_calisti = False

# --- SOL MENÜ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933991.png", width=60)
    st.header("⚙️ Veri Bağlantıları")
    dosya_girdi = st.text_input("Girdi Verileri (.xlsx):", r"seminerveriler_son.xlsx")
    dosya_sonuc = st.text_input("Model Sonuçları (.xlsx):", r"Model_Sonuclari_ilk.xlsx")
    st.markdown("---")
    if st.button("🔄 Sistemi Sıfırla (Başa Dön)"):
        st.session_state.hesaplandi = False
        st.session_state.model_calisti = False
        st.rerun()

# --- ANA BAŞLIK ---
st.title("🔋 EV Batarya Geri Dönüşüm Lojistik Ağı")
st.markdown("---")

# =====================================================
# KARŞILAMA EKRANI
# =====================================================
if not st.session_state.hesaplandi and not st.session_state.model_calisti:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='metric-card' style='text-align: center;'>
            <h3 style='color: white;'>KDS Kontrol Paneline Hoş Geldiniz</h3>
            <p style='color: #ccc; font-size: 16px;'>
                Bu Karar Destek Sistemi (KDS), Türkiye genelindeki elektrikli araç bataryalarının toplanması, 
                işlenmesi ve geri dönüştürülmesi için optimal tedarik zinciri ağını tasarlamak üzere geliştirilmiştir.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🚀 1. Adım: Aday Merkezleri Belirle (Pareto & Kümeleme)"):
            st.session_state.hesaplandi = True
            st.rerun()

# =====================================================
# ADIM 1: ADAY MERKEZLERİN BELİRLENMESİ
# =====================================================
if st.session_state.hesaplandi and not st.session_state.model_calisti:
    try:
        df_koordinat = pd.read_excel(dosya_girdi, sheet_name='Koordinatlar')
        df_DE = pd.read_excel(dosya_girdi, sheet_name='DE')
        veri_DE = df_DE.iloc[:, -5:].values 
        df_koordinat['Toplam_Talep'] = [np.sum(veri_DE[2*i : 2*i+2]) for i in range(81)]
        
        # --- PARETO ANALİZİ DÜZELTİLDİ ---
        df_pareto = df_koordinat.sort_values(by='Toplam_Talep', ascending=False).copy()
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
            isleme_aday_illeri.append({'Il_Adi': il_bilgisi['Il_Adi'], 'Enlem': il_bilgisi['Enlem'], 'Boylam': il_bilgisi['Boylam']})
            
        minimax_adaylari = [{'Il_Adi': il['Il_Adi'], 'Enlem': il['Enlem'] + 0.15, 'Boylam': il['Boylam'] + 0.15} for il in isleme_aday_illeri]
            
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("📊 1. Aşama Sonuçları")
            st.write(f"**📍 Toplama Adayları:** {len(aday_toplama)} il (Pareto %80)")
            isleme_isimleri = ", ".join([il['Il_Adi'] for il in isleme_aday_illeri])
            st.write(f"**⚙️ İşleme / Geri Dönüşüm:** {isleme_isimleri}")
            st.dataframe(aday_toplama[['Il_Adi', 'Toplam_Talep']].head(8), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🌟 2. Adım: Optimizasyon Modelini Çalıştır"):
                st.session_state.model_calisti = True
                st.rerun()

        with c2:
            m = folium.Map(location=[39.0, 35.0], zoom_start=6) 
            
            # 1. Aşama İkonlar ve Bilgi Kutuları (Tooltip) eklendi
            for idx, row in aday_toplama.iterrows():
                folium.Marker(location=[row['Enlem'], row['Boylam']], icon=folium.Icon(color="blue", icon="building", prefix='fa'), tooltip=f"Toplama Merkezi: {row['Il_Adi']}").add_to(m)
            for il in isleme_aday_illeri:
                folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="red", icon="industry", prefix='fa'), tooltip=f"İşleme Merkezi: {il['Il_Adi']}").add_to(m)
            for il in minimax_adaylari:
                folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="green", icon="recycle", prefix='fa'), tooltip=f"2. Ömür / Geri Dönüşüm: {il['Il_Adi']}").add_to(m)
            
            st_folium(m, width=900, height=500)
            
    except Exception as e: st.error(f"Hata: {e}")

# =====================================================
# ADIM 2: OPTİMİZASYON SONUÇLARI VE GÖRSELLEŞTİRME
# =====================================================
if st.session_state.model_calisti:
    try:
        df_koordinat = pd.read_excel(dosya_girdi, sheet_name='Koordinatlar')
        df_DE = pd.read_excel(dosya_girdi, sheet_name='DE')
        df_koordinat['Toplam_Talep'] = [np.sum(df_DE.iloc[:, -5:].values[2*i : 2*i+2]) for i in range(81)]
        
        # --- PARETO ANALİZİ DÜZELTİLDİ ---
        df_pareto = df_koordinat.sort_values(by='Toplam_Talep', ascending=False).copy()
        df_pareto['Kümülatif_%'] = df_pareto['Toplam_Talep'].cumsum() / df_pareto['Toplam_Talep'].sum()
        df_pareto['Fark'] = abs(df_pareto['Kümülatif_%'] - 0.80)
        optimum_index = df_pareto['Fark'].idxmin()
        aday_toplama = df_pareto.iloc[:optimum_index + 1].copy()
        
        merkezler_koordinat = KMeans(n_clusters=3, random_state=42, n_init=10).fit(aday_toplama[['Enlem', 'Boylam']].values).cluster_centers_
        isleme_aday_illeri = [{'Il_Adi': df_koordinat.iloc[distance.cdist([m], df_koordinat[['Enlem', 'Boylam']].values, 'euclidean').argmin()]['Il_Adi'], 
                               'Enlem': df_koordinat.iloc[distance.cdist([m], df_koordinat[['Enlem', 'Boylam']].values, 'euclidean').argmin()]['Enlem'], 
                               'Boylam': df_koordinat.iloc[distance.cdist([m], df_koordinat[['Enlem', 'Boylam']].values, 'euclidean').argmin()]['Boylam']} 
                              for m in merkezler_koordinat]
        minimax_adaylari = [{'Il_Adi': il['Il_Adi'], 'Enlem': il['Enlem'] + 0.15, 'Boylam': il['Boylam'] + 0.15} for il in isleme_aday_illeri]

        df_w = pd.read_excel(dosya_sonuc, sheet_name='W_Tesis_Akislari')
        df_z = pd.read_excel(dosya_sonuc, sheet_name='Z_Sehir_Toplama')
        df_stok = pd.read_excel(dosya_sonuc, sheet_name='Stoklar_I1_I2_I3_I4')
        df_tesis = pd.read_excel(dosya_sonuc, sheet_name='Tesis_Acilis_Kararlari')
        
        w1_donem = df_w[df_w['Akis_Tipi'].str.startswith('W1')].groupby('Donem')['Miktar'].sum()
        w23_donem = df_w[df_w['Akis_Tipi'].str.startswith('W2') | df_w['Akis_Tipi'].str.startswith('W3')].groupby('Donem')['Miktar'].sum()

        st.success("✅ Optimizasyon tamamlandı. Ağ akışları ve performans metrikleri aşağıdadır.")

        st.markdown("<h3 style='text-align: center;'>🗺️ Lojistik Akış Ağı</h3>", unsafe_allow_html=True)
        
        m_sonuc = folium.Map(location=[39.0, 35.0], zoom_start=6) 
        
        # 2. Aşama İkonlar ve Bilgi Kutuları (Tooltip) Düzeltildi
        for idx, row in aday_toplama.iterrows():
            folium.Marker(location=[row['Enlem'], row['Boylam']], icon=folium.Icon(color="blue", icon="building", prefix='fa'), tooltip=f"Toplama Merkezi: {row['Il_Adi']}").add_to(m_sonuc)
        for il in isleme_aday_illeri:
            folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="red", icon="industry", prefix='fa'), tooltip=f"İşleme Merkezi: {il['Il_Adi']}").add_to(m_sonuc)
        for il in minimax_adaylari:
            folium.Marker(location=[il['Enlem'], il['Boylam']], icon=folium.Icon(color="green", icon="recycle", prefix='fa'), tooltip=f"2. Ömür / Geri Dönüşüm: {il['Il_Adi']}").add_to(m_sonuc)

        # Oklar ve Çizgiler
        for index, toplama in aday_toplama.iterrows():
            en_kisa = float('inf'); hedef_isleme = None
            for isleme in isleme_aday_illeri:
                mesafe = distance.euclidean([toplama['Enlem'], toplama['Boylam']], [isleme['Enlem'], isleme['Boylam']])
                if mesafe < en_kisa: en_kisa = mesafe; hedef_isleme = isleme
            if hedef_isleme:
                html = f"<div style='width:150px'><b>{toplama['Il_Adi']} ➔ {hedef_isleme['Il_Adi']}</b><hr style='margin:2px;'>"
                for d, m_val in w1_donem.items(): html += f"Yıl {d}: {int(m_val/len(aday_toplama)):,} br<br>"
                folium.PolyLine(locations=[[toplama['Enlem'], toplama['Boylam']], [hedef_isleme['Enlem'], hedef_isleme['Boylam']]], color='#00b4d8', weight=2, dash_array='4', tooltip=folium.Tooltip(html+"</div>")).add_to(m_sonuc)

        for i, isleme_il in enumerate(isleme_aday_illeri):
            hedef_il = minimax_adaylari[i]
            html = f"<div style='width:150px'><b>{isleme_il['Il_Adi']} ➔ Entegre Tesis</b><hr style='margin:2px;'>"
            for d, m_val in w23_donem.items(): html += f"Yıl {d}: {int(m_val/len(isleme_aday_illeri)):,} br<br>"
            folium.PolyLine(locations=[[isleme_il['Enlem'], isleme_il['Boylam']], [hedef_il['Enlem'], hedef_il['Boylam']]], color='#ef233c', weight=3, tooltip=folium.Tooltip(html+"</div>")).add_to(m_sonuc)

        st_folium(m_sonuc, width=1500, height=550) 
        
        # --- PERFORMANS DASHBOARD ---
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>📈 Performans Kokpiti</h3>", unsafe_allow_html=True)
        
        grafik_tema = px.colors.qualitative.Pastel
        
        c1, c2 = st.columns(2)
        with c1:
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                z_trend = df_z.groupby('Donem')['Miktar'].sum().reset_index()
                fig1 = px.line(z_trend, x='Donem', y='Miktar', markers=True, title="Yıllık Toplanan Batarya (Z)", color_discrete_sequence=['#fca311'])
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        with c2:
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                tesis = df_tesis.groupby(['Donem', 'Tesis_Tipi']).size().reset_index(name='Adet')
                fig2 = px.bar(tesis, x='Donem', y='Adet', color='Tesis_Tipi', title="Aktif Tesis Yatırımları", barmode='group', color_discrete_sequence=grafik_tema)
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                stok = df_stok[df_stok['Stok_Tipi'].str.contains('I3')].groupby('Donem')['Miktar'].sum().reset_index()
                fig3 = px.bar(stok, x='Donem', y='Miktar', title="Hücre Stok Darboğazı (I3)", color='Miktar', color_continuous_scale='OrRd')
                fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        with c4:
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                w_top = df_w.groupby('Akis_Tipi')['Miktar'].sum().reset_index()
                fig4 = px.pie(w_top, names='Akis_Tipi', values='Miktar', title="Lojistik Ağ Yük Dağılımı", hole=0.5, color_discrete_sequence=grafik_tema)
                fig4.update_traces(textinfo='percent+label', textfont_color='white')
                fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e: st.error(f"Veri Okuma Hatası: {e}")
