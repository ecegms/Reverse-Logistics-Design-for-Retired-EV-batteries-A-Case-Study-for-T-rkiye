import folium
from streamlit_folium import st_folium

# ==========================================
# 4. MODÜL 1 İÇERİĞİ (GÜNCELLENMİŞ HARİTALI VERSİYON)
# ==========================================
def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme")
    
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    
    st.divider()
    
    # ADIM 1: VERİ YÜKLEME
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: EV Verisi Yükleme")
        st.write("İllere göre elektrikli araç sayılarını içeren Excel dosyasını yükleyin:")
        uploaded_file = st.file_uploader("", type=['xlsx', 'xls'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.session_state.mod1_data = "Veri Yüklendi" # Burada pd.read_excel(uploaded_file) yapacaksın
            st.success("Dosya başarıyla sisteme aktarıldı.")
            if st.button("Pareto Analizini Başlat", type="primary"):
                st.session_state.mod1_step = 1
                st.rerun()

    # ADIM 2: PARETO ANALİZİ VE HARİTASI
    if st.session_state.mod1_step >= 1:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi Sonuçları")
        st.info("Toplam talebin %80'ini karşılayan iller belirlendi ve haritaya Aday Toplama Merkezi olarak işlendi.")
        
        # Türkiye merkezli Folium haritası oluşturuluyor
        m_pareto = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya kendi Pareto algoritmanın çıktılarını bağlayacaksın. 
        # Temsili Pareto Çıktıları (İl Adı, Enlem, Boylam, Merkez Kodu)
        pareto_sonuclari = [
            ("İstanbul", 41.0082, 28.9784, "TM-34"),
            ("Ankara", 39.9334, 32.8597, "TM-06"),
            ("İzmir", 38.4237, 27.1428, "TM-35")
        ]
        
        for il, lat, lon, kod in pareto_sonuclari:
            # HTML destekli gelişmiş Tooltip (Fare ile üzerine gelince çıkan bilgi kutusu)
            tooltip_html = f"""
            <div style='font-family: Arial; font-size: 14px;'>
                <b>Tesis Tipi:</b> Aday Toplama Merkezi<br>
                <b>İl Adı:</b> {il}<br>
                <b>Merkez Kodu:</b> {kod}
            </div>
            """
            folium.Marker(
                [lat, lon], 
                tooltip=tooltip_html, 
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m_pareto)
            
        # Haritayı Streamlit ekrana bas
        st_folium(m_pareto, width=900, height=450, key="map_pareto")

        if st.button("K-Means Kümeleme Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 2
            st.rerun()

    # ADIM 3: K-MEANS VE HARİTASI
    if st.session_state.mod1_step >= 2:
        st.divider()
        st.subheader("Adım 3: K-Means Kümeleme Sonuçları")
        st.info("Pareto çıktıları kullanılarak Aday İşleme Merkezleri belirlendi.")
        
        m_kmeans = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya K-Means sonuçlarını bağlayacaksın.
        # Temsili K-Means Çıktıları
        kmeans_sonuclari = [
            ("Bursa", 40.1828, 29.0667, "İM-16"),
            ("Kayseri", 38.7312, 35.4787, "İM-38")
        ]
        
        for il, lat, lon, kod in kmeans_sonuclari:
            tooltip_html = f"""
            <div style='font-family: Arial; font-size: 14px;'>
                <b>Tesis Tipi:</b> Aday İşleme Merkezi<br>
                <b>İl Adı:</b> {il}<br>
                <b>Merkez Kodu:</b> {kod}
            </div>
            """
            # İşleme merkezleri için kırmızı renk kullanıyoruz
            folium.Marker(
                [lat, lon], 
                tooltip=tooltip_html, 
                icon=folium.Icon(color="red", icon="cog")
            ).add_to(m_kmeans)
            
        st_folium(m_kmeans, width=900, height=450, key="map_kmeans")

        if st.button("Minimax Algoritmasını Çalıştır", type="primary"):
            st.session_state.mod1_step = 3
            st.rerun()

    # ADIM 4: MINIMAX VE HARİTASI
    if st.session_state.mod1_step >= 3:
        st.divider()
        st.subheader("Adım 4: Minimax Algoritması Sonuçları")
        st.success("Tüm aday merkez konumları başarıyla tespit edildi! Aday Geri Dönüşüm ve İkinci Ömür Merkezleri eklendi.")
        
        m_minimax = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
        
        # TODO: Buraya Minimax sonuçlarını bağlayacaksın.
        # Temsili Minimax Çıktıları
        geri_donusum_merkezleri = [("Eskişehir", 39.7767, 30.5206, "GDM-26")]
        ikinci_omur_merkezleri = [("Konya", 37.8667, 32.4833, "İÖM-42")]
        
        for il, lat, lon, kod in geri_donusum_merkezleri:
            tooltip_html = f"<b>Tesis Tipi:</b> Aday Geri Dönüşüm Merkezi<br><b>İl:</b> {il}<br><b>Kod:</b> {kod}"
            folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="green", icon="recycle")).add_to(m_minimax)
            
        for il, lat, lon, kod in ikinci_omur_merkezleri:
            tooltip_html = f"<b>Tesis Tipi:</b> Aday İkinci Ömür Merkezi<br><b>İl:</b> {il}<br><b>Kod:</b> {kod}"
            folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="orange", icon="flash")).add_to(m_minimax)

        st_folium(m_minimax, width=900, height=450, key="map_minimax")
