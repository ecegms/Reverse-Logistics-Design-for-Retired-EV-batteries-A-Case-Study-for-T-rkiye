# Elektrikli Araç (EV) Bataryaları Tersine Lojistik Karar Destek Sistemi (KDS)

Bu belge, proje mimarisini, veri standartlarını ve yapay zeka ajanlarının kod geliştirme sırasında uyması gereken kesin kuralları tanımlar. Her görevden önce bu dosya referans alınmalıdır.

## 1. Proje Özeti ve Mimari
Türkiye geneli elektrikli araç bataryalarının tersine lojistik ağ tasarımı ve tesis yeri seçimi için geliştirilen, **Streamlit** tabanlı iki modüllü Karar Destek Sistemi.
- **Giriş Sistemi:** Kullanıcı doğrulama (login) ekranı bulunur.
- **Modül 1 (Talep ve Kümeleme Analizi):** Excel veri yükleme, %80 kümülatif talep hedefli Pareto analizi, Elbow (Dirsek) yöntemi ile K-Means kümeleme (Referans: Febrian & Mutasowifin, 2025), WCSS tablosu ve Folium harita görselleştirmesi.
- **Modül 2 (Optimizasyon ve Ağ Tasarımı):** Gurobi optimizasyon modeli çıktılarının analizi, finansal özet, yıllık kümülatif toplama grafikleri ve tesisler arası akış haritası.

## 2. Teknoloji Yığını (Tech Stack)
- **Dil:** Python 3.x
- **Arayüz:** Streamlit (`streamlit`, `streamlit-folium`)
- **Veri İşleme:** Pandas, NumPy, OpenPyXL
- **Analiz & Optimizasyon:** Scikit-learn (K-Means), Gurobi (`gurobipy`)
- **Görselleştirme:** Folium (Harita), Matplotlib / Plotly (Grafikler)

## 3. Veri Şeması ve Dosya Standartları
- **Ana Veri Dosyaları:** `seminerveriler_son.xlsx` ve `Model_Sonuclari.xlsx`
- **Zorunlu Sütun İsimleri (Modül 1):** `İl`, `EV_Sayısı` (Sütun isimleri değiştirilmemeli, büyük/küçük harf duyarlılığı korunmalıdır).
- **Birim Standartları:** Tüm ağırlık hesaplamaları ve tablo/grafik çıktıları **kg** cinsinden standartlaştırılmalıdır.
- **Tablo Formatı:** Veri tablolarında şehir isimleri düzgün formatlanmış (baş harfleri büyük, Türkçe karakterler bozulmadan) şekilde gösterilmelidir.
- **Hata Toleransı:** Excel dosyalarında beklenen bir sayfa (ör. `Finansal_Ozet`) veya sütun eksikse uygulama çökmemeli; `try-except` blokları ile yakalanıp kullanıcıya anlaşılır bir `st.warning` veya `st.error` mesajı gösterilmelidir.

## 4. Arayüz (UI) ve Harita Kuralları
- **Renk Paleti:** Kurumsal mavi tonları (Corporate Blue) hakim olmalıdır.
- **Harita Katmanları (Folium):**
  - Tesis türleri (Toplama, İşleme, Geri Dönüşüm, İkinci Ömür Merkezleri) birbirinden net ayrılan işaretçilerle (marker) gösterilmelidir.
  - **Akış Çizgileri:**
    - Toplama → İşleme merkezi arası: **Düz kırmızı çizgi (solid red)**
    - İşleme → İkinci Ömür / Geri Dönüşüm merkezi arası: **Kesikli siyah çizgi (dashed black)**

## 5. Ajanlar İçin Kırmızı Çizgiler (Kesin Kurallar)
1. **Önce Planla, Sonra Kodla:** Herhangi bir dosyada değişiklik yapmadan önce hangi dosyalarda ne yapılacağını maddeler halinde sun ve onay bekle.
2. **Çalışan Kodu Bozma:** Mevcut çalışan fonksiyonları, çalışan modülleri veya oturum (session_state) mantığını izinsiz silme ya da yeniden yazma.
3. **Kütüphane Kontrolü:** `requirements.txt` dışında yeni bir kütüphane eklemeden önce mutlaka onay al.
4. **Sütun ve Dosya Yolları:** Göreli dosya yolları (relative path) kullan; bilgisayarın yerel `C:\Users\...` gibi sabit yollarını koda gömme.
5. **Türkçe Arayüz:** Uygulama içi tüm başlıklar, butonlar, grafik eksenleri ve hata mesajları dilbilgisine uygun **Türkçe** olmalıdır.
