"""
Veri temizleme, Türkçe il normalizasyonu ve dinamik konum seçimi algoritmaları:
- Dinamik Pareto Analizi (en yakın kümülatif yüzde / argmin |% - eşik|)
- Dinamik K-Means Kümeleme ve Dirsek (Elbow / Kneedle) Yöntemi (Febrian & Mutasowifin, 2025)
- Dinamik Minimax / p-Center Optimizasyonu (Hakimi, 1964; 1965)

Bu modülde hiçbir sabitlenmiş (hardcoded) il sonucu, il sayısı veya tesis seçimi bulunmaz.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans


# ==========================================
# TÜRKİYE 81 İL COĞRAFİ REFERANS KOORDİNATLARI
# ==========================================
IL_KOORDINATLARI: Dict[str, Tuple[float, float]] = {
    "Adana": (37.0000, 35.3213),
    "Adıyaman": (37.7648, 38.2786),
    "Afyonkarahisar": (38.7507, 30.5567),
    "Ağrı": (39.7191, 43.0503),
    "Aksaray": (38.3687, 34.0370),
    "Amasya": (40.6499, 35.8353),
    "Ankara": (39.9334, 32.8597),
    "Antalya": (36.8969, 30.7133),
    "Ardahan": (41.1105, 42.7022),
    "Artvin": (41.1828, 41.8183),
    "Aydın": (37.8380, 27.8456),
    "Balıkesir": (39.6484, 27.8826),
    "Bartın": (41.6344, 32.3375),
    "Batman": (37.8812, 41.1351),
    "Bayburt": (40.2552, 40.2249),
    "Bilecik": (40.1451, 29.9799),
    "Bingöl": (38.8847, 40.4939),
    "Bitlis": (38.4006, 42.1095),
    "Bolu": (40.7392, 31.6116),
    "Burdur": (37.7183, 30.2823),
    "Bursa": (40.1828, 29.0667),
    "Çanakkale": (40.1553, 26.4142),
    "Çankırı": (40.6013, 33.6134),
    "Çorum": (40.5506, 34.9556),
    "Denizli": (37.7765, 29.0864),
    "Diyarbakır": (37.9144, 40.2306),
    "Düzce": (40.8438, 31.1565),
    "Edirne": (41.6744, 26.5583),
    "Elazığ": (38.6810, 39.2264),
    "Erzincan": (39.7500, 39.5000),
    "Erzurum": (39.9000, 41.2700),
    "Eskişehir": (39.7767, 30.5206),
    "Gaziantep": (37.0662, 37.3833),
    "Giresun": (40.9128, 38.3895),
    "Gümüşhane": (40.4606, 39.4814),
    "Hakkari": (37.5744, 43.7408),
    "Hatay": (36.2000, 36.1667),
    "Iğdır": (39.9237, 44.0450),
    "Isparta": (37.7648, 30.5566),
    "İstanbul": (41.0082, 28.9784),
    "İzmir": (38.4237, 27.1428),
    "Kahramanmaraş": (37.5858, 36.9371),
    "Karabük": (41.2061, 32.6228),
    "Karaman": (37.1811, 33.2222),
    "Kars": (40.6013, 43.0975),
    "Kastamonu": (41.3766, 33.7765),
    "Kayseri": (38.7312, 35.4787),
    "Kırıkkale": (39.8453, 33.5063),
    "Kırklareli": (41.7333, 27.2167),
    "Kırşehir": (39.1425, 34.1709),
    "Kilis": (36.7184, 37.1147),
    "Kocaeli": (40.7654, 29.9408),
    "Konya": (37.8667, 32.4833),
    "Kütahya": (39.4167, 29.9833),
    "Malatya": (38.3552, 38.3095),
    "Manisa": (38.6191, 27.4289),
    "Mardin": (37.3131, 40.7436),
    "Mersin": (36.8000, 34.6333),
    "Muğla": (37.2153, 28.3636),
    "Muş": (38.7304, 41.4990),
    "Nevşehir": (38.6244, 34.7144),
    "Niğde": (37.9667, 34.6833),
    "Ordu": (40.9839, 37.8764),
    "Osmaniye": (37.0742, 36.2475),
    "Rize": (41.0201, 40.5234),
    "Sakarya": (40.7569, 30.3783),
    "Samsun": (41.2867, 36.3300),
    "Siirt": (37.9333, 41.9500),
    "Sinop": (42.0231, 35.1531),
    "Sivas": (39.7477, 37.0179),
    "Şanlıurfa": (37.1591, 38.7969),
    "Şırnak": (37.5228, 42.4594),
    "Tekirdağ": (40.9780, 27.5085),
    "Tokat": (40.3167, 36.5500),
    "Trabzon": (41.0015, 39.7178),
    "Tunceli": (39.1079, 39.5401),
    "Uşak": (38.6823, 29.4082),
    "Van": (38.4891, 43.3889),
    "Yalova": (40.6500, 29.2667),
    "Yozgat": (39.8181, 34.8147),
    "Zonguldak": (41.4564, 31.7987),
}

_TR_LOWER_MAP = str.maketrans({
    "I": "ı",
    "İ": "i",
    "Â": "a",
    "â": "a",
    "Î": "i",
    "î": "i",
    "Û": "u",
    "û": "u",
})

_TR_UPPER_FIRST_MAP = {
    "i": "İ",
    "ı": "I",
    "ç": "Ç",
    "ğ": "Ğ",
    "ö": "Ö",
    "ş": "Ş",
    "ü": "Ü",
}

_ASCII_FOLD_MAP = str.maketrans({
    "ç": "c",
    "ğ": "g",
    "ı": "i",
    "ö": "o",
    "ş": "s",
    "ü": "u",
    "â": "a",
    "î": "i",
    "û": "u",
})


def _temiz_anahtar(metin: str) -> str:
    """Türkçe büyük/küçük harf ve Unicode birleşik karakterleri güvenli şekilde sadeleştirir."""
    nfc = unicodedata.normalize("NFC", str(metin).strip())
    # Python .title() vb. kaynaklı birleşik nokta karakterini temizle
    nfc = nfc.replace("\u0307", "")
    # Önce Türkçe özel büyük harfleri dönüştür, sonra lower() uygula
    kucuk = nfc.translate(_TR_LOWER_MAP).lower()
    # Fazla boşlukları tek boşluğa indir
    return " ".join(kucuk.split())


def _ascii_anahtar(metin: str) -> str:
    return _temiz_anahtar(metin).translate(_ASCII_FOLD_MAP)


def _il_esleme_tablosu_olustur() -> Tuple[Dict[str, str], Dict[str, str]]:
    tr_tablo: Dict[str, str] = {}
    ascii_tablo: Dict[str, str] = {}
    for il in IL_KOORDINATLARI:
        tr_tablo[_temiz_anahtar(il)] = il
        ascii_tablo[_ascii_anahtar(il)] = il

    # Yaygın alternatif yazımlar
    alternatifler = {
        "hakkâri": "Hakkari",
        "hakkari": "Hakkari",
        "afyon": "Afyonkarahisar",
        "içel": "Mersin",
        "icel": "Mersin",
        "maraş": "Kahramanmaraş",
        "k.maraş": "Kahramanmaraş",
        "k.maras": "Kahramanmaraş",
        "urfa": "Şanlıurfa",
        "antep": "Gaziantep",
    }
    for takma, gercek in alternatifler.items():
        tr_tablo[_temiz_anahtar(takma)] = gercek
        ascii_tablo[_ascii_anahtar(takma)] = gercek

    return tr_tablo, ascii_tablo


_TR_IL_ESLEME, _ASCII_IL_ESLEME = _il_esleme_tablosu_olustur()


def turkce_baslik_formatla(metin: str) -> str:
    """Türkçe İ/i ve I/ı kurallarını bozmadan baş harfleri büyük formata çevirir."""
    kelimeler = _temiz_anahtar(metin).split(" ")
    sonuc = []
    for kelime in kelimeler:
        if not kelime:
            continue
        ilk = _TR_UPPER_FIRST_MAP.get(kelime[0], kelime[0].upper())
        sonuc.append(ilk + kelime[1:])
    return " ".join(sonuc)


def il_adini_normalize_et(il_adi: Any) -> str:
    """
    Verilen il adını Türkçe karakterler bozulmadan kanonik 81 il adına eşler.
    Eğer bilinen 81 il dışındaysa Türkçe başlık formatıyla döndürür.
    """
    if il_adi is None or (isinstance(il_adi, float) and math.isnan(il_adi)):
        return ""
    tr_key = _temiz_anahtar(str(il_adi))
    if not tr_key:
        return ""
    if tr_key in _TR_IL_ESLEME:
        return _TR_IL_ESLEME[tr_key]
    ascii_key = _ascii_anahtar(str(il_adi))
    if ascii_key in _ASCII_IL_ESLEME:
        return _ASCII_IL_ESLEME[ascii_key]
    return turkce_baslik_formatla(str(il_adi))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki coğrafi koordinat arasındaki büyük daire (Haversine) mesafesini km cinsinden hesaplar."""
    r_dunya = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2.0 * r_dunya * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ==========================================
# VERİ OKUMA VE TEMİZLEME
# ==========================================
def veriyi_oku_ve_temizle(
    df: pd.DataFrame,
    koordinat_sozlugu: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Modül 1 giriş verisini doğrular ve temizler.
    - Zorunlu sütunlar: 'İl' ve 'EV_Sayısı' (büyük/küçük harf duyarlı).
    - Türkçe il adlarını normalize eder, sayısal olmayan/negatif satırları ayıklar,
      mükerrer illeri toplar ve koordinatı olmayan illeri uyarı olarak raporlar.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Geçerli bir veri tablosu sağlanmalıdır.")

    eksik_sutunlar = [s for s in ("İl", "EV_Sayısı") if s not in df.columns]
    if eksik_sutunlar:
        raise ValueError(
            f"Veri dosyasında zorunlu sütunlar eksik: {', '.join(eksik_sutunlar)}. "
            "Sütun isimleri tam olarak 'İl' ve 'EV_Sayısı' olmalıdır."
        )

    koord_kaynak = koordinat_sozlugu if koordinat_sozlugu is not None else IL_KOORDINATLARI
    uyarilar: List[str] = []

    calisma = df.copy()
    calisma["İl"] = calisma["İl"].apply(il_adini_normalize_et)
    calisma["EV_Sayısı"] = pd.to_numeric(calisma["EV_Sayısı"], errors="coerce")

    bos_il_sayisi = int((calisma["İl"] == "").sum())
    if bos_il_sayisi > 0:
        uyarilar.append(f"İl adı boş olan {bos_il_sayisi} satır analiz dışı bırakıldı.")
        calisma = calisma[calisma["İl"] != ""]

    gecersiz_ev_mask = calisma["EV_Sayısı"].isna() | (calisma["EV_Sayısı"] < 0)
    gecersiz_ev_sayisi = int(gecersiz_ev_mask.sum())
    if gecersiz_ev_sayisi > 0:
        uyarilar.append(
            f"Geçersiz veya negatif 'EV_Sayısı' içeren {gecersiz_ev_sayisi} satır temizlendi."
        )
        calisma = calisma[~gecersiz_ev_mask]

    if calisma.empty:
        raise ValueError("Temizleme sonrasında analiz edilebilecek geçerli kayıt kalmadı.")

    # İsteğe bağlı özel koordinat sütunları varsa koru
    ozel_koord_var = "Enlem" in calisma.columns and "Boylam" in calisma.columns
    if ozel_koord_var:
        calisma["Enlem"] = pd.to_numeric(calisma["Enlem"], errors="coerce")
        calisma["Boylam"] = pd.to_numeric(calisma["Boylam"], errors="coerce")

    # Mükerrer illeri birleştir
    if calisma["İl"].duplicated().any():
        if ozel_koord_var:
            calisma = calisma.groupby("İl", as_index=False).agg(
                {"EV_Sayısı": "sum", "Enlem": "first", "Boylam": "first"}
            )
        else:
            calisma = calisma.groupby("İl", as_index=False)["EV_Sayısı"].sum()
        uyarilar.append("Aynı il adına sahip tekrarlı satırlar toplanarak birleştirildi.")

    # Koordinat eşleştirmesi (önce tablodaki Enlem/Boylam, yoksa 81 il sözlüğü)
    enlemler: List[Optional[float]] = []
    boylamlar: List[Optional[float]] = []
    eslesmeyenler: List[str] = []

    for _, row in calisma.iterrows():
        il = row["İl"]
        lat_val: Optional[float] = None
        lon_val: Optional[float] = None
        if ozel_koord_var and pd.notna(row["Enlem"]) and pd.notna(row["Boylam"]):
            lat_val = float(row["Enlem"])
            lon_val = float(row["Boylam"])
        elif il in koord_kaynak:
            lat_val, lon_val = float(koord_kaynak[il][0]), float(koord_kaynak[il][1])
        else:
            eslesmeyenler.append(il)
        enlemler.append(lat_val)
        boylamlar.append(lon_val)

    calisma["Enlem"] = enlemler
    calisma["Boylam"] = boylamlar

    if eslesmeyenler:
        uyarilar.append(
            f"Koordinatı tanımlı olmayan iller çıkarıldı: {', '.join(sorted(set(eslesmeyenler)))}"
        )
        calisma = calisma.dropna(subset=["Enlem", "Boylam"]).reset_index(drop=True)

    if calisma.empty:
        raise ValueError("Koordinatı eşleşen geçerli bir il bulunamadı.")

    if float(calisma["EV_Sayısı"].sum()) <= 0.0:
        raise ValueError("Toplam 'EV_Sayısı' sıfırdan büyük olmalıdır.")

    calisma = calisma[["İl", "EV_Sayısı", "Enlem", "Boylam"]].reset_index(drop=True)
    return calisma, uyarilar


# ==========================================
# 1. DİNAMİK PARETO ANALİZİ
# ==========================================
def dinamik_pareto_analizi(
    df: pd.DataFrame,
    esik: float = 80.0,
    koordinat_sozlugu: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Tüm illeri EV_Sayısı'na göre azalan sıralar, kümülatif yüzdeyi hesaplar ve
    argmin(|Kümülatif_Yüzde - eşik|) kuralı ile en yakın kesim noktasını dinamik seçer.

    Güvenlik kuralları:
    - Her durumda en az 1 il seçilir.
    - İki ardışık il eşiğe tam eşit uzaklıktaysa daha az il içeren (kompakt) kesim seçilir.
    """
    if not (0.0 < float(esik) <= 100.0):
        raise ValueError("Pareto eşik değeri 0 ile 100 arasında olmalıdır.")

    temiz_df, uyarilar = veriyi_oku_ve_temizle(df, koordinat_sozlugu=koordinat_sozlugu)

    df_sorted = temiz_df.sort_values(
        by=["EV_Sayısı", "İl"], ascending=[False, True]
    ).reset_index(drop=True)

    toplam_ev = float(df_sorted["EV_Sayısı"].sum())
    if toplam_ev <= 0.0:
        raise ValueError("Pareto analizi için toplam EV sayısı sıfırdan büyük olmalıdır.")

    df_sorted["Kümülatif_Toplam"] = df_sorted["EV_Sayısı"].cumsum()
    df_sorted["Kümülatif_Yüzde"] = (df_sorted["Kümülatif_Toplam"] / toplam_ev) * 100.0

    farklar = np.abs(df_sorted["Kümülatif_Yüzde"].to_numpy(dtype=float) - float(esik))
    min_fark = float(np.min(farklar))

    # Eşit uzaklık durumunda (kayan nokta toleransı ile) en az il içeren ilk indeks seçilir
    esit_indeksler = np.where(np.isclose(farklar, min_fark, rtol=0.0, atol=1e-9))[0]
    en_iyi_idx = max(0, int(esit_indeksler[0]))

    pareto_df = df_sorted.iloc[: en_iyi_idx + 1].copy().reset_index(drop=True)

    return {
        "pareto_df": pareto_df,
        "tum_df": df_sorted,
        "secilen_il_sayisi": int(len(pareto_df)),
        "toplam_il_sayisi": int(len(df_sorted)),
        "esik": float(esik),
        "ulasilan_yuzde": float(pareto_df["Kümülatif_Yüzde"].iloc[-1]),
        "esige_mutlak_fark": float(farklar[en_iyi_idx]),
        "uyarilar": uyarilar,
    }


# ==========================================
# 2. DİNAMİK K-MEANS VE DİRSEK (ELBOW) YÖNTEMİ
# ==========================================
def _kneedle_optimum_k_bul(k_degerleri: Sequence[int], wcss: Sequence[float]) -> int:
    """
    (k, SSE) eğrisinde normalize edilmiş maksimum dik uzaklık (Kneedle) mantığıyla
    kırılma noktasını (optimum k) matematiksel olarak belirler.
    """
    n_nokta = len(k_degerleri)
    if n_nokta == 0:
        raise ValueError("Dirsek analizi için en az bir k değeri gereklidir.")
    if n_nokta == 1:
        return int(k_degerleri[0])
    if n_nokta == 2:
        # Ara nokta yok: SSE düşüyorsa k=2, aksi halde k=1
        return int(k_degerleri[1]) if wcss[1] < wcss[0] - 1e-12 else int(k_degerleri[0])

    ks = np.asarray(k_degerleri, dtype=float)
    ys = np.asarray(wcss, dtype=float)

    k_aralik = float(ks[-1] - ks[0])
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    y_aralik = y_max - y_min

    if k_aralik <= 1e-12 or y_aralik <= 1e-12:
        return int(k_degerleri[0])

    # [0, 1] eksen normalizasyonu
    x_norm = (ks - ks[0]) / k_aralik
    y_norm = (ys - y_min) / y_aralik

    # İlk ve son noktayı birleştiren doğruya olan işaretli dik uzaklık (eğri doğrunun altındayken pozitif)
    egim = y_norm[-1] - y_norm[0]
    payda = math.sqrt(1.0 + egim * egim)
    y_dogru = y_norm[0] + egim * x_norm
    dik_uzakliklar = (y_dogru - y_norm) / payda

    # Uç noktaların uzaklığı 0'dır; pozitif kırılma varsa maksimumunu al
    if float(np.max(dik_uzakliklar)) > 1e-12:
        en_iyi_idx = int(np.argmax(dik_uzakliklar))
    else:
        # Dışbükey kırılma yoksa ara noktalar içinde mutlak maksimumu seç
        ara_uzakliklar = np.abs(dik_uzakliklar[1:-1])
        en_iyi_idx = int(np.argmax(ara_uzakliklar)) + 1

    return int(k_degerleri[en_iyi_idx])


def dinamik_kmeans_ve_dirsek(
    pareto_df: pd.DataFrame,
    random_state: int = 42,
    secili_k: Optional[int] = None,
    koordinat_sozlugu: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Febrian & Mutasowifin (2025) yaklaşımına uygun olarak:
    - Yalnızca Pareto'dan gelen illerin (Enlem, Boylam) koordinatlarını kullanır
      (ağırlıksız, standartlaştırmasız, Öklid mesafesi).
    - Maksimum küme sayısını dinamik olarak k_max = min(9, N - 1) ile sınırlar
      (N < 10 ve N = 1, 2, 3 dahil tüm durumlarda çökmeden çalışır).
    - Her k için SSE (WCSS) hesaplar, random_state sabitler ve Kneedle (maksimum
      dik uzaklık) yöntemiyle optimum k değerini otomatik önerir.
    - Her kümenin merkezine (centroid) kendi küme üyeleri arasından Öklid mesafesi
      en yakın ili seçerek çakışmasız tam k adet Aday İşleme Merkezi döndürür.
    """
    if pareto_df is None or pareto_df.empty:
        raise ValueError("K-Means kümeleme için Pareto çıktısı boş olamaz.")

    koord_kaynak = koordinat_sozlugu if koordinat_sozlugu is not None else IL_KOORDINATLARI
    sehirler: List[str] = []
    coords: List[Tuple[float, float]] = []

    ozel_koord_var = "Enlem" in pareto_df.columns and "Boylam" in pareto_df.columns

    for _, row in pareto_df.iterrows():
        il_adi = il_adini_normalize_et(row["İl"])
        if not il_adi:
            continue
        if ozel_koord_var and pd.notna(row["Enlem"]) and pd.notna(row["Boylam"]):
            lat, lon = float(row["Enlem"]), float(row["Boylam"])
        elif il_adi in koord_kaynak:
            lat, lon = float(koord_kaynak[il_adi][0]), float(koord_kaynak[il_adi][1])
        else:
            continue
        if il_adi not in sehirler:
            sehirler.append(il_adi)
            coords.append((lat, lon))

    n_il = len(sehirler)
    if n_il == 0:
        raise ValueError("K-Means kümeleme için koordinatı geçerli il bulunamadı.")

    X = np.asarray(coords, dtype=float)
    benzersiz_nokta = int(len(np.unique(X, axis=0)))

    # Sınır durumu: tek il veya tüm koordinatlar özdeş
    if n_il == 1 or benzersiz_nokta <= 1:
        k_degerleri = [1]
        wcss = [0.0]
        optimum_k = 1
        k_kullanilan = 1
        etiketler = np.zeros(n_il, dtype=int)
        centroidler = np.array([X.mean(axis=0)], dtype=float)
    else:
        # İstenen kural: k_max = min(9, N - 1) (ve benzersiz nokta sayısını aşmama güvenliği)
        k_max = max(1, min(9, n_il - 1, benzersiz_nokta))
        k_degerleri = list(range(1, k_max + 1))
        wcss = []

        for k in k_degerleri:
            model_k = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=random_state,
            )
            model_k.fit(X)
            wcss.append(float(model_k.inertia_))

        optimum_k = _kneedle_optimum_k_bul(k_degerleri, wcss)

        if secili_k is not None:
            k_kullanilan = max(1, min(int(secili_k), k_max))
        else:
            k_kullanilan = optimum_k

        kmeans_final = KMeans(
            n_clusters=k_kullanilan,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        etiketler = kmeans_final.fit_predict(X)
        centroidler = kmeans_final.cluster_centers_

    # Her küme için kendi üyeleri arasından centroid'e Öklid mesafesi en yakın ili seç
    ham_merkezler: List[Dict[str, Any]] = []
    for kume_id in range(k_kullanilan):
        uye_indeksler = np.where(etiketler == kume_id)[0]
        if len(uye_indeksler) == 0:
            continue
        centroid = centroidler[kume_id]

        en_yakin_idx = None
        en_kucuk_dist = float("inf")
        for idx in uye_indeksler:
            lat, lon = X[idx]
            dist = math.sqrt((centroid[0] - lat) ** 2 + (centroid[1] - lon) ** 2)
            # Eşit mesafe halinde deterministik alfabetik seçim
            if dist < en_kucuk_dist - 1e-12 or (
                abs(dist - en_kucuk_dist) <= 1e-12
                and (en_yakin_idx is None or sehirler[idx] < sehirler[en_yakin_idx])
            ):
                en_kucuk_dist = dist
                en_yakin_idx = int(idx)

        secilen_il = sehirler[en_yakin_idx]
        secilen_lat, secilen_lon = float(X[en_yakin_idx][0]), float(X[en_yakin_idx][1])
        uye_iller = sorted(sehirler[int(i)] for i in uye_indeksler)

        ham_merkezler.append(
            {
                "İl": secilen_il,
                "Lat": secilen_lat,
                "Lon": secilen_lon,
                "Centroid_Lat": float(centroid[0]),
                "Centroid_Lon": float(centroid[1]),
                "Atanan_İller": uye_iller,
                "Küme_Boyutu": len(uye_iller),
            }
        )

    # Deterministik sıralama ve kodlama
    ham_merkezler.sort(key=lambda d: (-d["Küme_Boyutu"], d["İl"]))
    aday_isleme_merkezleri: List[Dict[str, Any]] = []
    for sira, merkez in enumerate(ham_merkezler, start=1):
        merkez["Kod"] = f"İM-{sira}"
        aday_isleme_merkezleri.append(merkez)

    return {
        "n_il": n_il,
        "k_max": int(k_degerleri[-1]),
        "k_degerleri": k_degerleri,
        "wcss": wcss,
        "optimum_k": int(optimum_k),
        "kullanilan_k": int(k_kullanilan),
        "aday_isleme_merkezleri": aday_isleme_merkezleri,
    }


# ==========================================
# 3. DİNAMİK MİNİMAX / p-CENTER MODELİ (HAKIMI, 1964; 1965)
# ==========================================
def dinamik_minimax_p_center(
    isleme_merkezleri: Sequence[Dict[str, Any]],
    p: int = 2,
) -> Dict[str, Any]:
    """
    K-Means'ten dinamik olarak üretilen k adet İşleme Merkezi üzerinde
    Hakimi (1964; 1965) kesin p-Center (Minimax) optimizasyonunu çözer.

    Amaç: Her işleme merkezinin kendisine en yakın seçili p tesise olan
    maksimum mesafesini (Z) minimize etmek (min Z).
    (p-Median olan toplam mesafe minimizasyonu ile karıştırılmaz; yalnızca
    aynı optimal Z* değerini veren çözümler arasında deterministik eşitlik
    kırıcı olarak ikinci aşama uygulanır.)
    """
    if not isleme_merkezleri:
        raise ValueError("Minimax (p-Center) için en az 1 aday işleme merkezi gereklidir.")

    # Tekrarlı aday varsa güvenli şekilde tekilleştir
    adaylar: List[Dict[str, Any]] = []
    gorulen: set = set()
    for m in isleme_merkezleri:
        il = il_adini_normalize_et(m["İl"])
        if il and il not in gorulen:
            gorulen.add(il)
            adaylar.append(
                {
                    "İl": il,
                    "Lat": float(m["Lat"]),
                    "Lon": float(m["Lon"]),
                    "Kod": m.get("Kod", f"İM-{len(adaylar) + 1}"),
                }
            )

    k = len(adaylar)
    if k == 0:
        raise ValueError("Geçerli aday işleme merkezi bulunamadı.")

    p_istenen = int(p)
    if p_istenen < 1:
        raise ValueError("Seçilecek tesis sayısı (p) en az 1 olmalıdır.")
    p_eff = min(p_istenen, k)

    # Dinamik k x k ikili Haversine mesafe matrisi (km)
    iller = [a["İl"] for a in adaylar]
    dist_mat = np.zeros((k, k), dtype=float)
    for u in range(k):
        for v in range(u + 1, k):
            d_uv = haversine_km(
                adaylar[u]["Lat"], adaylar[u]["Lon"],
                adaylar[v]["Lat"], adaylar[v]["Lon"],
            )
            dist_mat[u, v] = d_uv
            dist_mat[v, u] = d_uv

    mesafe_df = pd.DataFrame(
        np.round(dist_mat, 2), index=iller, columns=iller
    )

    # Aşama 1: Kesin p-Center (min Z)
    model = pulp.LpProblem("Dinamik_Minimax_p_Center", pulp.LpMinimize)
    # Python 3.13+ PuLP uyumluluğu için 0-1 Integer tanımı
    y = {
        v: pulp.LpVariable(f"y_{v}", lowBound=0, upBound=1, cat="Integer")
        for v in range(k)
    }
    x = {
        (u, v): pulp.LpVariable(f"x_{u}_{v}", lowBound=0, upBound=1, cat="Integer")
        for u in range(k)
        for v in range(k)
    }
    Z = pulp.LpVariable("Z_max", lowBound=0, cat="Continuous")

    model += Z, "Minimax_Maksimum_Mesafe"
    model += pulp.lpSum(y[v] for v in range(k)) == p_eff, "Tesis_Sayisi_p"

    for u in range(k):
        model += pulp.lpSum(x[(u, v)] for v in range(k)) == 1, f"Atama_{u}"
        model += (
            pulp.lpSum(dist_mat[u, v] * x[(u, v)] for v in range(k)) <= Z,
            f"Minimax_Kisit_{u}",
        )
        for v in range(k):
            model += x[(u, v)] <= y[v], f"Acik_Tesis_{u}_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status1 = model.solve(solver)
    durum1 = pulp.LpStatus[status1]
    if durum1 != "Optimal":
        raise RuntimeError(f"Minimax (p-Center) modeli optimal çözülemedi: {durum1}")

    z_star = float(pulp.value(Z))

    # Aşama 2: Z* optimalitesini kesinlikle bozmadan eşitlik kırıcı (lexicographic minimax)
    model += Z <= z_star + 1e-6, "Optimal_Z_Sabitleme"
    model.setObjective(
        pulp.lpSum(dist_mat[u, v] * x[(u, v)] for u in range(k) for v in range(k))
    )
    status2 = model.solve(solver)
    if pulp.LpStatus[status2] != "Optimal":
        # Güvenlik geri dönüşü: 1. aşama optimal çözümü zaten geçerlidir
        model.setObjective(Z)
        model.solve(solver)

    secilen_indeksler = [
        v for v in range(k) if round(float(pulp.value(y[v]) or 0.0)) == 1
    ]
    if len(secilen_indeksler) != p_eff:
        raise RuntimeError("Minimax çözümünde seçilen tesis sayısı p ile uyuşmuyor.")

    # Bağımsız matematiksel doğrulama: her işleme merkezinin en yakın seçili tesise mesafesi
    atamalar: List[Dict[str, Any]] = []
    gercek_max_mesafe = 0.0
    for u in range(k):
        en_yakin_v = min(secilen_indeksler, key=lambda v: (dist_mat[u, v], iller[v]))
        mesafe_km = float(dist_mat[u, en_yakin_v])
        if mesafe_km > gercek_max_mesafe:
            gercek_max_mesafe = mesafe_km
        atamalar.append(
            {
                "İşleme Merkezi": iller[u],
                "Atanan Tesis": iller[en_yakin_v],
                "Mesafe (km)": round(mesafe_km, 2),
            }
        )

    secilen_merkezler: List[Dict[str, Any]] = []
    for sira, v in enumerate(sorted(secilen_indeksler, key=lambda idx: iller[idx]), start=1):
        secilen_merkezler.append(
            {
                "İl": adaylar[v]["İl"],
                "Lat": adaylar[v]["Lat"],
                "Lon": adaylar[v]["Lon"],
                "Kod": f"MM-{sira:02d}",
                "GDM_Kod": f"GDM-{sira:02d}",
                "IOM_Kod": f"İÖM-{sira:02d}",
            }
        )

    return {
        "durum": "Optimal",
        "aday_sayisi_k": k,
        "p_istenen": p_istenen,
        "p_kullanilan": p_eff,
        "minimax_z_km": round(gercek_max_mesafe, 4),
        "secilen_merkezler": secilen_merkezler,
        "atamalar": atamalar,
        "mesafe_matrisi": mesafe_df,
    }


# ==========================================
# UÇTAN UCA DİNAMİK ZİNCİR
# ==========================================
def tum_zinciri_calistir(
    df: pd.DataFrame,
    esik: float = 80.0,
    p: int = 2,
    random_state: int = 42,
    koordinat_sozlugu: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Verilen EV veri seti üzerinde Pareto -> K-Means (Dirsek) -> Minimax (p-Center)
    zincirini hiçbir sabit sonuç kullanmadan uçtan uca çalıştırır.
    """
    pareto_res = dinamik_pareto_analizi(
        df, esik=esik, koordinat_sozlugu=koordinat_sozlugu
    )
    kmeans_res = dinamik_kmeans_ve_dirsek(
        pareto_res["pareto_df"],
        random_state=random_state,
        koordinat_sozlugu=koordinat_sozlugu,
    )
    minimax_res = dinamik_minimax_p_center(
        kmeans_res["aday_isleme_merkezleri"],
        p=p,
    )
    return {
        "pareto": pareto_res,
        "kmeans": kmeans_res,
        "minimax": minimax_res,
    }
