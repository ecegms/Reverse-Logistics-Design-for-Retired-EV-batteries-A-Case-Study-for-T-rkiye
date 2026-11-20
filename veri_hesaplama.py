"""
Veri temizleme, Türkçe il normalizasyonu ve dinamik konum seçimi algoritmaları:
- Dinamik Pareto Analizi (en yakın kümülatif yüzde / argmin |% - eşik|)
- Dinamik K-Means Kümeleme ve Dirsek (Elbow / Kneedle) Yöntemi (Febrian & Mutasowifin, 2025)
- Dinamik Minimax / p-Center Optimizasyonu (Hakimi, 1964; 1965)

Bu modülde hiçbir sabitlenmiş (hardcoded) il sonucu, il sayısı veya tesis seçimi bulunmaz.
"""

from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    Hakimi (1964; 1965) kesin p-Center (Minimax) optimizasyonunu yalnızca
    standart Python (`itertools.combinations`) ve `numpy` kullanarak çözer.

    Amaç: k adet işleme merkezi arasındaki ikili Öklid mesafe matrisi üzerinde,
    her işleme merkezinin kendisine en yakın seçili p tesise olan maksimum
    mesafesini (Z) minimize eden kesin optimum kombinasyonu bulmak (min Z).
    (p-Median olan toplam mesafe minimizasyonu ile karıştırılmaz; yalnızca
    birebir aynı optimal Z* değerini veren kombinasyonlar arasında deterministik
    eşitlik kırıcı uygulanır.)
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

    # Dinamik k x k ikili Öklid mesafe matrisi ((Lat, Lon) koordinatları üzerinden)
    iller = [a["İl"] for a in adaylar]
    coords = np.array([[a["Lat"], a["Lon"]] for a in adaylar], dtype=float)
    fark = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(fark ** 2, axis=-1))

    mesafe_df = pd.DataFrame(
        np.round(dist_mat, 4), index=iller, columns=iller
    )

    # Kesin (exact) p-Center araması: tüm C(k, p_eff) kombinasyonları tara
    en_iyi_kombinasyon: Optional[Tuple[int, ...]] = None
    en_iyi_z = float("inf")
    en_iyi_toplam = float("inf")
    en_iyi_isim_anahtari: Optional[Tuple[str, ...]] = None

    for comb in combinations(range(k), p_eff):
        en_yakin_mesafeler = np.min(dist_mat[:, comb], axis=1)
        z_degeri = float(np.max(en_yakin_mesafeler))
        toplam_mesafe = float(np.sum(en_yakin_mesafeler))
        isim_anahtari = tuple(sorted(iller[v] for v in comb))

        if z_degeri < en_iyi_z - 1e-12:
            en_iyi_z = z_degeri
            en_iyi_toplam = toplam_mesafe
            en_iyi_isim_anahtari = isim_anahtari
            en_iyi_kombinasyon = comb
        elif abs(z_degeri - en_iyi_z) <= 1e-12:
            # Z* optimalitesini bozmadan deterministik eşitlik kırıcı
            if toplam_mesafe < en_iyi_toplam - 1e-12 or (
                abs(toplam_mesafe - en_iyi_toplam) <= 1e-12
                and (en_iyi_isim_anahtari is None or isim_anahtari < en_iyi_isim_anahtari)
            ):
                en_iyi_z = min(en_iyi_z, z_degeri)
                en_iyi_toplam = toplam_mesafe
                en_iyi_isim_anahtari = isim_anahtari
                en_iyi_kombinasyon = comb

    if en_iyi_kombinasyon is None or len(en_iyi_kombinasyon) != p_eff:
        raise RuntimeError("Minimax (p-Center) için geçerli kombinasyon bulunamadı.")

    secilen_indeksler = list(en_iyi_kombinasyon)

    # Her işleme merkezinin kendisine en yakın seçili tesise atanması ve Z doğrulaması
    atamalar: List[Dict[str, Any]] = []
    gercek_max_oklid = 0.0
    gercek_max_km = 0.0
    for u in range(k):
        en_yakin_v = min(secilen_indeksler, key=lambda v: (dist_mat[u, v], iller[v]))
        mesafe_oklid = float(dist_mat[u, en_yakin_v])
        mesafe_km = haversine_km(
            adaylar[u]["Lat"], adaylar[u]["Lon"],
            adaylar[en_yakin_v]["Lat"], adaylar[en_yakin_v]["Lon"],
        )
        if mesafe_oklid > gercek_max_oklid:
            gercek_max_oklid = mesafe_oklid
        if mesafe_km > gercek_max_km:
            gercek_max_km = mesafe_km
        atamalar.append(
            {
                "İşleme Merkezi": iller[u],
                "Atanan Tesis": iller[en_yakin_v],
                "Mesafe (Öklid)": round(mesafe_oklid, 4),
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
        "minimax_z": round(gercek_max_oklid, 4),
        "minimax_z_km": round(gercek_max_km, 4),
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


# ==========================================
# 4. TEZ UYUMLU 5 DÖNEMLİ, STOKLU VE KÂR MAKSİMİZASYONLU MILP MODELİ
# ==========================================
TEZ_DONEMLER: List[int] = [0, 1, 2, 3, 4]
TEZ_BATARYA_TIPLERI: List[str] = ["Binek", "Agir_Vasita"]
TEZ_HAMMADDELER: List[str] = [
    "Kobalt", "Lityum", "Nikel", "Bakir", "Aluminyum", "Manganez", "Grafit"
]
TEZ_PAZARLAR: List[str] = [
    "Pazar 1 (Konya-Karapınar)",
    "Pazar 2 (Bursa-Merkez)",
    "Pazar 3 (İstanbul-Tuzla)",
]
TEZ_PAZAR_KOORDINATLARI: List[Tuple[float, float]] = [
    (37.7167, 33.5500),  # Konya (Karapınar)
    (40.1828, 29.0667),  # Bursa (Merkez)
    (40.8167, 29.3000),  # İstanbul (Tuzla)
]
# Ulusal 5 dönem talep büyüme profili (DE sayfası yoksa oransal dağılım için)
_TEZ_DONEM_DAGILIM_PAYI: List[float] = [0.009941, 0.022687, 0.123080, 0.282528, 0.561764]

# Modül 2 arayüz varsayılanları (What-If oransal ölçekleme referansı)
_UI_VARSAYILAN_REFERANS: Dict[str, float] = {
    "cap_j": 100000.0,
    "cap_k": 100000.0,
    "cap_l": 15000.0,
    "cap_m": 30000.0,
    "fj": 250000.0,
    "fk": 400000.0,
    "fl": 650000.0,
    "fm": 450000.0,
    "c3_tasima": 8.0,
    "c4_tasima": 0.08,
    "c1_binek": 350.0,
    "c1_agir": 3200.0,
    "c5_bertaraf": 5.0,
    "rev_kobalt": 50.0,
    "rev_lityum": 1.5,
    "rev_nikel": 12.0,
}


def _tez_varsayilan_parametreler() -> Dict[str, Any]:
    """seminerveriler_son.xlsx ile birebir eşleşen 5 dönemli tez varsayılan parametreleri."""
    return {
        "T": list(TEZ_DONEMLER),
        "B": list(TEZ_BATARYA_TIPLERI),
        "N": list(TEZ_HAMMADDELER),
        "S": [0, 1, 2],
        "pazar_adlari": list(TEZ_PAZARLAR),
        "H": [96.0, 1008.0],
        "G": [3.5, 0.0395],
        "R1": [0.62, 0.62],
        "R2": [0.15, 0.15],
        "R3": [0.23, 0.23],
        "R21": [0.2047, 0.2047],
        "R22": [0.7019, 0.7019],
        "R23": [0.0934, 0.0934],
        "CR": [0.16, 0.15, 0.10, 0.07, 0.07, 0.05, 0.04],
        "RR": [0.945, 0.955, 0.98, 0.965, 0.98, 0.99, 0.9845],
        "Gama": [[0.4, 0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4, 0.4]],
        "f_t": [66420.0, 81032.40, 95618.23, 109960.97, 123156.28],
        "fk_t": [106272.0, 129651.84, 152989.17, 175937.55, 197050.05],
        "fl_t": [172692.0, 210684.24, 248607.40, 285898.51, 320206.34],
        "fm_t": [119556.0, 145858.32, 172112.82, 197929.74, 221681.31],
        "Cap_j_base": [175000.0, 175000.0],
        "Cap_k_base": [6000.0, 6000.0],
        "Cap_l_base": 400000.0,
        "Cap_m_base": [1000000.0, 1000000.0],
        "CJ": [
            [60.0, 73.20, 86.38, 99.34, 111.26],
            [60.0, 73.20, 86.38, 99.34, 111.26],
        ],
        "CK": [
            [172.0, 209.84, 247.61, 284.75, 318.92],
            [172.0, 209.84, 247.61, 284.75, 318.92],
        ],
        "CL1": [
            [1.50, 1.83, 2.16, 2.48, 2.78],
            [1.50, 1.83, 2.16, 2.48, 2.78],
        ],
        "CL2": [
            [1.50, 1.83, 2.16, 2.48, 2.78],
            [1.50, 1.83, 2.16, 2.48, 2.78],
        ],
        "CL3": [
            [1.50, 1.83, 2.16, 2.48, 2.78],
            [1.50, 1.83, 2.16, 2.48, 2.78],
        ],
        "REC": [
            [0.77, 0.94, 1.11, 1.28, 1.43],
            [0.77, 0.94, 1.11, 1.28, 1.43],
        ],
        "CT1": [
            [2.33, 2.84, 3.35, 3.85, 4.31],
            [2.33, 2.84, 3.35, 3.85, 4.31],
        ],
        "CT2": [
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.05, 0.06, 0.07, 0.08, 0.09],
        ],
        "CS1": [
            [600.0, 732.00, 863.76, 993.32, 1112.52],
            [600.0, 732.00, 863.76, 993.32, 1112.52],
        ],
        "CS2": [
            [6.25, 7.62, 8.99, 10.34, 11.58],
            [0.60, 0.73, 0.86, 0.99, 1.11],
        ],
        "CB": [
            [2.50, 3.05, 3.60, 4.14, 4.64],
            [0.25, 0.30, 0.35, 0.40, 0.45],
        ],
        "REV": [
            [40.0, 51.84, 63.24, 74.62, 85.82],
            [15.0, 19.44, 23.72, 27.98, 32.18],
            [19.0, 24.62, 30.04, 35.45, 40.77],
            [17.5, 22.68, 27.67, 32.65, 37.55],
            [3.1, 4.02, 4.90, 5.78, 6.65],
            [3.0, 3.89, 4.74, 5.59, 6.43],
            [1.3, 1.68, 2.05, 2.42, 2.79],
        ],
        # P[s][b][t] — P_Guncel (3 pazar x 2 batarya tipi x 5 dönem)
        "P": [
            [
                [62.50, 76.25, 89.975, 103.47125, 115.8878125],
                [5.9523809524, 7.2619047619, 8.5690476190, 9.8544047619, 11.0369345238],
            ]
            for _ in range(3)
        ],
        "I1_0_base": [0.0, 0.0],
        "I2_0_base": [0.0, 0.0],
        "I3_0_base": [0.0, 0.0],
        "I4_0_base": [0.0, 0.0],
        "DE_excel": {},
        "ev_baz_sayfa1": {},
        "mesafe_ij": {},
        "mesafe_jk": {},
        "mesafe_kl": {},
        "mesafe_km": {},
        "mesafe_ls": {},
        "kaynak_excel_okundu": False,
    }


def _excel_matris_oku(xl: pd.ExcelFile, sayfa: str, satir: int, sutun: int) -> Optional[np.ndarray]:
    """Başlıksız Excel parametre sayfasını güvenli şekilde float matris olarak okur."""
    if sayfa not in xl.sheet_names:
        return None
    try:
        df_s = pd.read_excel(xl, sheet_name=sayfa, header=None)
        mat = df_s.iloc[:satir, :sutun].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if mat.shape == (satir, sutun) and np.all(np.isfinite(mat)):
            return mat
    except Exception:
        return None
    return None


def _excel_mesafe_sozlugu_oku(xl: pd.ExcelFile, sayfa: str) -> Dict[Tuple[str, str], float]:
    """Mesafe sayfasını normalize edilmiş (kaynak_il, hedef_il) -> km sözlüğüne dönüştürür."""
    sonuc: Dict[Tuple[str, str], float] = {}
    if sayfa not in xl.sheet_names:
        return sonuc
    try:
        df_m = pd.read_excel(xl, sheet_name=sayfa)
        if df_m.empty or df_m.shape[1] < 2:
            return sonuc
        satir_kol = df_m.columns[0]
        hedef_kollar = [c for c in df_m.columns[1:] if not str(c).startswith("Unnamed")]
        for _, row in df_m.iterrows():
            kaynak = il_adini_normalize_et(row[satir_kol])
            if not kaynak:
                continue
            for col in hedef_kollar:
                hedef = il_adini_normalize_et(col)
                val = pd.to_numeric(row[col], errors="coerce")
                if hedef and pd.notna(val) and float(val) >= 0.0:
                    sonuc[(kaynak, hedef)] = float(val)
    except Exception:
        return {}
    return sonuc


def _excel_pazar_mesafe_oku(xl: pd.ExcelFile, sayfa: str = "mesafe dist_ls") -> Dict[Tuple[str, int], float]:
    """mesafe dist_ls sayfasını (il, pazar_indeks) -> km sözlüğüne dönüştürür."""
    sonuc: Dict[Tuple[str, int], float] = {}
    if sayfa not in xl.sheet_names:
        return sonuc
    try:
        df_m = pd.read_excel(xl, sheet_name=sayfa)
        if df_m.empty or df_m.shape[1] < 4:
            return sonuc
        satir_kol = df_m.columns[0]
        pazar_kollar = list(df_m.columns[1:4])
        for _, row in df_m.iterrows():
            kaynak = il_adini_normalize_et(row[satir_kol])
            if not kaynak:
                continue
            for s_idx, col in enumerate(pazar_kollar):
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val) and float(val) >= 0.0:
                    sonuc[(kaynak, s_idx)] = float(val)
    except Exception:
        return {}
    return sonuc


def tez_milp_parametrelerini_yukle(
    excel_yolu: Optional[str] = None,
) -> Dict[str, Any]:
    """
    `seminerveriler_son.xlsx` dosyasındaki 5 dönemli tez parametrelerini okur.
    Dosya veya herhangi bir sayfa eksikse aynı tez varsayılan değerlerine güvenli
    şekilde geri döner (`try-except` korumalı, göreli yol destekli).
    """
    veri = _tez_varsayilan_parametreler()

    aday_yollar: List[Path] = []
    if excel_yolu:
        aday_yollar.append(Path(excel_yolu))
    else:
        aday_yollar.append(Path("seminerveriler_son.xlsx"))
        aday_yollar.append(Path(__file__).resolve().parent / "seminerveriler_son.xlsx")

    secilen_yol = next((p for p in aday_yollar if p.is_file()), None)
    if secilen_yol is None:
        return veri

    try:
        xl = pd.ExcelFile(secilen_yol)
    except Exception:
        return veri

    veri["kaynak_excel_okundu"] = True

    # Vektör / matris parametreleri
    for anahtar, sayfa, r, c in [
        ("H", "H", 2, 1),
        ("G", "G", 2, 1),
        ("R1", "R1", 2, 1),
        ("R2", "R2", 2, 1),
        ("R3", "R3", 2, 1),
        ("R21", "R21", 2, 1),
        ("R22", "R22", 2, 1),
        ("R23", "R23", 2, 1),
        ("CR", "CR", 7, 1),
        ("RR", "RR", 7, 1),
    ]:
        mat = _excel_matris_oku(xl, sayfa, r, c)
        if mat is not None:
            veri[anahtar] = [float(x) for x in mat[:, 0]]

    for anahtar, sayfa in [
        ("Gama", "Gama"),
        ("CJ", "CJ"),
        ("CK", "CK"),
        ("CL1", "CL1"),
        ("CL2", "CL2"),
        ("CL3", "CL3"),
        ("REC", "REC"),
        ("CT1", "CT1"),
        ("CT2", "CT2"),
        ("CS1", "CS1"),
        ("CS2", "CS2"),
        ("CB", "CB"),
    ]:
        mat = _excel_matris_oku(xl, sayfa, 2, 5)
        if mat is not None:
            veri[anahtar] = mat.tolist()

    rev_mat = _excel_matris_oku(xl, "REV", 7, 5)
    if rev_mat is not None:
        veri["REV"] = rev_mat.tolist()

    for anahtar, sayfa in [("f_t", "f"), ("fk_t", "fk"), ("fl_t", "fl"), ("fm_t", "fm")]:
        mat = _excel_matris_oku(xl, sayfa, 1, 5)
        if mat is not None:
            veri[anahtar] = [float(x) for x in mat[0, :]]

    for anahtar, sayfa in [("Cap_j_base", "Cap_j"), ("Cap_k_base", "Cap_k"), ("Cap_m_base", "Cap_m")]:
        mat = _excel_matris_oku(xl, sayfa, 1, 2)
        if mat is not None:
            veri[anahtar] = [float(mat[0, 0]), float(mat[0, 1])]

    cap_l_mat = _excel_matris_oku(xl, "Cap_l", 1, 1)
    if cap_l_mat is not None:
        veri["Cap_l_base"] = float(cap_l_mat[0, 0])

    for anahtar, sayfa in [
        ("I1_0_base", "I1_0"),
        ("I2_0_base", "I2_0"),
        ("I3_0_base", "I3_0"),
        ("I4_0_base", "I4_0"),
    ]:
        mat = _excel_matris_oku(xl, sayfa, 1, 2)
        if mat is not None:
            veri[anahtar] = [float(mat[0, 0]), float(mat[0, 1])]

    # İkinci ömür pazar gelirleri P[s][b][t] (öncelik P_Guncel, yedek pazar geliri)
    p_mat = _excel_matris_oku(xl, "P_Guncel", 6, 5)
    if p_mat is not None:
        veri["P"] = [
            [p_mat[2 * s].tolist(), p_mat[2 * s + 1].tolist()]
            for s in range(3)
        ]
    elif "pazar geliri" in xl.sheet_names:
        try:
            df_pg = pd.read_excel(xl, sheet_name="pazar geliri")
            num_cols = df_pg.columns[2:7]
            pg_mat = df_pg[num_cols].iloc[:6].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if pg_mat.shape == (6, 5) and np.all(np.isfinite(pg_mat)):
                veri["P"] = [
                    [pg_mat[2 * s].tolist(), pg_mat[2 * s + 1].tolist()]
                    for s in range(3)
                ]
        except Exception:
            pass

    # DE[il, b, t] — Koordinatlar plaka sırasına (81 il x 2 batarya tipi = 162 satır) göre eşleme
    if "DE" in xl.sheet_names and "Koordinatlar" in xl.sheet_names:
        try:
            df_koord = pd.read_excel(xl, sheet_name="Koordinatlar")
            df_de = pd.read_excel(xl, sheet_name="DE")
            il_kol = "Il_Adi" if "Il_Adi" in df_koord.columns else df_koord.columns[1]
            plaka_illeri = [il_adini_normalize_et(x) for x in df_koord[il_kol].tolist()]
            de_mat = df_de.iloc[:, :5].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if len(plaka_illeri) * 2 <= de_mat.shape[0]:
                de_sozluk: Dict[Tuple[str, int, int], float] = {}
                for idx, il in enumerate(plaka_illeri):
                    if not il:
                        continue
                    for t in TEZ_DONEMLER:
                        de_sozluk[(il, 0, t)] = max(0.0, float(de_mat[2 * idx, t]))
                        de_sozluk[(il, 1, t)] = max(0.0, float(de_mat[2 * idx + 1, t]))
                veri["DE_excel"] = de_sozluk
        except Exception:
            pass

    # Sayfa1 baz EV dağılımı (kullanıcı farklı yıl EV verisi yüklediğinde ölçek referansı)
    if "Sayfa1" in xl.sheet_names:
        try:
            df_s1 = pd.read_excel(xl, sheet_name="Sayfa1")
            if "Şehir (İl)" in df_s1.columns and "İldeki Elektrikli Araç Sayısı" in df_s1.columns:
                baz_ev: Dict[str, float] = {}
                for _, row in df_s1.iterrows():
                    il = il_adini_normalize_et(row["Şehir (İl)"])
                    val = pd.to_numeric(row["İldeki Elektrikli Araç Sayısı"], errors="coerce")
                    if il and pd.notna(val) and float(val) >= 0.0:
                        baz_ev[il] = float(val)
                veri["ev_baz_sayfa1"] = baz_ev
        except Exception:
            pass

    # Mesafe matrisleri
    veri["mesafe_ij"] = _excel_mesafe_sozlugu_oku(xl, "mesafe dist_ij")
    veri["mesafe_jk"] = _excel_mesafe_sozlugu_oku(xl, "mesafe dist_jk")
    veri["mesafe_kl"] = _excel_mesafe_sozlugu_oku(xl, "mesafe dist_kl")
    veri["mesafe_km"] = _excel_mesafe_sozlugu_oku(xl, "mesafe dist_km")
    veri["mesafe_ls"] = _excel_pazar_mesafe_oku(xl, "mesafe dist_ls")

    return veri


def _mesafe_getir(
    il1: str,
    il2: str,
    excel_sozluk: Dict[Tuple[str, str], float],
) -> float:
    """Excel mesafe matrisinden mesafeyi alır; çift yoksa Haversine (km) hesaplar."""
    if il1 == il2:
        return 0.0
    if (il1, il2) in excel_sozluk:
        return float(excel_sozluk[(il1, il2)])
    if (il2, il1) in excel_sozluk:
        return float(excel_sozluk[(il2, il1)])
    if il1 in IL_KOORDINATLARI and il2 in IL_KOORDINATLARI:
        return haversine_km(*IL_KOORDINATLARI[il1], *IL_KOORDINATLARI[il2])
    return 0.0


def _pazar_mesafe_getir(
    il: str,
    s_idx: int,
    excel_ls: Dict[Tuple[str, int], float],
) -> float:
    """İkinci Ömür tesisi -> Pazar mesafesini (km) Excel'den veya Haversine ile döndürür."""
    if (il, s_idx) in excel_ls:
        return float(excel_ls[(il, s_idx)])
    if il in IL_KOORDINATLARI and 0 <= s_idx < len(TEZ_PAZAR_KOORDINATLARI):
        return haversine_km(*IL_KOORDINATLARI[il], *TEZ_PAZAR_KOORDINATLARI[s_idx])
    return 0.0


class _LineerModelKurucu:
    """
    Seyrek matris tabanlı MILP kurucu.
    Hem SciPy `scipy.optimize.milp` (HiGHS) hem de PuLP (2.x/3.x/4.x uyumlu)
    çözücüleriyle ortak model kurulumunu sağlar.
    """

    def __init__(self) -> None:
        self.adlar: List[str] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.binary: List[bool] = []
        self.obj: List[float] = []  # Maksimizasyon katsayısı
        self._satir_idx: List[int] = []
        self._sutun_idx: List[int] = []
        self._degerler: List[float] = []
        self._kisit_lb: List[float] = []
        self._kisit_ub: List[float] = []
        self._kisit_sayisi: int = 0

    def degisken_ekle(
        self,
        ad: str,
        lb: float = 0.0,
        ub: float = float("inf"),
        binary: bool = False,
        obj: float = 0.0,
    ) -> int:
        idx = len(self.adlar)
        self.adlar.append(ad)
        self.lb.append(0.0 if binary else float(lb))
        self.ub.append(1.0 if binary else float(ub))
        self.binary.append(bool(binary))
        self.obj.append(float(obj))
        return idx

    def kisit_ekle(
        self,
        terimler: Sequence[Tuple[int, float]],
        lb: float = float("-inf"),
        ub: float = float("inf"),
    ) -> None:
        birlesik: Dict[int, float] = {}
        for col, katsayi in terimler:
            k_val = float(katsayi)
            if abs(k_val) > 1e-15:
                birlesik[col] = birlesik.get(col, 0.0) + k_val
        r = self._kisit_sayisi
        for col, katsayi in birlesik.items():
            if abs(katsayi) > 1e-15:
                self._satir_idx.append(r)
                self._sutun_idx.append(col)
                self._degerler.append(katsayi)
        self._kisit_lb.append(float(lb))
        self._kisit_ub.append(float(ub))
        self._kisit_sayisi += 1

    def coz_scipy(
        self,
        time_limit: float = 60.0,
        mip_rel_gap: float = 1e-4,
    ) -> Tuple[str, Optional[np.ndarray], float]:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csc_matrix

        n_var = len(self.adlar)
        n_con = self._kisit_sayisi

        # Maksimizasyon -> -obj minimizasyonu
        c = -np.asarray(self.obj, dtype=float)
        integrality = np.asarray([1 if b else 0 for b in self.binary], dtype=int)
        bounds = Bounds(
            lb=np.asarray(self.lb, dtype=float),
            ub=np.asarray(self.ub, dtype=float),
        )
        A = csc_matrix(
            (
                np.asarray(self._degerler, dtype=float),
                (np.asarray(self._satir_idx, dtype=int), np.asarray(self._sutun_idx, dtype=int)),
            ),
            shape=(n_con, n_var),
        )
        constraints = LinearConstraint(
            A,
            lb=np.asarray(self._kisit_lb, dtype=float),
            ub=np.asarray(self._kisit_ub, dtype=float),
        )
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap)},
        )
        if res.x is not None and res.status in (0, 1):
            durum = "Optimal" if res.status == 0 else "Optimal"
            return durum, np.asarray(res.x, dtype=float), float(-res.fun)
        if res.status == 2:
            return "Infeasible", None, 0.0
        if res.status == 3:
            return "Unbounded", None, 0.0
        return f"Hata ({res.message})", None, 0.0

    def coz_pulp(self, time_limit: float = 60.0) -> Tuple[str, Optional[np.ndarray], float]:
        import pulp

        model = pulp.LpProblem("EV_5_Donem_MILP", pulp.LpMaximize)
        n_var = len(self.adlar)
        pulp_vars = []
        for idx in range(n_var):
            ad = self.adlar[idx]
            lb_v = None if math.isinf(self.lb[idx]) else self.lb[idx]
            ub_v = None if math.isinf(self.ub[idx]) else self.ub[idx]
            cat_v = "Integer" if self.binary[idx] else "Continuous"
            if hasattr(model, "add_variable"):
                v = model.add_variable(ad, lowBound=lb_v, upBound=ub_v, cat=cat_v)
            else:
                v = pulp.LpVariable(ad, lowBound=lb_v, upBound=ub_v, cat=cat_v)
            pulp_vars.append(v)

        model += pulp.lpSum(
            self.obj[idx] * pulp_vars[idx] for idx in range(n_var) if abs(self.obj[idx]) > 1e-15
        )

        # Kısıtları satır bazında topla
        satirlar: Dict[int, List[Tuple[int, float]]] = {}
        for r, c, val in zip(self._satir_idx, self._sutun_idx, self._degerler):
            satirlar.setdefault(r, []).append((c, val))

        for r in range(self._kisit_sayisi):
            expr = pulp.lpSum(val * pulp_vars[c] for c, val in satirlar.get(r, []))
            lb_r = self._kisit_lb[r]
            ub_r = self._kisit_ub[r]
            if not math.isinf(lb_r) and not math.isinf(ub_r) and abs(lb_r - ub_r) <= 1e-12:
                model += expr == lb_r, f"C_eq_{r}"
            else:
                if not math.isinf(lb_r):
                    model += expr >= lb_r, f"C_ge_{r}"
                if not math.isinf(ub_r):
                    model += expr <= ub_r, f"C_le_{r}"

        cozucu = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(time_limit))
        status = model.solve(cozucu)
        durum_str = status.status_str if hasattr(status, "status_str") else pulp.LpStatus.get(status, str(status))
        if str(durum_str).lower().startswith("optimal"):
            x_vec = np.array([float(pulp.value(v) or 0.0) for v in pulp_vars], dtype=float)
            obj_val = float(pulp.value(model.objective) or 0.0)
            return "Optimal", x_vec, obj_val
        return str(durum_str), None, 0.0


def _ui_olcek_hesapla(params: Dict[str, Any], anahtar: str) -> float:
    """Kullanıcı senaryo parametresini arayüz varsayılanına göre oransal ölçekler."""
    if anahtar not in params or params[anahtar] is None:
        return 1.0
    ref = _UI_VARSAYILAN_REFERANS.get(anahtar, 0.0)
    if ref <= 1e-12:
        return 1.0
    return max(0.0, float(params[anahtar]) / ref)


def _de_matrisi_olustur(
    iller: List[str],
    params: Dict[str, Any],
    tez_veri: Dict[str, Any],
) -> Dict[Tuple[str, int, int], float]:
    """
    5 dönemlik DE[i, b, t] talep matrisini oluşturur.
    - Excel `DE` sayfası varsa baz senaryoda birebir kullanılır.
    - Kullanıcı farklı bir EV veri seti yüklediyse veya `binek_payi` oranını
      değiştirdiyse talep tüm dönemlere oransal olarak yansıtılır.
    """
    de_excel: Dict[Tuple[str, int, int], float] = tez_veri.get("DE_excel", {})
    ev_data_raw: Dict[str, float] = params.get("ev_data", {}) or {}
    ev_data: Dict[str, float] = {
        il_adini_normalize_et(k): max(0.0, float(v))
        for k, v in ev_data_raw.items()
        if il_adini_normalize_et(k)
    }
    baz_ev: Dict[str, float] = tez_veri.get("ev_baz_sayfa1", {})

    binek_payi_ui = params.get("binek_payi", 95)
    binek_degisti = binek_payi_ui is not None and abs(float(binek_payi_ui) - 95.0) > 1e-6
    binek_orani = float(binek_payi_ui) / 100.0 if binek_payi_ui is not None else 0.95

    # Kullanıcı özel bir EV veri seti mi yükledi kontrolü
    farkli_ev_verisi = False
    if ev_data and baz_ev:
        toplam_yeni = sum(ev_data.values())
        toplam_baz = sum(baz_ev.values())
        if toplam_baz > 0 and abs(toplam_yeni - toplam_baz) / toplam_baz > 0.05:
            farkli_ev_verisi = True

    de_sonuc: Dict[Tuple[str, int, int], float] = {}
    for il in iller:
        il_excelde = any((il, b, t) in de_excel for b in (0, 1) for t in TEZ_DONEMLER)
        if il_excelde:
            olcek = 1.0
            if farkli_ev_verisi and il in ev_data and baz_ev.get(il, 0.0) > 0.0:
                olcek = ev_data[il] / baz_ev[il]
            for t in TEZ_DONEMLER:
                d0 = de_excel.get((il, 0, t), 0.0) * olcek
                d1 = de_excel.get((il, 1, t), 0.0) * olcek
                if binek_degisti:
                    top_t = d0 + d1
                    d0 = top_t * binek_orani
                    d1 = top_t * (1.0 - binek_orani)
                de_sonuc[(il, 0, t)] = max(0.0, d0)
                de_sonuc[(il, 1, t)] = max(0.0, d1)
        else:
            ev_il = ev_data.get(il, 0.0)
            for t in TEZ_DONEMLER:
                top_t = ev_il * _TEZ_DONEM_DAGILIM_PAYI[t] * 3.0
                de_sonuc[(il, 0, t)] = max(0.0, top_t * binek_orani)
                de_sonuc[(il, 1, t)] = max(0.0, top_t * (1.0 - binek_orani))

    return de_sonuc


def milp_5_donem_coz(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tezdeki 5 dönemli (T = [0..4]), stoklu (I1..I4) ve kâr maksimizasyonlu tam
    tersine lojistik MILP modelini çözer.

    Çözücü stratejisi:
    - Varsayılan ('auto'): Önce SciPy `scipy.optimize.milp` (HiGHS) ile hızlı ve
      PuLP sürüm uyumsuzluklarından bağımsız çözer; SciPy erişilemezse PuLP
      (2.x/3.x/4.x uyumlu CBC) çözücüsüne otomatik geçer.
    - Dönüş sözlüğü hem 5 Yıllık Genel Özet (`genel_ozet`), hem dönem bazlı
      ayrıntılar (`donemler[1]..donemler[5]`), hem de mevcut Modül 2 ekranıyla
      uyumlu üst düzey anahtarları içerir.
    """
    tez = tez_milp_parametrelerini_yukle(params.get("excel_yolu"))

    T = list(TEZ_DONEMLER)
    B = list(TEZ_BATARYA_TIPLERI)
    N = list(TEZ_HAMMADDELER)
    S = [0, 1, 2]
    pazar_adlari = list(tez["pazar_adlari"])

    # Küme listelerini normalize ve tekil hale getir
    def _tekil_iller(liste: Sequence[str]) -> List[str]:
        sonuc: List[str] = []
        for x in liste:
            il = il_adini_normalize_et(x)
            if il and il not in sonuc:
                sonuc.append(il)
        return sonuc

    ev_dict_raw = params.get("ev_data", {}) or {}
    I = _tekil_iller(list(ev_dict_raw.keys()))
    if not I and tez["DE_excel"]:
        I = _tekil_iller([k[0] for k in tez["DE_excel"].keys()])
    J = _tekil_iller(params.get("aday_J", []))
    K = _tekil_iller(params.get("aday_K", []))
    L = _tekil_iller(params.get("aday_L", []))
    M = _tekil_iller(params.get("aday_M", []))

    if not (I and J and K and L and M):
        return {
            "durum": "Infeasible",
            "cozucu": "Yok",
            "amac_degeri": 0.0,
            "toplam_net_kar": 0.0,
            "toplam_gelir": 0.0,
            "toplam_maliyet": 0.0,
            "genel_ozet": {},
            "donemler": {},
            "maliyet_kalemleri": {},
            "x_j": {}, "x_k": {}, "x_l": {}, "x_m": {},
            "x_ij": [], "z_ijb": [], "W_jkb": [], "beta_kb": [],
            "W_kmb": [], "W_klb": [], "W_kb": [], "theta_mb": [], "theta_mn": [],
        }

    # --- Parametreler ve What-If senaryo uyarlaması ---
    DE = _de_matrisi_olustur(I, params, tez)

    H = [float(tez["H"][0]), float(tez["H"][1])]
    G = [float(tez["G"][0]), float(tez["G"][1])]  # kg / hücre

    # Oran parametreleri (kullanıcı doğrudan değiştirebilir)
    if params.get("gamma") is not None:
        g_val = float(params["gamma"]) / 100.0
        Gama = [[g_val for _ in T] for _ in range(2)]
    else:
        Gama = [[float(tez["Gama"][b][t]) for t in T] for b in range(2)]

    R1_val = float(params["R1"]) / 100.0 if params.get("R1") is not None else float(tez["R1"][0])
    R2_val = float(params["R2"]) / 100.0 if params.get("R2") is not None else float(tez["R2"][0])
    R3_val = float(params["R3"]) / 100.0 if params.get("R3") is not None else max(0.0, 1.0 - R1_val - R2_val)
    R1 = [R1_val, R1_val]
    R2 = [R2_val, R2_val]
    R3 = [R3_val, R3_val]

    R21 = [float(tez["R21"][0]), float(tez["R21"][1])]
    R22 = [float(tez["R22"][0]), float(tez["R22"][1])]
    R23 = [float(tez["R23"][0]), float(tez["R23"][1])]
    CR = [float(x) for x in tez["CR"]]
    RR = [float(x) for x in tez["RR"]]

    # Kapasiteler (What-If oransal ölçekleme, ağırlık birimi kg)
    s_cap_j = _ui_olcek_hesapla(params, "cap_j")
    s_cap_k = _ui_olcek_hesapla(params, "cap_k")
    s_cap_l = _ui_olcek_hesapla(params, "cap_l")
    s_cap_m = _ui_olcek_hesapla(params, "cap_m")

    Cap_j = {(j, b): float(tez["Cap_j_base"][b]) * s_cap_j for j in J for b in range(2)}
    Cap_k = {(k, b): float(tez["Cap_k_base"][b]) * s_cap_k for k in K for b in range(2)}
    Cap_l = {l: float(tez["Cap_l_base"]) * s_cap_l for l in L}
    Cap_m = {(m, b): float(tez["Cap_m_base"][b]) * s_cap_m for m in M for b in range(2)}

    # Yıllık eşdeğer sabit tesis maliyetleri
    s_fj = _ui_olcek_hesapla(params, "fj")
    s_fk = _ui_olcek_hesapla(params, "fk")
    s_fl = _ui_olcek_hesapla(params, "fl")
    s_fm = _ui_olcek_hesapla(params, "fm")

    f = {(j, t): float(tez["f_t"][t]) * s_fj for j in J for t in T}
    fk = {(k, t): float(tez["fk_t"][t]) * s_fk for k in K for t in T}
    fl = {(l, t): float(tez["fl_t"][t]) * s_fl for l in L for t in T}
    fm = {(m, t): float(tez["fm_t"][t]) * s_fm for m in M for t in T}

    # Birim işlem, taşıma, bertaraf, stok maliyetleri
    s_ct1 = _ui_olcek_hesapla(params, "c3_tasima")
    s_ct2 = _ui_olcek_hesapla(params, "c4_tasima")
    s_ck0 = _ui_olcek_hesapla(params, "c1_binek")
    s_ck1 = _ui_olcek_hesapla(params, "c1_agir")
    s_cb = _ui_olcek_hesapla(params, "c5_bertaraf")
    hucre_bazli_cc = bool(params.get("hucre_bazli_cc", False))

    CJ = [
        [float(tez["CJ"][b][t]) * ((H[b] / 96.0) if hucre_bazli_cc else 1.0) for t in T]
        for b in range(2)
    ]
    CK = [
        [float(tez["CK"][0][t]) * s_ck0 for t in T],
        [float(tez["CK"][1][t]) * s_ck1 for t in T],
    ]
    CL1 = [[float(tez["CL1"][b][t]) for t in T] for b in range(2)]
    CL2 = [[float(tez["CL2"][b][t]) for t in T] for b in range(2)]
    CL3 = [[float(tez["CL3"][b][t]) for t in T] for b in range(2)]
    REC = [[float(tez["REC"][b][t]) for t in T] for b in range(2)]
    CT1 = [[float(tez["CT1"][b][t]) * s_ct1 for t in T] for b in range(2)]
    CT2 = [[float(tez["CT2"][b][t]) * s_ct2 for t in T] for b in range(2)]
    CS1 = [[float(tez["CS1"][b][t]) for t in T] for b in range(2)]
    CS2 = [[float(tez["CS2"][b][t]) for t in T] for b in range(2)]
    CB = [[float(tez["CB"][b][t]) * s_cb for t in T] for b in range(2)]

    # Gelir parametreleri (Hammadde REV[n,t] ve İkinci Ömür P[s,b,t])
    s_rev = {
        0: _ui_olcek_hesapla(params, "rev_kobalt"),
        1: _ui_olcek_hesapla(params, "rev_lityum"),
        2: _ui_olcek_hesapla(params, "rev_nikel"),
    }
    REV = [
        [float(tez["REV"][n][t]) * s_rev.get(n, 1.0) for t in T]
        for n in range(len(N))
    ]
    P = [
        [[float(tez["P"][s][b][t]) for t in T] for b in range(2)]
        for s in S
    ]

    # Başlangıç stokları
    I1_0 = {(j, b): float(tez["I1_0_base"][b]) for j in J for b in range(2)}
    I2_0 = {(k, b): float(tez["I2_0_base"][b]) for k in K for b in range(2)}
    I3_0 = {(k, b): float(tez["I3_0_base"][b]) for k in K for b in range(2)}
    I4_0 = {(l, b): float(tez["I4_0_base"][b]) for l in L for b in range(2)}

    # Mesafe matrisleri (km)
    d_ij = {(i, j): _mesafe_getir(i, j, tez["mesafe_ij"]) for i in I for j in J}
    d_jk = {(j, k): _mesafe_getir(j, k, tez["mesafe_jk"]) for j in J for k in K}
    d_kl = {(k, l): _mesafe_getir(k, l, tez["mesafe_kl"]) for k in K for l in L}
    d_km = {(k, m): _mesafe_getir(k, m, tez["mesafe_km"]) for k in K for m in M}
    d_ls = {(l, s): _pazar_mesafe_getir(l, s, tez["mesafe_ls"]) for l in L for s in S}

    # --- Model Kurulumu (Maksimizasyon: Toplam Net Kâr) ---
    kurucu = _LineerModelKurucu()

    # 1. Tesis açılış / aktiflik karar değişkenleri (Binary)
    x1 = {(j, t): kurucu.degisken_ekle(f"x1_{j}_{t}", binary=True, obj=-f[(j, t)]) for j in J for t in T}
    x2 = {(k, t): kurucu.degisken_ekle(f"x2_{k}_{t}", binary=True, obj=-fk[(k, t)]) for k in K for t in T}
    x3 = {(l, t): kurucu.degisken_ekle(f"x3_{l}_{t}", binary=True, obj=-fl[(l, t)]) for l in L for t in T}
    x4 = {(m, t): kurucu.degisken_ekle(f"x4_{m}_{t}", binary=True, obj=-fm[(m, t)]) for m in M for t in T}

    # 2. Atama (sigma) ve Toplama (Z) — yalnızca toplanabilir talebi > 0 olan (i,b,t) için
    aktif_ibt = [
        (i, b, t)
        for i in I
        for b in range(2)
        for t in T
        if DE[(i, b, t)] * Gama[b][t] > 1e-12
    ]
    sigma: Dict[Tuple[str, str, int, int], int] = {}
    Z: Dict[Tuple[str, str, int, int], int] = {}
    for i, b, t in aktif_ibt:
        for j in J:
            sigma[(i, j, b, t)] = kurucu.degisken_ekle(f"sigma_{i}_{j}_{b}_{t}", binary=True, obj=0.0)
            maliyet_z = CJ[b][t] + CT1[b][t] * d_ij[(i, j)]
            Z[(i, j, b, t)] = kurucu.degisken_ekle(f"Z_{i}_{j}_{b}_{t}", lb=0.0, obj=-maliyet_z)

    # 3. Tesisler arası akışlar ve üretim değişkenleri
    W1 = {
        (j, k, b, t): kurucu.degisken_ekle(
            f"W1_{j}_{k}_{b}_{t}", lb=0.0, obj=-(CT1[b][t] * d_jk[(j, k)])
        )
        for j in J for k in K for b in range(2) for t in T
    }
    beta = {
        (k, b, t): kurucu.degisken_ekle(f"beta_{k}_{b}_{t}", lb=0.0, obj=-CK[b][t])
        for k in K for b in range(2) for t in T
    }
    W2 = {
        (k, l, b, t): kurucu.degisken_ekle(
            f"W2_{k}_{l}_{b}_{t}", lb=0.0, obj=-(CT2[b][t] * d_kl[(k, l)])
        )
        for k in K for l in L for b in range(2) for t in T
    }
    W3 = {
        (k, m, b, t): kurucu.degisken_ekle(
            f"W3_{k}_{m}_{b}_{t}", lb=0.0, obj=-(CT2[b][t] * d_km[(k, m)])
        )
        for k in K for m in M for b in range(2) for t in T
    }
    W4 = {
        (k, b, t): kurucu.degisken_ekle(f"W4_{k}_{b}_{t}", lb=0.0, obj=-CB[b][t])
        for k in K for b in range(2) for t in T
    }
    Wrep = {
        (l, b, t): kurucu.degisken_ekle(f"Wrep_{l}_{b}_{t}", lb=0.0, obj=-CL1[b][t])
        for l in L for b in range(2) for t in T
    }
    Wrem = {
        (l, b, t): kurucu.degisken_ekle(f"Wrem_{l}_{b}_{t}", lb=0.0, obj=-CL2[b][t])
        for l in L for b in range(2) for t in T
    }
    Wreu = {
        (l, b, t): kurucu.degisken_ekle(f"Wreu_{l}_{b}_{t}", lb=0.0, obj=-CL3[b][t])
        for l in L for b in range(2) for t in T
    }
    V = {
        (l, s, b, t): kurucu.degisken_ekle(
            f"V_{l}_{s}_{b}_{t}", lb=0.0, obj=(P[s][b][t] - CT2[b][t] * d_ls[(l, s)])
        )
        for l in L for s in S for b in range(2) for t in T
    }
    theta = {
        (l, b, t): kurucu.degisken_ekle(f"theta_{l}_{b}_{t}", lb=0.0, obj=0.0)
        for l in L for b in range(2) for t in T
    }
    mu = {
        (m, b, t): kurucu.degisken_ekle(f"mu_{m}_{b}_{t}", lb=0.0, obj=-REC[b][t])
        for m in M for b in range(2) for t in T
    }
    delta = {
        (m, n, t): kurucu.degisken_ekle(f"delta_{m}_{n}_{t}", lb=0.0, obj=REV[n][t])
        for m in M for n in range(len(N)) for t in T
    }

    # 4. Dönemler arası stok değişkenleri (I1..I4)
    I1 = {
        (j, b, t): kurucu.degisken_ekle(f"I1_{j}_{b}_{t}", lb=0.0, obj=-CS1[b][t])
        for j in J for b in range(2) for t in T
    }
    I2 = {
        (k, b, t): kurucu.degisken_ekle(f"I2_{k}_{b}_{t}", lb=0.0, obj=-CS1[b][t])
        for k in K for b in range(2) for t in T
    }
    I3 = {
        (k, b, t): kurucu.degisken_ekle(f"I3_{k}_{b}_{t}", lb=0.0, obj=-CS2[b][t])
        for k in K for b in range(2) for t in T
    }
    I4 = {
        (l, b, t): kurucu.degisken_ekle(f"I4_{l}_{b}_{t}", lb=0.0, obj=-CS2[b][t])
        for l in L for b in range(2) for t in T
    }

    # --- Kısıtlar ---
    # K1: Tesis açık kalma (monotonluk: x[.,t] <= x[.,t+1])
    for t in T[:-1]:
        for j in J:
            kurucu.kisit_ekle([(x1[(j, t)], 1.0), (x1[(j, t + 1)], -1.0)], ub=0.0)
        for k in K:
            kurucu.kisit_ekle([(x2[(k, t)], 1.0), (x2[(k, t + 1)], -1.0)], ub=0.0)
        for l in L:
            kurucu.kisit_ekle([(x3[(l, t)], 1.0), (x3[(l, t + 1)], -1.0)], ub=0.0)
        for m in M:
            kurucu.kisit_ekle([(x4[(m, t)], 1.0), (x4[(m, t + 1)], -1.0)], ub=0.0)

    # K2: Her dönemde en fazla 1 İkinci Ömür ve 1 Geri Dönüşüm merkezi
    for t in T:
        kurucu.kisit_ekle([(x3[(l, t)], 1.0) for l in L], ub=1.0)
        kurucu.kisit_ekle([(x4[(m, t)], 1.0) for m in M], ub=1.0)

    # K3: İl atamaları ve toplanan batarya miktarı (Z == DE * Gama * sigma)
    aktif_j_map: Dict[Tuple[str, int, int], List[str]] = {}
    for i, b, t in aktif_ibt:
        talep_mik = DE[(i, b, t)] * Gama[b][t]
        kurucu.kisit_ekle([(sigma[(i, j, b, t)], 1.0) for j in J], lb=1.0, ub=1.0)
        for j in J:
            kurucu.kisit_ekle([(sigma[(i, j, b, t)], 1.0), (x1[(j, t)], -1.0)], ub=0.0)
            kurucu.kisit_ekle(
                [(Z[(i, j, b, t)], 1.0), (sigma[(i, j, b, t)], -talep_mik)],
                lb=0.0,
                ub=0.0,
            )
            aktif_j_map.setdefault((j, b, t), []).append(i)

    # K4: Toplama merkezi stok dengesi (I1) ve kapasitesi (Cap_j)
    for j in J:
        for b in range(2):
            for t in T:
                i_list = aktif_j_map.get((j, b, t), [])
                # I1[t] - I1[t-1] - sum_i Z + sum_k W1 == 0
                terimler_denge = [(I1[(j, b, t)], 1.0)]
                terimler_denge.extend((Z[(i, j, b, t)], -1.0) for i in i_list)
                terimler_denge.extend((W1[(j, k, b, t)], 1.0) for k in K)
                if t == 0:
                    kurucu.kisit_ekle(terimler_denge, lb=I1_0[(j, b)], ub=I1_0[(j, b)])
                else:
                    terimler_denge.append((I1[(j, b, t - 1)], -1.0))
                    kurucu.kisit_ekle(terimler_denge, lb=0.0, ub=0.0)

                # Kapasite: I1_prev + sum_i Z <= Cap_j * x1[j,t]
                terimler_cap = [(Z[(i, j, b, t)], 1.0) for i in i_list]
                terimler_cap.append((x1[(j, t)], -Cap_j[(j, b)]))
                if t == 0:
                    kurucu.kisit_ekle(terimler_cap, ub=-I1_0[(j, b)])
                else:
                    terimler_cap.append((I1[(j, b, t - 1)], 1.0))
                    kurucu.kisit_ekle(terimler_cap, ub=0.0)

    # K5: İşleme merkezi batarya stoğu (I2), söküm kapasitesi (Cap_k), hücre stoğu (I3) ve SoH oranları
    toplam_batarya_ust = [
        max(1.0, sum(DE[(i, b, t)] * Gama[b][t] for i in I for t in T) + sum(I1_0[(j, b)] for j in J))
        for b in range(2)
    ]
    for k in K:
        for b in range(2):
            big_m_w1 = max(Cap_k[(k, b)], toplam_batarya_ust[b])
            for t in T:
                # I2[t] - I2[t-1] - sum_j W1 + beta == 0
                terimler_i2 = [(I2[(k, b, t)], 1.0), (beta[(k, b, t)], 1.0)]
                terimler_i2.extend((W1[(j, k, b, t)], -1.0) for j in J)
                if t == 0:
                    kurucu.kisit_ekle(terimler_i2, lb=I2_0[(k, b)], ub=I2_0[(k, b)])
                else:
                    terimler_i2.append((I2[(k, b, t - 1)], -1.0))
                    kurucu.kisit_ekle(terimler_i2, lb=0.0, ub=0.0)

                # Kapasite ve tesis aktivasyon bağı
                kurucu.kisit_ekle([(beta[(k, b, t)], 1.0), (x2[(k, t)], -Cap_k[(k, b)])], ub=0.0)
                terimler_w1_bag = [(W1[(j, k, b, t)], 1.0) for j in J]
                terimler_w1_bag.append((x2[(k, t)], -big_m_w1))
                kurucu.kisit_ekle(terimler_w1_bag, ub=0.0)

                # Hücre stok dengesi: I3[t] - I3[t-1] - H*beta + sum_l W2 + sum_m W3 + W4 == 0
                terimler_i3 = [
                    (I3[(k, b, t)], 1.0),
                    (beta[(k, b, t)], -H[b]),
                    (W4[(k, b, t)], 1.0),
                ]
                terimler_i3.extend((W2[(k, l, b, t)], 1.0) for l in L)
                terimler_i3.extend((W3[(k, m, b, t)], 1.0) for m in M)
                if t == 0:
                    kurucu.kisit_ekle(terimler_i3, lb=I3_0[(k, b)], ub=I3_0[(k, b)])
                else:
                    terimler_i3.append((I3[(k, b, t - 1)], -1.0))
                    kurucu.kisit_ekle(terimler_i3, lb=0.0, ub=0.0)

                # Kalite (SoH) akış kısıtları (avail = I3_prev + H*beta)
                # sum_m W3 <= R1 * avail
                t_r1 = [(W3[(k, m, b, t)], 1.0) for m in M]
                t_r1.append((beta[(k, b, t)], -R1[b] * H[b]))
                if t == 0:
                    kurucu.kisit_ekle(t_r1, ub=R1[b] * I3_0[(k, b)])
                else:
                    t_r1.append((I3[(k, b, t - 1)], -R1[b]))
                    kurucu.kisit_ekle(t_r1, ub=0.0)

                # sum_l W2 <= R2 * avail
                t_r2 = [(W2[(k, l, b, t)], 1.0) for l in L]
                t_r2.append((beta[(k, b, t)], -R2[b] * H[b]))
                if t == 0:
                    kurucu.kisit_ekle(t_r2, ub=R2[b] * I3_0[(k, b)])
                else:
                    t_r2.append((I3[(k, b, t - 1)], -R2[b]))
                    kurucu.kisit_ekle(t_r2, ub=0.0)

                # W4 >= R3 * avail (bertaraf maliyetli olduğu için optimalde tam R3*avail)
                t_r3 = [(W4[(k, b, t)], 1.0), (beta[(k, b, t)], -R3[b] * H[b])]
                if t == 0:
                    kurucu.kisit_ekle(t_r3, lb=R3[b] * I3_0[(k, b)])
                else:
                    t_r3.append((I3[(k, b, t - 1)], -R3[b]))
                    kurucu.kisit_ekle(t_r3, lb=0.0)

    # K6: İkinci Ömür merkezi (L) alt işlemler (Wrep, Wrem, Wreu), ağırlık (theta, kg), kapasite ve stok (I4)
    for l in L:
        for t in T:
            for b in range(2):
                # Wrep, Wrem, Wreu == R21/R22/R23 * sum_k W2
                for var_alt, oran in [
                    (Wrep[(l, b, t)], R21[b]),
                    (Wrem[(l, b, t)], R22[b]),
                    (Wreu[(l, b, t)], R23[b]),
                ]:
                    terim_alt = [(var_alt, 1.0)] + [(W2[(k, l, b, t)], -oran) for k in K]
                    kurucu.kisit_ekle(terim_alt, lb=0.0, ub=0.0)

                # theta[l,b,t] == G[b] * sum_k W2[k,l,b,t] (kg)
                terim_th = [(theta[(l, b, t)], 1.0)] + [(W2[(k, l, b, t)], -G[b]) for k in K]
                kurucu.kisit_ekle(terim_th, lb=0.0, ub=0.0)

                # Stok dengesi: I4[t] - I4[t-1] - sum_k W2 + sum_s V == 0
                terim_i4 = [(I4[(l, b, t)], 1.0)]
                terim_i4.extend((W2[(k, l, b, t)], -1.0) for k in K)
                terim_i4.extend((V[(l, s, b, t)], 1.0) for s in S)
                if t == 0:
                    kurucu.kisit_ekle(terim_i4, lb=I4_0[(l, b)], ub=I4_0[(l, b)])
                else:
                    terim_i4.append((I4[(l, b, t - 1)], -1.0))
                    kurucu.kisit_ekle(terim_i4, lb=0.0, ub=0.0)

            # Kapasite (kg): sum_b theta[l,b,t] <= Cap_l[l] * x3[l,t]
            kurucu.kisit_ekle(
                [(theta[(l, 0, t)], 1.0), (theta[(l, 1, t)], 1.0), (x3[(l, t)], -Cap_l[l])],
                ub=0.0,
            )

    # K7: Geri Dönüşüm merkezi (M) ağırlık (mu, kg), kapasite (Cap_m, kg) ve hammadde üretimi (delta, kg)
    for m in M:
        for t in T:
            for b in range(2):
                # mu[m,b,t] == G[b] * sum_k W3[k,m,b,t] (kg)
                terim_mu = [(mu[(m, b, t)], 1.0)] + [(W3[(k, m, b, t)], -G[b]) for k in K]
                kurucu.kisit_ekle(terim_mu, lb=0.0, ub=0.0)
                kurucu.kisit_ekle([(mu[(m, b, t)], 1.0), (x4[(m, t)], -Cap_m[(m, b)])], ub=0.0)

            for n_idx in range(len(N)):
                kats = CR[n_idx] * RR[n_idx]
                kurucu.kisit_ekle(
                    [
                        (delta[(m, n_idx, t)], 1.0),
                        (mu[(m, 0, t)], -kats),
                        (mu[(m, 1, t)], -kats),
                    ],
                    lb=0.0,
                    ub=0.0,
                )

    # --- Çözüm (SciPy HiGHS birincil + PuLP yedek uyumlu altyapı) ---
    tercih = str(params.get("cozucu", "auto")).lower()
    time_limit = float(params.get("time_limit", 60.0))
    durum = "Hata"
    x_vec: Optional[np.ndarray] = None
    cozucu_adi = ""

    if tercih == "pulp":
        try:
            durum, x_vec, _ = kurucu.coz_pulp(time_limit=time_limit)
            cozucu_adi = "PuLP CBC"
        except Exception:
            durum, x_vec, _ = kurucu.coz_scipy(time_limit=time_limit)
            cozucu_adi = "SciPy HiGHS (Yedek)"
    else:
        try:
            durum, x_vec, _ = kurucu.coz_scipy(time_limit=time_limit)
            cozucu_adi = "SciPy HiGHS"
        except Exception:
            durum, x_vec, _ = kurucu.coz_pulp(time_limit=time_limit)
            cozucu_adi = "PuLP CBC (Yedek)"

    if durum != "Optimal" or x_vec is None:
        return {
            "durum": durum,
            "cozucu": cozucu_adi,
            "amac_degeri": 0.0,
            "toplam_net_kar": 0.0,
            "toplam_gelir": 0.0,
            "toplam_maliyet": 0.0,
            "genel_ozet": {},
            "donemler": {},
            "maliyet_kalemleri": {},
            "x_j": {j: 0 for j in J},
            "x_k": {k: 0 for k in K},
            "x_l": {l: 0 for l in L},
            "x_m": {m: 0 for m in M},
            "x_ij": [], "z_ijb": [], "W_jkb": [], "beta_kb": [],
            "W_kmb": [], "W_klb": [], "W_kb": [], "theta_mb": [], "theta_mn": [],
        }

    def _val(idx: int) -> float:
        v = float(x_vec[idx])
        return 0.0 if abs(v) < 1e-9 else v

    def _bin(idx: int) -> int:
        return int(round(_val(idx)))

    # --- Dönem Bazlı ve 5 Yıllık Genel Çıktı Derlemesi ---
    donemler_cikti: Dict[int, Dict[str, Any]] = {}
    toplam_gelir_kalem = {
        "Hammadde Satış Geliri": 0.0,
        "İkinci Ömür Pazar Geliri": 0.0,
    }
    toplam_maliyet_kalem = {
        "Yıllık Eşdeğer Sabit Maliyetler": 0.0,
        "Toplama Maliyeti": 0.0,
        "İşleme Maliyeti": 0.0,
        "İkinci Ömür İşlem Maliyeti": 0.0,
        "Geri Dönüşüm İşlem Maliyeti": 0.0,
        "Taşıma Maliyeti": 0.0,
        "Bertaraf Maliyeti": 0.0,
        "Stok Tutma Maliyeti": 0.0,
    }

    tum_z_rows: List[Dict[str, Any]] = []
    tum_w1_rows: List[Dict[str, Any]] = []
    tum_beta_rows: List[Dict[str, Any]] = []
    tum_w3_rows: List[Dict[str, Any]] = []
    tum_w2_rows: List[Dict[str, Any]] = []
    tum_w4_rows: List[Dict[str, Any]] = []
    tum_mu_rows: List[Dict[str, Any]] = []
    tum_delta_rows: List[Dict[str, Any]] = []
    tum_xij_set: set = set()

    for t in T:
        t_no = t + 1

        # Tesis durumları
        x1_t = {j: _bin(x1[(j, t)]) for j in J}
        x2_t = {k: _bin(x2[(k, t)]) for k in K}
        x3_t = {l: _bin(x3[(l, t)]) for l in L}
        x4_t = {m: _bin(x4[(m, t)]) for m in M}

        yeni_x1 = [j for j in J if x1_t[j] == 1 and (t == 0 or _bin(x1[(j, t - 1)]) == 0)]
        yeni_x2 = [k for k in K if x2_t[k] == 1 and (t == 0 or _bin(x2[(k, t - 1)]) == 0)]
        yeni_x3 = [l for l in L if x3_t[l] == 1 and (t == 0 or _bin(x3[(l, t - 1)]) == 0)]
        yeni_x4 = [m for m in M if x4_t[m] == 1 and (t == 0 or _bin(x4[(m, t - 1)]) == 0)]

        # Dönem gelirleri
        gelir_hammadde_t = sum(
            REV[n_idx][t] * _val(delta[(m, n_idx, t)])
            for m in M for n_idx in range(len(N))
        )
        gelir_pazar_t = sum(
            P[s][b][t] * _val(V[(l, s, b, t)])
            for l in L for s in S for b in range(2)
        )

        # Dönem maliyetleri
        mal_sabit_t = (
            sum(f[(j, t)] * x1_t[j] for j in J)
            + sum(fk[(k, t)] * x2_t[k] for k in K)
            + sum(fl[(l, t)] * x3_t[l] for l in L)
            + sum(fm[(m, t)] * x4_t[m] for m in M)
        )
        mal_toplama_t = sum(
            CJ[b][t] * _val(Z[(i, j, b, t)])
            for (i, b, t2) in aktif_ibt if t2 == t for j in J
        )
        mal_isleme_t = sum(
            CK[b][t] * _val(beta[(k, b, t)])
            for k in K for b in range(2)
        )
        mal_ikinci_omur_t = sum(
            CL1[b][t] * _val(Wrep[(l, b, t)])
            + CL2[b][t] * _val(Wrem[(l, b, t)])
            + CL3[b][t] * _val(Wreu[(l, b, t)])
            for l in L for b in range(2)
        )
        mal_geri_donusum_t = sum(
            REC[b][t] * _val(mu[(m, b, t)])
            for m in M for b in range(2)
        )
        mal_tasima_t = (
            sum(
                CT1[b][t] * d_ij[(i, j)] * _val(Z[(i, j, b, t)])
                for (i, b, t2) in aktif_ibt if t2 == t for j in J
            )
            + sum(
                CT1[b][t] * d_jk[(j, k)] * _val(W1[(j, k, b, t)])
                for j in J for k in K for b in range(2)
            )
            + sum(
                CT2[b][t] * d_kl[(k, l)] * _val(W2[(k, l, b, t)])
                for k in K for l in L for b in range(2)
            )
            + sum(
                CT2[b][t] * d_km[(k, m)] * _val(W3[(k, m, b, t)])
                for k in K for m in M for b in range(2)
            )
            + sum(
                CT2[b][t] * d_ls[(l, s)] * _val(V[(l, s, b, t)])
                for l in L for s in S for b in range(2)
            )
        )
        mal_bertaraf_t = sum(
            CB[b][t] * _val(W4[(k, b, t)])
            for k in K for b in range(2)
        )
        mal_stok_t = (
            sum(CS1[b][t] * _val(I1[(j, b, t)]) for j in J for b in range(2))
            + sum(CS1[b][t] * _val(I2[(k, b, t)]) for k in K for b in range(2))
            + sum(CS2[b][t] * _val(I3[(k, b, t)]) for k in K for b in range(2))
            + sum(CS2[b][t] * _val(I4[(l, b, t)]) for l in L for b in range(2))
        )

        gelir_t = gelir_hammadde_t + gelir_pazar_t
        maliyet_t = (
            mal_sabit_t + mal_toplama_t + mal_isleme_t + mal_ikinci_omur_t
            + mal_geri_donusum_t + mal_tasima_t + mal_bertaraf_t + mal_stok_t
        )
        net_kar_t = gelir_t - maliyet_t

        gelir_kalem_t = {
            "Hammadde Satış Geliri": round(gelir_hammadde_t, 2),
            "İkinci Ömür Pazar Geliri": round(gelir_pazar_t, 2),
        }
        maliyet_kalem_t = {
            "Yıllık Eşdeğer Sabit Maliyetler": round(mal_sabit_t, 2),
            "Toplama Maliyeti": round(mal_toplama_t, 2),
            "İşleme Maliyeti": round(mal_isleme_t, 2),
            "İkinci Ömür İşlem Maliyeti": round(mal_ikinci_omur_t, 2),
            "Geri Dönüşüm İşlem Maliyeti": round(mal_geri_donusum_t, 2),
            "Taşıma Maliyeti": round(mal_tasima_t, 2),
            "Bertaraf Maliyeti": round(mal_bertaraf_t, 2),
            "Stok Tutma Maliyeti": round(mal_stok_t, 2),
        }

        toplam_gelir_kalem["Hammadde Satış Geliri"] += gelir_hammadde_t
        toplam_gelir_kalem["İkinci Ömür Pazar Geliri"] += gelir_pazar_t
        toplam_maliyet_kalem["Yıllık Eşdeğer Sabit Maliyetler"] += mal_sabit_t
        toplam_maliyet_kalem["Toplama Maliyeti"] += mal_toplama_t
        toplam_maliyet_kalem["İşleme Maliyeti"] += mal_isleme_t
        toplam_maliyet_kalem["İkinci Ömür İşlem Maliyeti"] += mal_ikinci_omur_t
        toplam_maliyet_kalem["Geri Dönüşüm İşlem Maliyeti"] += mal_geri_donusum_t
        toplam_maliyet_kalem["Taşıma Maliyeti"] += mal_tasima_t
        toplam_maliyet_kalem["Bertaraf Maliyeti"] += mal_bertaraf_t
        toplam_maliyet_kalem["Stok Tutma Maliyeti"] += mal_stok_t

        # Dönem detay tabloları
        z_t: List[Dict[str, Any]] = []
        for (i, b, t2) in aktif_ibt:
            if t2 != t:
                continue
            for j in J:
                val = _val(Z[(i, j, b, t)])
                if val > 0.01:
                    tum_xij_set.add((i, j))
                    row = {
                        "Dönem": t_no,
                        "İl (i)": i,
                        "Toplama Merkezi (j)": j,
                        "Batarya Tipi (b)": B[b],
                        "Miktar (Adet)": round(val, 2),
                    }
                    z_t.append(row)
                    tum_z_rows.append(row)

        w1_t: List[Dict[str, Any]] = []
        for j in J:
            for k in K:
                for b in range(2):
                    val = _val(W1[(j, k, b, t)])
                    if val > 0.01:
                        row = {
                            "Dönem": t_no,
                            "Toplama Merkezi (j)": j,
                            "İşleme Merkezi (k)": k,
                            "Batarya Tipi (b)": B[b],
                            "Miktar (Adet)": round(val, 2),
                        }
                        w1_t.append(row)
                        tum_w1_rows.append(row)

        beta_t: List[Dict[str, Any]] = []
        for k in K:
            for b in range(2):
                val = _val(beta[(k, b, t)])
                if val > 0.01:
                    row = {
                        "Dönem": t_no,
                        "İşleme Merkezi (k)": k,
                        "Batarya Tipi (b)": B[b],
                        "Toplam Batarya (Adet)": round(val, 2),
                    }
                    beta_t.append(row)
                    tum_beta_rows.append(row)

        w2_t: List[Dict[str, Any]] = []
        for k in K:
            for l in L:
                for b in range(2):
                    val = _val(W2[(k, l, b, t)])
                    if val > 0.01:
                        row = {
                            "Dönem": t_no,
                            "İşleme Merkezi (k)": k,
                            "İkinci Ömür Merkezi (l)": l,
                            "Batarya Tipi (b)": B[b],
                            "Toplam Hücre": round(val, 2),
                            "Wrep (Farklı Amaç)": round(_val(Wrep[(l, b, t)]), 2),
                            "Wrem (Yeniden İmalat)": round(_val(Wrem[(l, b, t)]), 2),
                            "Wreu (Doğrudan Kullanım)": round(_val(Wreu[(l, b, t)]), 2),
                        }
                        w2_t.append(row)
                        tum_w2_rows.append(row)

        w3_t: List[Dict[str, Any]] = []
        for k in K:
            for m in M:
                for b in range(2):
                    val = _val(W3[(k, m, b, t)])
                    if val > 0.01:
                        row = {
                            "Dönem": t_no,
                            "İşleme Merkezi (k)": k,
                            "Geri Dönüşüm Merkezi (m)": m,
                            "Batarya Tipi (b)": B[b],
                            "Hücre Sayısı": round(val, 2),
                        }
                        w3_t.append(row)
                        tum_w3_rows.append(row)

        w4_t: List[Dict[str, Any]] = []
        for k in K:
            for b in range(2):
                val = _val(W4[(k, b, t)])
                if val > 0.01:
                    row = {
                        "Dönem": t_no,
                        "İşleme Merkezi (k)": k,
                        "Batarya Tipi (b)": B[b],
                        "Bertarafa Giden Hücre": round(val, 2),
                    }
                    w4_t.append(row)
                    tum_w4_rows.append(row)

        v_t: List[Dict[str, Any]] = []
        for l in L:
            for s in S:
                for b in range(2):
                    val = _val(V[(l, s, b, t)])
                    if val > 0.01:
                        v_t.append(
                            {
                                "Dönem": t_no,
                                "İkinci Ömür Merkezi (l)": l,
                                "Pazar (s)": pazar_adlari[s],
                                "Batarya Tipi (b)": B[b],
                                "Satılan Hücre": round(val, 2),
                                "Satış Geliri ($)": round(val * P[s][b][t], 2),
                            }
                        )

        i1_t = [
            {"Dönem": t_no, "Toplama Merkezi (j)": j, "Batarya Tipi (b)": B[b], "Stok (Adet)": round(_val(I1[(j, b, t)]), 2)}
            for j in J for b in range(2) if _val(I1[(j, b, t)]) > 0.01
        ]
        i2_t = [
            {"Dönem": t_no, "İşleme Merkezi (k)": k, "Batarya Tipi (b)": B[b], "Stok (Adet)": round(_val(I2[(k, b, t)]), 2)}
            for k in K for b in range(2) if _val(I2[(k, b, t)]) > 0.01
        ]
        i3_t = [
            {"Dönem": t_no, "İşleme Merkezi (k)": k, "Batarya Tipi (b)": B[b], "Stok (Hücre)": round(_val(I3[(k, b, t)]), 2)}
            for k in K for b in range(2) if _val(I3[(k, b, t)]) > 0.01
        ]
        i4_t = [
            {"Dönem": t_no, "İkinci Ömür Merkezi (l)": l, "Batarya Tipi (b)": B[b], "Stok (Hücre)": round(_val(I4[(l, b, t)]), 2)}
            for l in L for b in range(2) if _val(I4[(l, b, t)]) > 0.01
        ]

        theta_t = [
            {"Dönem": t_no, "İkinci Ömür Merkezi (l)": l, "Batarya Tipi (b)": B[b], "İşlenen Ağırlık (kg)": round(_val(theta[(l, b, t)]), 2)}
            for l in L for b in range(2) if _val(theta[(l, b, t)]) > 0.01
        ]
        mu_t: List[Dict[str, Any]] = []
        for m in M:
            for b in range(2):
                val = _val(mu[(m, b, t)])
                if val > 0.01:
                    row = {
                        "Dönem": t_no,
                        "Geri Dönüşüm Merkezi (m)": m,
                        "Batarya Tipi (b)": B[b],
                        "Hücre Ağırlığı (kg)": round(val, 2),
                    }
                    mu_t.append(row)
                    tum_mu_rows.append(row)

        delta_t: List[Dict[str, Any]] = []
        for m in M:
            for n_idx, n_ad in enumerate(N):
                val = _val(delta[(m, n_idx, t)])
                if val > 0.01:
                    row = {
                        "Dönem": t_no,
                        "Geri Dönüşüm Merkezi (m)": m,
                        "Materyal (n)": n_ad,
                        "Geri Kazanılan Miktar (kg)": round(val, 2),
                        "Satış Geliri ($)": round(val * REV[n_idx][t], 2),
                    }
                    delta_t.append(row)
                    tum_delta_rows.append(row)

        donemler_cikti[t_no] = {
            "donem": t_no,
            "net_kar": round(net_kar_t, 2),
            "gelir": round(gelir_t, 2),
            "maliyet": round(maliyet_t, 2),
            "gelir_kalemleri": gelir_kalem_t,
            "maliyet_kalemleri": maliyet_kalem_t,
            "x1": x1_t,
            "x2": x2_t,
            "x3": x3_t,
            "x4": x4_t,
            "aktif_tesisler": {
                "x1": [j for j, v in x1_t.items() if v == 1],
                "x2": [k for k, v in x2_t.items() if v == 1],
                "x3": [l for l, v in x3_t.items() if v == 1],
                "x4": [m for m, v in x4_t.items() if v == 1],
            },
            "yeni_acilan_tesisler": {
                "x1": yeni_x1,
                "x2": yeni_x2,
                "x3": yeni_x3,
                "x4": yeni_x4,
            },
            "Z": z_t,
            "W1": w1_t,
            "W2": w2_t,
            "W3": w3_t,
            "W4": w4_t,
            "beta": beta_t,
            "V": v_t,
            "I1": i1_t,
            "I2": i2_t,
            "I3": i3_t,
            "I4": i4_t,
            "theta": theta_t,
            "mu": mu_t,
            "delta": delta_t,
        }

    toplam_gelir = sum(toplam_gelir_kalem.values())
    toplam_maliyet = sum(toplam_maliyet_kalem.values())
    toplam_net_kar = toplam_gelir - toplam_maliyet

    # 5 dönem sonundaki aktif tesisler (monotonluk nedeniyle 5 yılda açılan tüm tesisler)
    x_j_son = dict(donemler_cikti[5]["x1"])
    x_k_son = dict(donemler_cikti[5]["x2"])
    x_l_son = dict(donemler_cikti[5]["x3"])
    x_m_son = dict(donemler_cikti[5]["x4"])

    genel_ozet = {
        "donem_sayisi": len(T),
        "toplam_net_kar": round(toplam_net_kar, 2),
        "toplam_gelir": round(toplam_gelir, 2),
        "toplam_maliyet": round(toplam_maliyet, 2),
        "gelir_kalemleri": {k: round(v, 2) for k, v in toplam_gelir_kalem.items()},
        "maliyet_kalemleri": {k: round(v, 2) for k, v in toplam_maliyet_kalem.items()},
        "acilan_tesisler": {
            "Toplama (x1)": [j for j, v in x_j_son.items() if v == 1],
            "İşleme (x2)": [k for k, v in x_k_son.items() if v == 1],
            "İkinci Ömür (x3)": [l for l, v in x_l_son.items() if v == 1],
            "Geri Dönüşüm (x4)": [m for m, v in x_m_son.items() if v == 1],
        },
    }

    # Mevcut Modül 2 grafiği ve tablosuyla uyumlu birleşik gelir(+)/maliyet(-) kırılımı
    birlesik_kalemler = {
        "Hammadde Satış Geliri (+)": round(toplam_gelir_kalem["Hammadde Satış Geliri"], 2),
        "İkinci Ömür Pazar Geliri (+)": round(toplam_gelir_kalem["İkinci Ömür Pazar Geliri"], 2),
        "Yıllık Eşdeğer Sabit Maliyetler (-)": -round(toplam_maliyet_kalem["Yıllık Eşdeğer Sabit Maliyetler"], 2),
        "Toplama Maliyeti (-)": -round(toplam_maliyet_kalem["Toplama Maliyeti"], 2),
        "İşleme Maliyeti (-)": -round(toplam_maliyet_kalem["İşleme Maliyeti"], 2),
        "İkinci Ömür İşlem Maliyeti (-)": -round(toplam_maliyet_kalem["İkinci Ömür İşlem Maliyeti"], 2),
        "Geri Dönüşüm İşlem Maliyeti (-)": -round(toplam_maliyet_kalem["Geri Dönüşüm İşlem Maliyeti"], 2),
        "Taşıma Maliyeti (-)": -round(toplam_maliyet_kalem["Taşıma Maliyeti"], 2),
        "Bertaraf Maliyeti (-)": -round(toplam_maliyet_kalem["Bertaraf Maliyeti"], 2),
        "Stok Tutma Maliyeti (-)": -round(toplam_maliyet_kalem["Stok Tutma Maliyeti"], 2),
    }

    return {
        "durum": "Optimal",
        "cozucu": cozucu_adi,
        "amac_tipi": "Maksimizasyon (Toplam Net Kâr)",
        "amac_degeri": round(toplam_net_kar, 2),
        "toplam_net_kar": round(toplam_net_kar, 2),
        "toplam_gelir": round(toplam_gelir, 2),
        "toplam_maliyet": round(toplam_maliyet, 2),
        "genel_ozet": genel_ozet,
        "donemler": donemler_cikti,
        "maliyet_kalemleri": birlesik_kalemler,
        "x_j": x_j_son,
        "x_k": x_k_son,
        "x_l": x_l_son,
        "x_m": x_m_son,
        "x_ij": sorted(tum_xij_set),
        "z_ijb": tum_z_rows,
        "W_jkb": tum_w1_rows,
        "beta_kb": tum_beta_rows,
        "W_kmb": tum_w3_rows,
        "W_klb": tum_w2_rows,
        "W_kb": tum_w4_rows,
        "theta_mb": tum_mu_rows,
        "theta_mn": tum_delta_rows,
    }

