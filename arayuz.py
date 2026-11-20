import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import pulp
import importlib
import veri_hesaplama as _veri_hesaplama
if not hasattr(_veri_hesaplama, "milp_5_donem_coz"):
    _veri_hesaplama = importlib.reload(_veri_hesaplama)
from veri_hesaplama import (
    IL_KOORDINATLARI,
    dinamik_kmeans_ve_dirsek,
    dinamik_minimax_p_center,
    dinamik_pareto_analizi,
    haversine_km,
    il_adini_normalize_et,
    milp_5_donem_coz,
    tum_zinciri_calistir,
    veriyi_oku_ve_temizle,
)

# Sayfa ayarları
st.set_page_config(page_title="EV Batarya Tersine Lojistik KDS", layout="wide")

# ==========================================
# GÖRSEL TASARIM (GLOBAL TEMA CSS)
# ==========================================
def apply_global_theme():
    global_css = """
    <style>
    /* Genel sayfa ve üst çubuk zemini (Açık gri-beyaz kurumsal tema) */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FA !important;
        color: #1E293B !important;
    }
    [data-testid="stHeader"] {
        background-color: #F5F7FA !important;
    }

    /* Ana içerik başlık ve etiket kontrastı (sidebar hariç) */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    .main h1,
    .main h2,
    .main h3,
    .main h4 {
        color: #1E293B !important;
    }
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    .main label {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    [data-testid="stMain"] hr,
    .main hr {
        border-color: #E2E8F0 !important;
    }

    /* Birincil (Primary) butonlar: Kurumsal Mavi + Beyaz Yazı */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primary"],
    div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"] {
        background-color: #2B6CB0 !important;
        color: #FFFFFF !important;
        border: 1px solid #245B93 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(43, 108, 176, 0.18) !important;
        transition: background-color 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] p,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] p,
    div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primary"] p,
    div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"] p {
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"]:hover {
        background-color: #245B93 !important;
        border-color: #1A3E62 !important;
        box-shadow: 0 4px 12px rgba(36, 91, 147, 0.26) !important;
    }

    /* İkincil (Secondary) butonlar: Beyaz Zemin + Koyu Lacivert/Mavi Çerçeve ve Yazı */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"],
    div[data-testid="stFormSubmitButton"] > button[kind="secondary"],
    div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-secondary"],
    div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #2B6CB0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(30, 41, 59, 0.07) !important;
        transition: all 0.18s ease !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] p {
        color: #1E293B !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover {
        background-color: #EBF4FF !important;
        color: #1E3A8A !important;
        border-color: #1E40AF !important;
        box-shadow: 0 4px 10px rgba(43, 108, 176, 0.15) !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover p {
        color: #1E3A8A !important;
    }

    /* Form giriş kutuları, sayısal alanlar, selectbox ve multiselect */
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #1E293B !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: #1E293B !important;
    }
    div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="select"]:focus-within > div {
        border-color: #2B6CB0 !important;
        box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.18) !important;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border-color: #E2E8F0 !important;
    }
    /* Multiselect etiketleri: Kırmızı yerine Kurumsal Mavi */
    span[data-baseweb="tag"] {
        background-color: #2B6CB0 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] span,
    span[data-baseweb="tag"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Slider bileşenleri: Kurumsal Mavi */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2B6CB0 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 1px 4px rgba(30, 41, 59, 0.25) !important;
    }
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
        color: #2B6CB0 !important;
        font-weight: 600 !important;
    }

    /* Dosya yükleme alanı (st.file_uploader) */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #94A3B8 !important;
        border-radius: 12px !important;
        color: #1E293B !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span {
        color: #475569 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #F8FAFC !important;
        color: #2B6CB0 !important;
        border: 1px solid #2B6CB0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #EBF4FF !important;
    }

    /* Açılır paneller (st.expander) */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #2B6CB0 !important;
    }

    /* Sekmeler (st.tabs): Kurumsal Mavi aktif vurgu */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2B6CB0 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2B6CB0 !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #E2E8F0 !important;
    }

    /* Metrikler (st.metric) ve Tablolar (st.dataframe) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1E293B !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Bilgi ve durum kutuları (st.info / st.success / st.warning / st.error) */
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
        background-color: #EBF4FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 10px !important;
        color: #1E40AF !important;
    }
    [data-testid="stAlertContentInfo"],
    [data-testid="stAlertContentInfo"] p {
        color: #1E40AF !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
        background-color: #ECFDF5 !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 10px !important;
        color: #065F46 !important;
    }
    [data-testid="stAlertContentSuccess"],
    [data-testid="stAlertContentSuccess"] p {
        color: #065F46 !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
        background-color: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-radius: 10px !important;
        color: #92400E !important;
    }
    [data-testid="stAlertContentWarning"],
    [data-testid="stAlertContentWarning"] p {
        color: #92400E !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
        background-color: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-radius: 10px !important;
        color: #991B1B !important;
    }
    [data-testid="stAlertContentError"],
    [data-testid="stAlertContentError"] p {
        color: #991B1B !important;
    }
    </style>
    """
    st.markdown(global_css, unsafe_allow_html=True)

apply_global_theme()

# ==========================================
# VARSAYILAN ÖRNEK EV VERİSİ
# ==========================================
VARSAYILAN_EV_VERISI = {
    "İstanbul": 134054, "Ankara": 47965, "İzmir": 26992, "Antalya": 17436, "Bursa": 16673,
    "Kocaeli": 8927, "Konya": 7210, "Adana": 5877, "Kayseri": 5657, "Gaziantep": 5539,
    "Mersin": 5507, "Muğla": 5129, "Denizli": 3899, "Sakarya": 3727, "Samsun": 3711,
    "Eskişehir": 3262, "Aydın": 3203, "Balıkesir": 3142, "Tekirdağ": 3089, "Manisa": 2851,
    "Trabzon": 2426, "Hatay": 1954, "Kahramanmaraş": 1703, "Diyarbakır": 1667, "Afyonkarahisar": 1538,
    "Isparta": 1476, "Çanakkale": 1468, "Elazığ": 1442, "Yalova": 1228, "Sivas": 1088,
    "Ordu": 1086, "Kütahya": 1071, "Erzurum": 1046, "Malatya": 1039, "Şanlıurfa": 1016,
    "Düzce": 888, "Edirne": 874, "Bolu": 868, "Zonguldak": 815, "Uşak": 778,
    "Rize": 694, "Nevşehir": 691, "Aksaray": 681, "Kırklareli": 642, "Çorum": 613,
    "Giresun": 613, "Tokat": 592, "Osmaniye": 566, "Burdur": 538, "Van": 516,
    "Amasya": 475, "Mardin": 460, "Bilecik": 422, "Karabük": 388, "Karaman": 379,
    "Batman": 371, "Yozgat": 370, "Niğde": 347, "Kastamonu": 333, "Adıyaman": 310,
    "Kırıkkale": 284, "Kırşehir": 275, "Erzincan": 256, "Çankırı": 213, "Bartın": 201,
    "Sinop": 201, "Artvin": 154, "Bingöl": 129, "Gümüşhane": 110, "Siirt": 105,
    "Bayburt": 94, "Kilis": 85, "Şırnak": 82, "Kars": 81, "Bitlis": 80,
    "Muş": 66, "Ağrı": 65, "Iğdır": 65, "Tunceli": 54, "Ardahan": 38, "Hakkari": 14
}

# ==========================================
# 1. SESSION STATE YÖNETİMİ
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'mod1_data' not in st.session_state:
    st.session_state.mod1_data = None
if 'mod1_step' not in st.session_state:
    st.session_state.mod1_step = 0
if 'senaryo_gecmisi' not in st.session_state:
    st.session_state.senaryo_gecmisi = []

def _hesaplama_durumunu_temizle():
    for k in [
        'pareto_sonuclari', 'pareto_meta', 'kmeans_sonuclari', 'kmeans_calisti',
        'optimum_k', 'wcss', 'k_degerleri', 'minimax_sonuclari', 'milp_sonuc'
    ]:
        if k in st.session_state:
            del st.session_state[k]

def reset_system():
    st.session_state.mod1_data = None
    st.session_state.mod1_step = 0
    st.session_state.senaryo_gecmisi = []
    _hesaplama_durumunu_temizle()
    st.success("Sistem hafızası sıfırlandı. Yeni veri yüklenebilir.")

def go_to_dashboard():
    st.session_state.current_view = 'dashboard'

# ==========================================
# 1.5 KULLANICI VERİTABANI VE GÜVENLİK (SQLite)
# ==========================================
KULLANICI_DB_YOLU = Path(__file__).resolve().parent / "kullanicilar.db"

def _sifre_hashle(sifre: str) -> str:
    salt = secrets.token_hex(16)
    hash_hex = hashlib.pbkdf2_hmac(
        "sha256",
        sifre.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt}${hash_hex}"

def _sifre_dogrula(sifre: str, kayitli_hash: str) -> bool:
    if not kayitli_hash or "$" not in kayitli_hash:
        return False
    salt, beklenen_hash = kayitli_hash.split("$", 1)
    hesaplanan_hash = hashlib.pbkdf2_hmac(
        "sha256",
        sifre.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return hmac.compare_digest(hesaplanan_hash, beklenen_hash)

def kullanici_veritabanini_hazirla():
    with sqlite3.connect(KULLANICI_DB_YOLU) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kullanicilar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_soyad TEXT NOT NULL,
                kullanici_adi TEXT NOT NULL UNIQUE,
                sifre_hash TEXT NOT NULL,
                kayit_tarihi TEXT NOT NULL
            )
            """
        )
        mevcut_admin = conn.execute(
            "SELECT 1 FROM kullanicilar WHERE LOWER(kullanici_adi) = LOWER(?)",
            ("admin",),
        ).fetchone()
        if mevcut_admin is None:
            conn.execute(
                """
                INSERT INTO kullanicilar (ad_soyad, kullanici_adi, sifre_hash, kayit_tarihi)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "Sistem Yöneticisi",
                    "admin",
                    _sifre_hashle("proje2026"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        conn.commit()

def kullanici_kaydet(ad_soyad: str, kullanici_adi: str, sifre: str):
    temiz_ad = ad_soyad.strip()
    temiz_kadi = kullanici_adi.strip()
    try:
        with sqlite3.connect(KULLANICI_DB_YOLU) as conn:
            mevcut = conn.execute(
                "SELECT 1 FROM kullanicilar WHERE LOWER(kullanici_adi) = LOWER(?)",
                (temiz_kadi,),
            ).fetchone()
            if mevcut is not None:
                return False, "Bu kullanıcı adı zaten kayıtlı."

            conn.execute(
                """
                INSERT INTO kullanicilar (ad_soyad, kullanici_adi, sifre_hash, kayit_tarihi)
                VALUES (?, ?, ?, ?)
                """,
                (
                    temiz_ad,
                    temiz_kadi,
                    _sifre_hashle(sifre),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
        return True, "Kayıt başarıyla tamamlandı! 'Giriş Yap' sekmesinden hemen giriş yapabilirsiniz."
    except sqlite3.IntegrityError:
        return False, "Bu kullanıcı adı zaten kayıtlı."
    except Exception as e:
        return False, f"Kayıt sırasında veritabanı hatası oluştu: {e}"

def kullanici_dogrula(kullanici_adi: str, sifre: str):
    temiz_kadi = kullanici_adi.strip()
    with sqlite3.connect(KULLANICI_DB_YOLU) as conn:
        satir = conn.execute(
            """
            SELECT id, ad_soyad, kullanici_adi, sifre_hash
            FROM kullanicilar
            WHERE LOWER(kullanici_adi) = LOWER(?)
            """,
            (temiz_kadi,),
        ).fetchone()
    if satir is None:
        return None
    if _sifre_dogrula(sifre, satir[3]):
        return {"id": satir[0], "ad_soyad": satir[1], "kullanici_adi": satir[2]}
    return None

# ==========================================
# 2. GİRİŞ EKRANI (Login & Kayıt Ol)
# ==========================================
def show_login():
    try:
        kullanici_veritabanini_hazirla()
    except Exception as e:
        st.error(f"Kullanıcı veritabanı hazırlanırken hata oluştu: {e}")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            padding: 20px 24px 24px 24px !important;
            box-shadow: 0 10px 28px rgba(30, 41, 59, 0.08) !important;
        }
        [data-testid="stForm"] {
            border: none !important;
            padding: 8px 0 0 0 !important;
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='text-align: center; color: #1E293B; font-weight: 800; letter-spacing: -0.4px; margin-bottom: 8px;'>EV Batarya Tersine Lojistik Ağı KDS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #475569; font-size: 16px; margin-bottom: 28px;'>Sisteme Giriş Yapınız veya Yeni Hesap Oluşturunuz</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.25, 1])
    with col2:
        with st.container(border=True):
            tab_giris, tab_kayit = st.tabs(["Giriş Yap", "Kayıt Ol"])

            with tab_giris:
                with st.form("login_form"):
                    username = st.text_input("Kullanıcı Adı")
                    password = st.text_input("Şifre", type="password")
                    submit = st.form_submit_button("Giriş Yap", type="primary", use_container_width=True)

                    if submit:
                        if not username.strip() or not password:
                            st.warning("Lütfen kullanıcı adı ve şifre alanlarını doldurunuz.")
                        else:
                            try:
                                kullanici = kullanici_dogrula(username, password)
                                if kullanici is not None:
                                    st.session_state.logged_in = True
                                    st.session_state.current_view = 'dashboard'
                                    st.session_state.aktif_kullanici = kullanici
                                    st.rerun()
                                else:
                                    st.error("Hatalı kullanıcı adı veya şifre!")
                            except Exception as e:
                                st.error(f"Giriş doğrulaması sırasında hata oluştu: {e}")

            with tab_kayit:
                with st.form("register_form"):
                    reg_ad_soyad = st.text_input("Ad Soyad")
                    reg_username = st.text_input("Kullanıcı Adı", key="reg_username_input")
                    reg_password = st.text_input("Şifre", type="password", key="reg_password_input")
                    reg_password_confirm = st.text_input("Şifre Tekrarı", type="password", key="reg_password_confirm_input")
                    reg_submit = st.form_submit_button("Kayıt Ol", type="primary", use_container_width=True)

                    if reg_submit:
                        if (
                            not reg_ad_soyad.strip()
                            or not reg_username.strip()
                            or not reg_password
                            or not reg_password_confirm
                        ):
                            st.warning("Lütfen tüm alanları doldurunuz.")
                        elif len(reg_password) < 6:
                            st.warning("Şifre en az 6 karakter olmalıdır.")
                        elif reg_password != reg_password_confirm:
                            st.error("Girilen şifreler birbiriyle eşleşmiyor!")
                        else:
                            basarili, mesaj = kullanici_kaydet(reg_ad_soyad, reg_username, reg_password)
                            if basarili:
                                st.success(mesaj)
                            else:
                                st.error(mesaj)

# ==========================================
# 2.5 SOL YAN SÜREÇ TAKİP BÖLMESİ (Sidebar)
# ==========================================
def render_progress_sidebar():
    view = st.session_state.get('current_view', 'dashboard')
    mod1_step = st.session_state.get('mod1_step', 0)
    has_milp_result = 'milp_sonuc' in st.session_state and st.session_state.milp_sonuc is not None

    aktif_adimlar = set()
    if view == 'dashboard':
        aktif_adimlar.add('login')
    elif view == 'module1':
        if mod1_step <= 0:
            aktif_adimlar.add('veri_yukleme')
        elif mod1_step == 1:
            aktif_adimlar.update({'aday_tesis', 'pareto'})
        elif mod1_step == 2:
            aktif_adimlar.update({'aday_tesis', 'kmeans'})
        else:
            aktif_adimlar.update({'aday_tesis', 'minimax'})
    elif view == 'module2':
        if has_milp_result:
            aktif_adimlar.add('sonuclar')
        else:
            aktif_adimlar.add('senaryo')

    def _item_html(key_name, label, is_sub=False):
        aktif = key_name in aktif_adimlar
        dot_cls = "kds-dot is-sub" if is_sub else "kds-dot"
        item_cls = "kds-substep-item" if is_sub else "kds-step-item"
        if aktif:
            dot_cls += " is-active"
            item_cls += " is-active"
        return f"<div class='{item_cls}'><span class='{dot_cls}'></span><span>{label}</span></div>"

    sidebar_html = f"""
    <style>
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #3B82C4 0%, #245B93 100%) !important;
        border-right: 1px solid #154360 !important;
    }}
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
        color: #FFFFFF !important;
    }}
    .kds-brand-card {{
        background-color: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 20px;
    }}
    section[data-testid="stSidebar"] .kds-brand-subtitle {{
        font-size: 11px;
        letter-spacing: 1.5px;
        color: rgba(255, 255, 255, 0.78) !important;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
    }}
    section[data-testid="stSidebar"] .kds-brand-title {{
        font-size: 17px;
        font-weight: 700;
        color: #FFFFFF !important;
        line-height: 1.3;
    }}
    .kds-sidebar-title {{
        font-size: 1.02rem;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: #FFFFFF !important;
        padding-bottom: 12px;
        margin-bottom: 18px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.28);
    }}
    .kds-step-list {{
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding-left: 2px;
    }}
    .kds-step-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.98rem;
        font-weight: 600;
        line-height: 1.35;
        color: #FFFFFF !important;
    }}
    .kds-step-item.is-active {{
        font-weight: 800;
    }}
    .kds-substep-list {{
        display: flex;
        flex-direction: column;
        gap: 11px;
        margin-top: -5px;
        margin-bottom: 2px;
        margin-left: 6px;
        padding-left: 18px;
        border-left: 1.5px solid rgba(255, 255, 255, 0.30);
    }}
    .kds-substep-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.89rem;
        font-weight: 500;
        line-height: 1.35;
        color: rgba(255, 255, 255, 0.94) !important;
    }}
    .kds-substep-item.is-active {{
        font-weight: 700;
        color: #FFFFFF !important;
    }}
    .kds-dot {{
        flex-shrink: 0;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background-color: #2D3748;
        border: 1.5px solid #CBD5E1;
        box-sizing: border-box;
    }}
    .kds-dot.is-sub {{
        width: 11px;
        height: 11px;
    }}
    .kds-dot.is-active {{
        background-color: #E53935;
        border: 2px solid #FFFFFF;
        box-shadow: 0 0 8px rgba(229, 57, 53, 0.85);
    }}
    </style>
    <div class="kds-brand-card">
        <div class="kds-brand-subtitle">KARAR DESTEK SİSTEMİ</div>
        <div class="kds-brand-title">EA Batarya Tersine Lojistik Ağı</div>
    </div>
    <div class="kds-sidebar-title">Süreç Takibi</div>
    <div class="kds-step-list">
        {_item_html('login', 'Login')}
        {_item_html('veri_yukleme', 'Veri Yükleme')}
        {_item_html('aday_tesis', 'Aday Tesis Belirleme')}
        <div class="kds-substep-list">
            {_item_html('pareto', 'Pareto Analizi', is_sub=True)}
            {_item_html('kmeans', 'K-Means Algoritması ve Dirsek Metodu', is_sub=True)}
            {_item_html('minimax', 'Minimax', is_sub=True)}
        </div>
        {_item_html('senaryo', 'Senaryo Analizi')}
        {_item_html('sonuclar', 'Sonuçlar')}
    </div>
    """
    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

# ==========================================
# 3. ANA PANEL (Dashboard)
# ==========================================
def show_dashboard():
    dashboard_css = """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FA !important;
        color: #1E293B !important;
    }
    [data-testid="stHeader"] {
        background-color: #F5F7FA !important;
    }
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 1180px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    /* Modül 1 & Modül 2: 3D Dikey Dikdörtgen Kurumsal Mavi Kartlar */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        width: 390px !important;
        max-width: 100% !important;
        min-height: 400px !important;
        padding: 36px 32px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        background: linear-gradient(180deg, #3B82C4 0%, #245B93 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-bottom: 6px solid #154360 !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 24px rgba(21, 67, 96, 0.24), 0 4px 10px rgba(21, 67, 96, 0.14) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, border-bottom-width 0.1s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(180deg, #448DD1 0%, #2966A3 100%) !important;
        transform: translateY(-5px) !important;
        border-bottom: 6px solid #123850 !important;
        box-shadow: 0 20px 34px rgba(21, 67, 96, 0.32), 0 8px 14px rgba(21, 67, 96, 0.18) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:active,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:active,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:active {
        transform: translateY(3px) !important;
        border-bottom: 2px solid #154360 !important;
        box-shadow: 0 4px 10px rgba(21, 67, 96, 0.22) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] [data-testid="stMarkdownContainer"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        text-align: center !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] p {
        margin: 0 !important;
        font-size: 0.96rem !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        color: rgba(255, 255, 255, 0.92) !important;
        white-space: normal !important;
        text-align: center !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] strong,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] strong,
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] strong {
        display: block !important;
        font-size: 1.28rem !important;
        font-weight: 800 !important;
        line-height: 1.35 !important;
        letter-spacing: 0.4px !important;
        color: #FFFFFF !important;
        margin-bottom: 18px !important;
    }

    /* Güvenli Çıkış Yap Butonu */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 2px solid #334155 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(30, 41, 59, 0.08) !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"] p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] p {
        color: #1E293B !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #1E293B !important;
        box-shadow: 0 6px 14px rgba(30, 41, 59, 0.18) !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover p,
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover p {
        color: #FFFFFF !important;
    }
    </style>
    """
    st.markdown(dashboard_css, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #1E293B; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 12px;'>Sistem Kontrol Paneli</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #475569; font-size: 17px; max-width: 780px; margin: 0 auto 36px auto; line-height: 1.65;'>
    Bu Karar Destek Sistemi, ömrünü tamamlamış elektrikli araç bataryalarının Türkiye genelinde toplanması ve geri kazanılması için tersine lojistik ağını tasarlamak amacıyla geliştirilmiştir.
    </p>
    """, unsafe_allow_html=True)

    _, col1, col2, _ = st.columns([0.5, 2.6, 2.6, 0.5], gap="large")
    with col1:
        if st.button("**VERİ GİRİŞİ VE ADAY MERKEZ BELİRLEME**\n\nPareto, K-Means ve Minimax algoritmaları ile aday tesis konumlarını belirleyin.", key="btn_mod1", type="primary"):
            st.session_state.current_view = 'module1'
            st.rerun()
    with col2:
        if st.button("**AĞ OPTİMİZASYONU VE SENARYO ANALİZİ**\n\nParametreleri değiştirerek MILP modelini çalıştırın ve tüm karar değişkenlerini analiz edin.", key="btn_mod2", type="primary"):
            st.session_state.current_view = 'module2'
            st.rerun()

    st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
    _, col_logout, _ = st.columns([4, 2.5, 4])
    with col_logout:
        st.button("Güvenli Çıkış Yap", on_click=lambda: st.session_state.update({'logged_in': False, 'current_view': 'login'}), use_container_width=True)

# ==========================================
# 4. MODÜL 1 İÇERİĞİ
# ==========================================
def show_module1():
    st.header("Modül 1: Aday Toplama Merkezi Belirleme")
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    st.divider()
    
    if st.session_state.mod1_step >= 0:
        st.subheader("Adım 1: EV Verisi Yükleme")
        st.write("Sütun başlıkları 'İl' ve 'EV_Sayısı' olan Excel dosyasını yükleyin (veya boş bırakıp varsayılan 81 il verisini kullanın):")
        col_up, col_def = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("Excel Yükle", type=['xlsx', 'xls'], label_visibility="collapsed")
        with col_def:
            if st.button("Varsayılan 81 İl Verisini Kullan", use_container_width=True):
                df_def = pd.DataFrame(list(VARSAYILAN_EV_VERISI.items()), columns=['İl', 'EV_Sayısı'])
                temiz_df, _ = veriyi_oku_ve_temizle(df_def)
                _hesaplama_durumunu_temizle()
                st.session_state.mod1_data = temiz_df
                st.session_state._son_yuklenen_dosya = None
                st.session_state.mod1_step = 1
                st.rerun()

        if uploaded_file is not None:
            try:
                dosya_imza = (uploaded_file.name, uploaded_file.size)
                df_ham = pd.read_excel(uploaded_file)
                temiz_df, uyarilar = veriyi_oku_ve_temizle(df_ham)
                for uyari in uyarilar:
                    st.warning(uyari)
                if st.session_state.get('_son_yuklenen_dosya') != dosya_imza:
                    _hesaplama_durumunu_temizle()
                    st.session_state._son_yuklenen_dosya = dosya_imza
                    st.session_state.mod1_step = 0
                st.session_state.mod1_data = temiz_df
                st.success(f"Veri başarıyla işlendi. Toplam {len(temiz_df)} geçerli il tespit edildi.")
                if st.button("Pareto Analizini Başlat", type="primary"):
                    _hesaplama_durumunu_temizle()
                    st.session_state.mod1_step = 1
                    st.rerun()
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")

    if st.session_state.mod1_step >= 1 and st.session_state.mod1_data is not None:
        st.divider()
        st.subheader("Adım 2: Pareto Analizi Sonuçları")
        try:
            pareto_sonuc = dinamik_pareto_analizi(st.session_state.mod1_data, esik=80.0)
            pareto_df = pareto_sonuc["pareto_df"]
            st.session_state.pareto_sonuclari = pareto_df
            st.session_state.pareto_meta = pareto_sonuc
            st.info(
                f"%{pareto_sonuc['esik']:.0f} hedefine en yakın kümülatif talep (%{pareto_sonuc['ulasilan_yuzde']:.2f}) "
                f"ile en yüksek yoğunluklu {pareto_sonuc['secilen_il_sayisi']} il, Aday Toplama Merkezi (j) olarak belirlendi."
            )

            m_pareto = folium.Map(location=[39.0, 35.0], zoom_start=5.5)
            for index, row in pareto_df.iterrows():
                il_adi = il_adini_normalize_et(row['İl'])
                if il_adi in IL_KOORDINATLARI:
                    lat, lon = IL_KOORDINATLARI[il_adi]
                    tooltip_html = f"<b>Tesis:</b> Aday Toplama Merkezi<br><b>İl:</b> {il_adi}<br><b>EV:</b> {row['EV_Sayısı']:,.0f}"
                    folium.Marker([lat, lon], tooltip=tooltip_html, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m_pareto)

            st_folium(m_pareto, width=900, height=420, key="map_pareto")

            if st.button("K-Means Kümeleme Algoritmasını Çalıştır", type="primary"):
                st.session_state.mod1_step = 2
                st.session_state.kmeans_calisti = False
                if 'minimax_sonuclari' in st.session_state:
                    del st.session_state['minimax_sonuclari']
                st.rerun()
        except Exception as e:
            st.error(f"Pareto analizi sırasında hata oluştu: {e}")

    if st.session_state.mod1_step >= 2:
        st.divider()
        st.subheader("Adım 3: Dirsek Metodu ve K-Means Kümeleme")
        if not st.session_state.get('kmeans_calisti', False):
            try:
                pareto_df = st.session_state.pareto_sonuclari
                kmeans_res = dinamik_kmeans_ve_dirsek(pareto_df, random_state=42)

                st.session_state.optimum_k = kmeans_res["optimum_k"]
                st.session_state.wcss = kmeans_res["wcss"]
                st.session_state.k_degerleri = kmeans_res["k_degerleri"]
                st.session_state.kmeans_sonuclari = kmeans_res["aday_isleme_merkezleri"]
                st.session_state.kmeans_calisti = True
                if 'minimax_sonuclari' in st.session_state:
                    del st.session_state['minimax_sonuclari']
            except Exception as e:
                st.error(f"K-Means kümeleme sırasında hata oluştu: {e}")

        if st.session_state.get('kmeans_calisti', False):
            opt_k = st.session_state.optimum_k
            col_grafik, col_table, _ = st.columns([1.35, 0.95, 0.7], gap="medium")
            with col_grafik:
                fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=100)
                ax.plot(
                    st.session_state.k_degerleri,
                    st.session_state.wcss,
                    marker='o',
                    markersize=5,
                    color='#2F70AF',
                    linewidth=2,
                )
                opt_idx = st.session_state.k_degerleri.index(opt_k)
                opt_wcss = st.session_state.wcss[opt_idx]
                ax.plot(opt_k, opt_wcss, marker='o', markersize=8, color='red')
                ax.set_title(f"Dirsek Grafiği (Optimum k={opt_k})", fontsize=11, fontweight='bold', pad=10, color='#1E293B')
                ax.set_xlabel("Küme Sayısı (k)", fontsize=9.5, labelpad=6, color='#334155')
                ax.set_ylabel("SSE", fontsize=9.5, labelpad=6, color='#334155')
                ax.set_xticks(st.session_state.k_degerleri)
                ax.tick_params(axis='both', labelsize=8.5, colors='#475569')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, linestyle='--', alpha=0.5)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            with col_table:
                sse_df = pd.DataFrame({"Küme Sayısı (k)": st.session_state.k_degerleri, "SSE": [f"{v:,.2f}" for v in st.session_state.wcss]})
                st.dataframe(sse_df, use_container_width=True, hide_index=True, height=355)

            st.markdown("#### K-Means Küme Merkezleri (Aday İşleme Merkezleri)")
            try:
                kmeans_merkezler = st.session_state.get('kmeans_sonuclari', [])
                if not kmeans_merkezler:
                    st.warning("Görüntülenecek K-Means küme merkezi bulunamadı.")
                else:
                    kume_renkleri = [
                        "#1D4ED8", "#0F766E", "#7C3AED", "#D97706", "#0284C7",
                        "#15803D", "#4F46E5", "#B45309", "#0E7490"
                    ]

                    pareto_koord_bilgi = {}
                    pareto_df_gorsel = st.session_state.get('pareto_sonuclari')
                    if pareto_df_gorsel is not None and not pareto_df_gorsel.empty:
                        ozel_koord_var = "Enlem" in pareto_df_gorsel.columns and "Boylam" in pareto_df_gorsel.columns
                        for _, prow in pareto_df_gorsel.iterrows():
                            p_il = il_adini_normalize_et(prow['İl'])
                            if not p_il:
                                continue
                            if ozel_koord_var and pd.notna(prow['Enlem']) and pd.notna(prow['Boylam']):
                                p_lat, p_lon = float(prow['Enlem']), float(prow['Boylam'])
                            elif p_il in IL_KOORDINATLARI:
                                p_lat, p_lon = IL_KOORDINATLARI[p_il]
                            else:
                                continue
                            p_ev = prow['EV_Sayısı'] if 'EV_Sayısı' in pareto_df_gorsel.columns and pd.notna(prow['EV_Sayısı']) else None
                            pareto_koord_bilgi[p_il] = (p_lat, p_lon, p_ev)

                    tablo_satirlari = []
                    m_kmeans = folium.Map(location=[39.0, 35.0], zoom_start=6)

                    # Önce kümeye göre renklendirilmiş Pareto şehirlerini (Aday Toplama Merkezleri) çiz
                    for idx, merkez in enumerate(kmeans_merkezler, start=1):
                        kume_rengi = kume_renkleri[(idx - 1) % len(kume_renkleri)]
                        kod = merkez.get("Kod", f"İM-{idx}")
                        en_yakin_il = il_adini_normalize_et(merkez.get("İl", "")) or "-"
                        c_lat = float(merkez.get("Centroid_Lat", merkez.get("Lat")))
                        c_lon = float(merkez.get("Centroid_Lon", merkez.get("Lon")))
                        atanan_iller = merkez.get("Atanan_İller", [])

                        tablo_satirlari.append({
                            "Küme No": idx,
                            "Merkez Kodu": kod,
                            "Enlem": f"{c_lat:.4f}",
                            "Boylam": f"{c_lon:.4f}",
                            "En Yakın İl": en_yakin_il,
                            "İl Sayısı": int(merkez.get("Küme_Boyutu", len(atanan_iller))),
                        })

                        for ham_il in atanan_iller:
                            il_adi = il_adini_normalize_et(ham_il)
                            if il_adi in pareto_koord_bilgi:
                                lat, lon, ev_sayisi = pareto_koord_bilgi[il_adi]
                            elif il_adi in IL_KOORDINATLARI:
                                lat, lon = IL_KOORDINATLARI[il_adi]
                                ev_sayisi = None
                            else:
                                continue

                            ev_satiri = f"<br><b>EV Sayısı:</b> {ev_sayisi:,.0f}" if ev_sayisi is not None else ""
                            il_popup_html = (
                                f"<b>Aday Toplama Merkezi</b><br>"
                                f"<b>İl:</b> {il_adi}<br>"
                                f"<b>Küme No:</b> Küme {idx} ({kod})"
                                f"{ev_satiri}"
                            )
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=6,
                                color="#FFFFFF",
                                weight=1.5,
                                fill=True,
                                fill_color=kume_rengi,
                                fill_opacity=0.88,
                                tooltip=f"{il_adi} (Küme {idx} — {kod})",
                                popup=folium.Popup(il_popup_html, max_width=240),
                            ).add_to(m_kmeans)

                    # Ardından K-Means tam küme merkezlerini (Aday İşleme Merkezleri) üst katmanda belirgin kırmızı işaretçiyle çiz
                    for idx, merkez in enumerate(kmeans_merkezler, start=1):
                        kod = merkez.get("Kod", f"İM-{idx}")
                        en_yakin_il = il_adini_normalize_et(merkez.get("İl", "")) or "-"
                        c_lat = float(merkez.get("Centroid_Lat", merkez.get("Lat")))
                        c_lon = float(merkez.get("Centroid_Lon", merkez.get("Lon")))

                        merkez_popup_html = (
                            f"<b>K-Means Küme Merkezi ({kod})</b><br>"
                            f"<b>Küme No:</b> {idx}<br>"
                            f"<b>Enlem:</b> {c_lat:.6f}<br>"
                            f"<b>Boylam:</b> {c_lon:.6f}<br>"
                            f"<b>En Yakın İl:</b> {en_yakin_il}"
                        )
                        folium.Marker(
                            location=[c_lat, c_lon],
                            tooltip=f"Küme {idx} Merkezi ({kod}) — En Yakın İl: {en_yakin_il}",
                            popup=folium.Popup(merkez_popup_html, max_width=260),
                            icon=folium.Icon(color="red", icon="cog"),
                        ).add_to(m_kmeans)

                    kume_df = pd.DataFrame(tablo_satirlari)
                    st.dataframe(
                        kume_df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(360, 38 + 35 * len(kume_df)),
                    )
                    st_folium(m_kmeans, width=900, height=440, key="map_kmeans_clusters")
            except Exception as e:
                st.warning(f"K-Means küme merkezleri görselleştirilirken bir sorun oluştu: {e}")

            if st.button("Minimax Algoritmasını Çalıştır", type="primary"):
                st.session_state.mod1_step = 3
                st.rerun()

    if st.session_state.mod1_step >= 3:
        st.divider()
        st.subheader("Adım 4: Minimax Algoritması Sonuçları")
        try:
            isleme_merkezleri = st.session_state.get('kmeans_sonuclari', [])
            minimax_res = dinamik_minimax_p_center(isleme_merkezleri, p=2)
            st.session_state.minimax_sonuclari = minimax_res
            secilen_iller_str = ", ".join(m["İl"] for m in minimax_res["secilen_merkezler"])
            st.success(
                f"Dinamik Minimax (p-Center, p={minimax_res['p_kullanilan']}) optimizasyonu tamamlandı! "
                f"Seçilen merkezler: {secilen_iller_str} "
                f"(Maksimum Öklid Mesafesi Z = {minimax_res['minimax_z']:,.4f} | ≈ {minimax_res['minimax_z_km']:,.2f} km)."
            )

            m_minimax = folium.Map(location=[39.0, 35.0], zoom_start=5.5)

            renk_toplama = "#2F70AF"
            renk_isleme = "#DC2626"
            renk_ortak = "#F97316"

            harita_noktalari = []

            if 'pareto_sonuclari' in st.session_state:
                pareto_df_mm = st.session_state.pareto_sonuclari
                ozel_koord_mm = "Enlem" in pareto_df_mm.columns and "Boylam" in pareto_df_mm.columns
                for _, row in pareto_df_mm.iterrows():
                    il_adi = il_adini_normalize_et(row['İl'])
                    if ozel_koord_mm and pd.notna(row['Enlem']) and pd.notna(row['Boylam']):
                        lat, lon = float(row['Enlem']), float(row['Boylam'])
                    elif il_adi in IL_KOORDINATLARI:
                        lat, lon = float(IL_KOORDINATLARI[il_adi][0]), float(IL_KOORDINATLARI[il_adi][1])
                    else:
                        continue
                    popup_html = (
                        f"<b>Aday Toplama Merkezi</b><br>"
                        f"<b>Konum:</b> {il_adi}<br>"
                        f"<b>Enlem:</b> {lat:.6f}<br>"
                        f"<b>Boylam:</b> {lon:.6f}"
                    )
                    harita_noktalari.append({
                        "lat": lat,
                        "lon": lon,
                        "renk": renk_toplama,
                        "tooltip": f"Aday Toplama Merkezi: {il_adi}",
                        "popup": popup_html,
                    })

            if 'kmeans_sonuclari' in st.session_state:
                for merkez in st.session_state.kmeans_sonuclari:
                    il_adi = il_adini_normalize_et(merkez['İl'])
                    lat, lon = float(merkez['Lat']), float(merkez['Lon'])
                    kod = merkez.get('Kod', '')
                    kod_ek = f" ({kod})" if kod else ""
                    popup_html = (
                        f"<b>Aday İşleme Merkezi{kod_ek}</b><br>"
                        f"<b>Konum:</b> {il_adi}<br>"
                        f"<b>Enlem:</b> {lat:.6f}<br>"
                        f"<b>Boylam:</b> {lon:.6f}"
                    )
                    harita_noktalari.append({
                        "lat": lat,
                        "lon": lon,
                        "renk": renk_isleme,
                        "tooltip": f"Aday İşleme Merkezi: {il_adi}{kod_ek}",
                        "popup": popup_html,
                    })

            for m in minimax_res["secilen_merkezler"]:
                il_adi = il_adini_normalize_et(m["İl"])
                lat = float(m["Lat"])
                lon = float(m["Lon"])
                kod = m.get("Kod", "")
                kod_ek = f" ({kod})" if kod else ""
                popup_html = (
                    f"<b>Aday Geri Dönüşüm &amp; İkinci Ömür Merkezi{kod_ek}</b><br>"
                    f"<b>Konum:</b> {il_adi}<br>"
                    f"<b>Enlem:</b> {lat:.6f}<br>"
                    f"<b>Boylam:</b> {lon:.6f}<br>"
                    f"<span style='font-size: 11px; color: #475569;'>"
                    f"Kesin tesis türü ataması (Geri Dönüşüm / İkinci Ömür) "
                    f"Modül 2 optimizasyon modelinde belirlenecektir."
                    f"</span>"
                )
                harita_noktalari.append({
                    "lat": lat,
                    "lon": lon,
                    "renk": renk_ortak,
                    "tooltip": f"Aday Geri Dönüşüm & İkinci Ömür Merkezi: {il_adi}{kod_ek}",
                    "popup": popup_html,
                })

            # Çakışan / çok yakın (< 0.05 derece) noktaları dinamik grupla (yalnızca görsel çizim için)
            ziyaret = [False] * len(harita_noktalari)
            offset_yaricap = 0.12
            yakinlik_esigi = 0.05

            for i in range(len(harita_noktalari)):
                if ziyaret[i]:
                    continue
                grup_indeksler = [i]
                ziyaret[i] = True
                kuyruk = [i]
                while kuyruk:
                    mevcut = kuyruk.pop(0)
                    m_lat, m_lon = harita_noktalari[mevcut]["lat"], harita_noktalari[mevcut]["lon"]
                    for j in range(len(harita_noktalari)):
                        if not ziyaret[j]:
                            d_lat = harita_noktalari[j]["lat"] - m_lat
                            d_lon = harita_noktalari[j]["lon"] - m_lon
                            if float(np.hypot(d_lat, d_lon)) < yakinlik_esigi:
                                ziyaret[j] = True
                                grup_indeksler.append(j)
                                kuyruk.append(j)

                n_grup = len(grup_indeksler)
                for sira, nokta_idx in enumerate(grup_indeksler):
                    nokta = harita_noktalari[nokta_idx]
                    if n_grup == 1:
                        d_lat_off, d_lon_off = 0.0, 0.0
                    elif n_grup == 2:
                        d_lat_off = 0.0
                        d_lon_off = -offset_yaricap if sira == 0 else offset_yaricap
                    else:
                        aci = (np.pi / 2.0) + (2.0 * np.pi * sira / n_grup)
                        d_lat_off = float(offset_yaricap * np.sin(aci))
                        d_lon_off = float(offset_yaricap * np.cos(aci))

                    cizim_lat = nokta["lat"] + d_lat_off
                    cizim_lon = nokta["lon"] + d_lon_off

                    folium.CircleMarker(
                        location=[cizim_lat, cizim_lon],
                        radius=7.5,
                        color="#FFFFFF",
                        weight=2,
                        fill=True,
                        fill_color=nokta["renk"],
                        fill_opacity=0.9,
                        tooltip=folium.Tooltip(nokta["tooltip"], sticky=True),
                        popup=folium.Popup(nokta["popup"], max_width=310),
                    ).add_to(m_minimax)

            st.markdown(
                f"""
                <div style="display: flex; flex-wrap: wrap; gap: 18px; align-items: center; padding: 8px 14px; margin-bottom: 10px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.35); font-size: 14px;">
                    <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_toplama}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Mavi:</b> Aday Toplama Merkezleri</span>
                    <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_isleme}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Kırmızı:</b> Aday İşleme Merkezleri</span>
                    <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_ortak}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Turuncu:</b> Aday Geri Dönüşüm / İkinci Ömür Merkezleri (Ortak Aday)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st_folium(m_minimax, width=900, height=450, key="map_minimax")
        except Exception as e:
            st.error(f"Minimax (p-Center) hesaplaması sırasında hata oluştu: {e}")

        st.write("<br>", unsafe_allow_html=True)
        if st.button("Modül 2: Optimizasyon Modelini Çöz ➡", type="primary"):
            st.session_state.current_view = 'module2'
            st.rerun()

# ==========================================
# 5. MODÜL 2: MILP OPTİMİZASYON VE SENARYO ANALİZİ
# ==========================================
def solve_milp_model(params):
    """
    Tezdeki 5 dönemli (t = 1..5), stoklu (I1..I4) ve kâr maksimizasyonlu tam
    MILP modelini SciPy HiGHS / PuLP uyumlu altyapı ile çözer; hem 5 yıllık
    genel özeti hem dönem bazlı (`donemler[1..5]`) çıktıları döndürür.
    """
    return milp_5_donem_coz(params)

def _render_opt_tesis_haritasi(
    x1_dict,
    x2_dict,
    x3_dict,
    x4_dict,
    map_key,
    w1_rows=None,
    w2_rows=None,
    w3_rows=None,
    height=440,
):
    renk_toplama = "#2B6CB0"
    renk_isleme = "#DC2626"
    renk_iom = "#F97316"
    renk_gdm = "#16A34A"

    harita_noktalari = []
    tesis_gruplari = [
        ("Toplama Merkezi", x1_dict or {}, renk_toplama),
        ("İşleme Merkezi", x2_dict or {}, renk_isleme),
        ("İkinci Ömür Merkezi", x3_dict or {}, renk_iom),
        ("Geri Dönüşüm Merkezi", x4_dict or {}, renk_gdm),
    ]

    for tur_adi, tesis_sozluk, renk in tesis_gruplari:
        for il_ham, acik in tesis_sozluk.items():
            if int(round(acik or 0)) != 1:
                continue
            il_adi = il_adini_normalize_et(il_ham)
            if il_adi not in IL_KOORDINATLARI:
                continue
            lat, lon = float(IL_KOORDINATLARI[il_adi][0]), float(IL_KOORDINATLARI[il_adi][1])
            popup_html = (
                f"<b>{tur_adi}</b><br>"
                f"<b>Konum:</b> {il_adi}<br>"
                f"<b>Enlem:</b> {lat:.6f}<br>"
                f"<b>Boylam:</b> {lon:.6f}"
            )
            harita_noktalari.append({
                "lat": lat,
                "lon": lon,
                "renk": renk,
                "tooltip": f"{tur_adi}: {il_adi}",
                "popup": popup_html,
            })

    st.markdown(
        f"""
        <div style="display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 8px 14px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #E2E8F0; background-color: #FFFFFF; font-size: 13.5px; color: #1E293B;">
            <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_toplama}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Mavi:</b> Toplama</span>
            <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_isleme}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Kırmızı:</b> İşleme</span>
            <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_iom}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Turuncu:</b> İkinci Ömür</span>
            <span><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{renk_gdm}; border:1.5px solid #FFFFFF; box-shadow:0 0 0 1px rgba(15,23,42,0.2); margin-right:6px; vertical-align:middle;"></span><b>Yeşil:</b> Geri Dönüşüm</span>
            <span style="color:#475569;">| <span style="display:inline-block; width:18px; border-top:2.5px solid #DC2626; vertical-align:middle; margin-right:4px;"></span>Toplama → İşleme</span>
            <span style="color:#475569;"><span style="display:inline-block; width:18px; border-top:2.5px dashed #111827; vertical-align:middle; margin-right:4px;"></span>İşleme → İÖM / GDM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not harita_noktalari:
        st.info("Bu dönem için haritada gösterilecek aktif tesis bulunmamaktadır.")
        return

    m_opt = folium.Map(location=[39.0, 35.0], zoom_start=5.5)

    # Akış çizgileri (README.md: Toplama→İşleme düz kırmızı, İşleme→İÖM/GDM kesikli siyah)
    cizilen_w1 = set()
    for r in (w1_rows or []):
        j_il = il_adini_normalize_et(r.get("Toplama Merkezi (j)", ""))
        k_il = il_adini_normalize_et(r.get("İşleme Merkezi (k)", ""))
        if j_il and k_il and j_il != k_il and (j_il, k_il) not in cizilen_w1:
            if j_il in IL_KOORDINATLARI and k_il in IL_KOORDINATLARI:
                cizilen_w1.add((j_il, k_il))
                folium.PolyLine(
                    locations=[IL_KOORDINATLARI[j_il], IL_KOORDINATLARI[k_il]],
                    color="#DC2626",
                    weight=2.2,
                    opacity=0.72,
                    tooltip=f"Toplama → İşleme: {j_il} → {k_il}",
                ).add_to(m_opt)

    cizilen_w23 = set()
    for r in (w2_rows or []):
        k_il = il_adini_normalize_et(r.get("İşleme Merkezi (k)", ""))
        l_il = il_adini_normalize_et(r.get("İkinci Ömür Merkezi (l)", ""))
        if k_il and l_il and k_il != l_il and ("iom", k_il, l_il) not in cizilen_w23:
            if k_il in IL_KOORDINATLARI and l_il in IL_KOORDINATLARI:
                cizilen_w23.add(("iom", k_il, l_il))
                folium.PolyLine(
                    locations=[IL_KOORDINATLARI[k_il], IL_KOORDINATLARI[l_il]],
                    color="#111827",
                    weight=2.2,
                    opacity=0.78,
                    dash_array="6, 6",
                    tooltip=f"İşleme → İkinci Ömür: {k_il} → {l_il}",
                ).add_to(m_opt)

    for r in (w3_rows or []):
        k_il = il_adini_normalize_et(r.get("İşleme Merkezi (k)", ""))
        m_il = il_adini_normalize_et(r.get("Geri Dönüşüm Merkezi (m)", ""))
        if k_il and m_il and k_il != m_il and ("gdm", k_il, m_il) not in cizilen_w23:
            if k_il in IL_KOORDINATLARI and m_il in IL_KOORDINATLARI:
                cizilen_w23.add(("gdm", k_il, m_il))
                folium.PolyLine(
                    locations=[IL_KOORDINATLARI[k_il], IL_KOORDINATLARI[m_il]],
                    color="#111827",
                    weight=2.2,
                    opacity=0.78,
                    dash_array="6, 6",
                    tooltip=f"İşleme → Geri Dönüşüm: {k_il} → {m_il}",
                ).add_to(m_opt)

    # Çakışan / çok yakın (< 0.05 derece) açık tesisler için dinamik mikro görsel kaydırma
    ziyaret = [False] * len(harita_noktalari)
    offset_yaricap = 0.12
    yakinlik_esigi = 0.05

    for i in range(len(harita_noktalari)):
        if ziyaret[i]:
            continue
        grup_indeksler = [i]
        ziyaret[i] = True
        kuyruk = [i]
        while kuyruk:
            mevcut = kuyruk.pop(0)
            m_lat, m_lon = harita_noktalari[mevcut]["lat"], harita_noktalari[mevcut]["lon"]
            for j in range(len(harita_noktalari)):
                if not ziyaret[j]:
                    d_lat = harita_noktalari[j]["lat"] - m_lat
                    d_lon = harita_noktalari[j]["lon"] - m_lon
                    if float(np.hypot(d_lat, d_lon)) < yakinlik_esigi:
                        ziyaret[j] = True
                        grup_indeksler.append(j)
                        kuyruk.append(j)

        n_grup = len(grup_indeksler)
        for sira, nokta_idx in enumerate(grup_indeksler):
            nokta = harita_noktalari[nokta_idx]
            if n_grup == 1:
                d_lat_off, d_lon_off = 0.0, 0.0
            elif n_grup == 2:
                d_lat_off = 0.0
                d_lon_off = -offset_yaricap if sira == 0 else offset_yaricap
            else:
                aci = (np.pi / 2.0) + (2.0 * np.pi * sira / n_grup)
                d_lat_off = float(offset_yaricap * np.sin(aci))
                d_lon_off = float(offset_yaricap * np.cos(aci))

            cizim_lat = nokta["lat"] + d_lat_off
            cizim_lon = nokta["lon"] + d_lon_off

            folium.CircleMarker(
                location=[cizim_lat, cizim_lon],
                radius=7.5,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fill_color=nokta["renk"],
                fill_opacity=0.9,
                tooltip=folium.Tooltip(nokta["tooltip"], sticky=True),
                popup=folium.Popup(nokta["popup"], max_width=290),
            ).add_to(m_opt)

    st_folium(m_opt, width=900, height=height, key=map_key)


def _tablo_veya_bilgi(kayitlar, bos_mesaj="Bu dönem için kayıt bulunmamaktadır."):
    if kayitlar:
        st.dataframe(pd.DataFrame(kayitlar), use_container_width=True, hide_index=True)
    else:
        st.info(bos_mesaj)


def show_module2():
    st.header("Modül 2: MILP Ağ Optimizasyonu ve Senaryo Analizi")
    col_back, col_reset = st.columns([8, 2])
    col_back.button("⬅ Kontrol Paneline Dön", on_click=go_to_dashboard)
    col_reset.button("🔄 Hafızayı Sıfırla", on_click=reset_system)
    st.divider()

    if st.session_state.mod1_data is not None:
        aktif_df, _ = veriyi_oku_ve_temizle(st.session_state.mod1_data)
    else:
        aktif_df, _ = veriyi_oku_ve_temizle(
            pd.DataFrame(list(VARSAYILAN_EV_VERISI.items()), columns=['İl', 'EV_Sayısı'])
        )

    ev_data = {
        il_adini_normalize_et(r['İl']): float(r['EV_Sayısı'])
        for _, r in aktif_df.iterrows()
        if il_adini_normalize_et(r['İl']) in IL_KOORDINATLARI
    }

    otomatik_zincir = None
    if not (
        'pareto_sonuclari' in st.session_state
        and 'kmeans_sonuclari' in st.session_state
        and 'minimax_sonuclari' in st.session_state
    ):
        otomatik_zincir = tum_zinciri_calistir(aktif_df, esik=80.0, p=2, random_state=42)

    if 'pareto_sonuclari' in st.session_state:
        varsayilan_J = [il_adini_normalize_et(x) for x in st.session_state.pareto_sonuclari['İl'].tolist()]
    else:
        varsayilan_J = otomatik_zincir["pareto"]["pareto_df"]["İl"].tolist()

    if 'kmeans_sonuclari' in st.session_state:
        varsayilan_K = [il_adini_normalize_et(x['İl']) for x in st.session_state.kmeans_sonuclari]
    else:
        varsayilan_K = [x['İl'] for x in otomatik_zincir["kmeans"]["aday_isleme_merkezleri"]]

    if 'minimax_sonuclari' in st.session_state:
        minimax_adaylar = [il_adini_normalize_et(x['İl']) for x in st.session_state.minimax_sonuclari['secilen_merkezler']]
    elif 'kmeans_sonuclari' in st.session_state:
        minimax_hesap = dinamik_minimax_p_center(st.session_state.kmeans_sonuclari, p=2)
        minimax_adaylar = [x['İl'] for x in minimax_hesap['secilen_merkezler']]
    else:
        minimax_adaylar = [x['İl'] for x in otomatik_zincir["minimax"]["secilen_merkezler"]]

    varsayilan_L = list(minimax_adaylar)
    varsayilan_M = list(minimax_adaylar)

    st.subheader("1. Senaryo Parametrelerini Yapılandırın (What-If Analizi)")
    senaryo_adi = st.text_input("Senaryo Adı", value=f"Senaryo {len(st.session_state.senaryo_gecmisi) + 1}")

    with st.expander("A. Talep, Geri Toplama ve Batarya Kalite (SoH) Oranları", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        gamma = c1.slider("Geri Toplama Oranı γ (%)", 5, 100, 40, 5)
        binek_payi = c2.slider("Binek Araç Payı (%)", 50, 99, 95, 1)
        R1 = c3.number_input("Geri Dönüşüm Oranı R1 (%)", min_value=0.0, max_value=100.0, value=62.0, step=1.0)
        R2 = c4.number_input("İkinci Ömür Oranı R2 (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        R3 = round(100.0 - (R1 + R2), 2)
        c5.metric("Bertaraf Oranı R3 (%)", f"%{R3}", delta="Otomatik", delta_color="off")

    with st.expander("B. Tesis Kapasiteleri ve Sabit İnşa Maliyetleri", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        cap_j = k1.number_input("Toplama Kapasitesi", min_value=0, max_value=None, value=100000, step=5000)
        cap_k = k2.number_input("İşleme Kapasitesi", min_value=0, max_value=None, value=100000, step=5000)
        cap_l = k3.number_input("İkinci Ömür Kapasitesi (Ton)", min_value=0, max_value=None, value=15000, step=500)
        cap_m = k4.number_input("Geri Dönüşüm Kapasitesi (Ton)", min_value=0, max_value=None, value=30000, step=1000)

        m1, m2, m3, m4 = st.columns(4)
        fj = m1.number_input("Toplama İnşa ($)", min_value=0, max_value=None, value=250000, step=25000)
        fk = m2.number_input("İşleme İnşa ($)", min_value=0, max_value=None, value=400000, step=25000)
        fl = m3.number_input("İkinci Ömür İnşa ($)", min_value=0, max_value=None, value=650000, step=25000)
        fm = m4.number_input("Geri Dönüşüm İnşa ($)", min_value=0, max_value=None, value=450000, step=25000)

    with st.expander("C. Birim Taşıma, İşlem Maliyetleri ve Satış Fiyatları", expanded=False):
        t1, t2, t3, t4, t5 = st.columns(5)
        c3_tasima = t1.number_input("Batarya Taşıma C3", min_value=0.0, max_value=None, value=8.0, step=0.5)
        c4_tasima = t2.number_input("Hücre Taşıma C4", min_value=0.0, max_value=None, value=0.08, step=0.01)
        c1_binek = t3.number_input("İşleme Binek C1", min_value=0.0, max_value=None, value=350.0, step=50.0)
        c1_agir = t4.number_input("İşleme Ağır C1", min_value=0.0, max_value=None, value=3200.0, step=100.0)
        c5_bertaraf = t5.number_input("Bertaraf C5", min_value=0.0, max_value=None, value=5.0, step=1.0)

        g1, g2, g3, g4 = st.columns(4)
        rev_kobalt = g1.number_input("Kobalt ($/kg)", min_value=0.0, max_value=None, value=50.0, step=5.0)
        rev_lityum = g2.number_input("Lityum ($/kg)", min_value=0.0, max_value=None, value=1.5, step=0.2)
        rev_nikel = g3.number_input("Nikel ($/kg)", min_value=0.0, max_value=None, value=12.0, step=1.0)
        hucre_bazli_cc = g4.checkbox("Hücre Bazlı Toplama Maliyeti", value=False)

    with st.expander("D. Aday Tesis Lokasyon Kümeleri (J, K, L, M)", expanded=False):
        tum_iller = sorted(list(IL_KOORDINATLARI.keys()))
        aday_J = st.multiselect("Aday Toplama (J)", options=tum_iller, default=[x for x in varsayilan_J if x in tum_iller])
        aday_K = st.multiselect("Aday İşleme (K)", options=tum_iller, default=[x for x in varsayilan_K if x in tum_iller])
        aday_L = st.multiselect("Aday İkinci Ömür (L)", options=tum_iller, default=[x for x in varsayilan_L if x in tum_iller])
        aday_M = st.multiselect("Aday Geri Dönüşüm (M)", options=tum_iller, default=[x for x in varsayilan_M if x in tum_iller])

    if st.button("🚀 Optimizasyon Modelini (MILP) Çalıştır", type="primary", use_container_width=True):
        if R3 < 0 or not (aday_J and aday_K and aday_L and aday_M):
            st.error("Lütfen R1+R2 <= 100 olduğundan ve her tesis için en az 1 il seçildiğinden emin olun.")
        else:
            with st.spinner("MILP modeli çözülüyor..."):
                try:
                    params = {
                        'ev_data': ev_data, 'gamma': gamma, 'binek_payi': binek_payi,
                        'R1': R1, 'R2': R2, 'R3': R3,
                        'cap_j': cap_j, 'cap_k': cap_k, 'cap_l': cap_l, 'cap_m': cap_m,
                        'fj': fj, 'fk': fk, 'fl': fl, 'fm': fm,
                        'c3_tasima': c3_tasima, 'c4_tasima': c4_tasima,
                        'c1_binek': c1_binek, 'c1_agir': c1_agir, 'c5_bertaraf': c5_bertaraf,
                        'rev_kobalt': rev_kobalt, 'rev_lityum': rev_lityum, 'rev_nikel': rev_nikel,
                        'hucre_bazli_cc': hucre_bazli_cc,
                        'aday_J': aday_J, 'aday_K': aday_K, 'aday_L': aday_L, 'aday_M': aday_M
                    }
                    sonuc = solve_milp_model(params)
                    st.session_state.milp_sonuc = sonuc

                    if sonuc.get('durum') == 'Optimal':
                        net_kar_val = float(sonuc.get('toplam_net_kar', sonuc.get('amac_degeri', 0.0)))
                        gelir_val = float(sonuc.get('toplam_gelir', 0.0))
                        maliyet_val = float(sonuc.get('toplam_maliyet', 0.0))
                        st.session_state.senaryo_gecmisi.append({
                            "Senaryo": senaryo_adi,
                            "Durum": sonuc['durum'],
                            "Toplam Gelir ($)": round(gelir_val, 2),
                            "Toplam Maliyet ($)": round(maliyet_val, 2),
                            "Toplam Net Kâr ($)": round(net_kar_val, 2),
                            "Açılan Toplama": sum(sonuc.get('x_j', {}).values()),
                            "Açılan İşleme": sum(sonuc.get('x_k', {}).values()),
                            "Açılan İkinci Ömür": sum(sonuc.get('x_l', {}).values()),
                            "Açılan Geri Dönüşüm": sum(sonuc.get('x_m', {}).values()),
                        })
                except Exception as e:
                    st.error(f"Optimizasyon modeli çalıştırılırken beklenmedik bir hata oluştu: {e}")

    if 'milp_sonuc' in st.session_state and st.session_state.milp_sonuc is not None:
        res = st.session_state.milp_sonuc
        st.divider()
        if res.get('durum') != 'Optimal':
            st.error(f"Model uygun çözüm bulamadı! Durum: **{res.get('durum', 'Bilinmiyor')}**.")
        else:
            st.subheader("2. Optimizasyon Sonuçları (5 Dönemli Ağ Tasarımı)")
            donemler_sozluk = res.get('donemler', {})

            ana_sekmeler = st.tabs([
                "📊 5 Dönem Karşılaştırma",
                "Dönem 1",
                "Dönem 2",
                "Dönem 3",
                "Dönem 4",
                "Dönem 5",
            ])

            # --------------------------------------------------
            # SEKME 0: 5 DÖNEM KARŞILAŞTIRMA
            # --------------------------------------------------
            with ana_sekmeler[0]:
                toplam_net_kar = float(res.get('toplam_net_kar', res.get('amac_degeri', 0.0)))
                toplam_gelir = float(res.get('toplam_gelir', 0.0))
                toplam_maliyet = float(res.get('toplam_maliyet', 0.0))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Çözüm Durumu", res.get('durum', 'Optimal'))
                m2.metric("5 Yıllık Toplam Net Kâr", f"${toplam_net_kar:,.0f}")
                m3.metric("5 Yıllık Toplam Gelir", f"${toplam_gelir:,.0f}")
                m4.metric("5 Yıllık Toplam Maliyet", f"${toplam_maliyet:,.0f}")

                karsilastirma_satirlari = []
                for t_no in range(1, 6):
                    dv = donemler_sozluk.get(t_no) or donemler_sozluk.get(str(t_no))
                    if dv is None:
                        continue
                    karsilastirma_satirlari.append({
                        "Dönem": f"Dönem {t_no}",
                        "Gelir ($)": float(dv.get("gelir", 0.0)),
                        "Maliyet ($)": float(dv.get("maliyet", 0.0)),
                        "Net Kâr ($)": float(dv.get("net_kar", 0.0)),
                        "Aktif Toplama": sum((dv.get("x1") or {}).values()),
                        "Aktif İşleme": sum((dv.get("x2") or {}).values()),
                        "Aktif İkinci Ömür": sum((dv.get("x3") or {}).values()),
                        "Aktif Geri Dönüşüm": sum((dv.get("x4") or {}).values()),
                    })

                if karsilastirma_satirlari:
                    st.markdown("#### Dönemler Bazında Özet Karşılaştırma")
                    df_karsilastirma_gosterim = pd.DataFrame([
                        {
                            "Dönem": s["Dönem"],
                            "Gelir ($)": f"${s['Gelir ($)']:,.2f}",
                            "Maliyet ($)": f"${s['Maliyet ($)']:,.2f}",
                            "Net Kâr ($)": f"${s['Net Kâr ($)']:,.2f}",
                            "Aktif Toplama": s["Aktif Toplama"],
                            "Aktif İşleme": s["Aktif İşleme"],
                            "Aktif İkinci Ömür": s["Aktif İkinci Ömür"],
                            "Aktif Geri Dönüşüm": s["Aktif Geri Dönüşüm"],
                        }
                        for s in karsilastirma_satirlari
                    ])
                    st.dataframe(
                        df_karsilastirma_gosterim,
                        use_container_width=True,
                        hide_index=True,
                        height=min(280, 38 + 35 * len(df_karsilastirma_gosterim)),
                    )

                    col_graf, col_kalem = st.columns([1.45, 1], gap="medium")
                    with col_graf:
                        donem_etiketler = [s["Dönem"] for s in karsilastirma_satirlari]
                        gelirler = [s["Gelir ($)"] for s in karsilastirma_satirlari]
                        maliyetler = [s["Maliyet ($)"] for s in karsilastirma_satirlari]
                        net_karlar = [s["Net Kâr ($)"] for s in karsilastirma_satirlari]

                        x_pos = np.arange(len(donem_etiketler))
                        bar_w = 0.26

                        fig_comp, ax_comp = plt.subplots(figsize=(7.2, 3.9), dpi=100)
                        ax_comp.bar(x_pos - bar_w, gelirler, width=bar_w, label="Gelir", color="#2E8B57")
                        ax_comp.bar(x_pos, maliyetler, width=bar_w, label="Maliyet", color="#64748B")
                        ax_comp.bar(x_pos + bar_w, net_karlar, width=bar_w, label="Net Kâr", color="#2B6CB0")
                        ax_comp.axhline(0, color="#334155", linewidth=0.8)
                        ax_comp.set_xticks(x_pos)
                        ax_comp.set_xticklabels(donem_etiketler, fontsize=9, color="#1E293B")
                        ax_comp.set_ylabel("Tutar ($)", fontsize=9.5, color="#334155")
                        ax_comp.set_title("5 Dönem Gelir, Maliyet ve Net Kâr Karşılaştırması", fontsize=11, fontweight="bold", pad=10, color="#1E293B")
                        ax_comp.tick_params(axis="y", labelsize=8.5, colors="#475569")
                        ax_comp.spines["top"].set_visible(False)
                        ax_comp.spines["right"].set_visible(False)
                        ax_comp.grid(axis="y", linestyle="--", alpha=0.45)
                        ax_comp.legend(frameon=False, fontsize=8.5, loc="upper left")
                        fig_comp.tight_layout()
                        st.pyplot(fig_comp, use_container_width=True)
                        plt.close(fig_comp)

                    with col_kalem:
                        if res.get('maliyet_kalemleri'):
                            st.markdown("**5 Yıllık Gelir / Maliyet Kırılımı**")
                            df_cost = pd.DataFrame({
                                "Kalem": list(res['maliyet_kalemleri'].keys()),
                                "Tutar ($)": [f"${v:,.2f}" for v in res['maliyet_kalemleri'].values()],
                            })
                            st.dataframe(df_cost, use_container_width=True, hide_index=True, height=340)
                else:
                    st.warning("Dönem bazlı karşılaştırma verisi bulunamadı.")

                st.markdown("#### 5 Yıllık Genel Ağ Haritası (Açılan Tüm Tesisler)")
                _render_opt_tesis_haritasi(
                    res.get('x_j'),
                    res.get('x_k'),
                    res.get('x_l'),
                    res.get('x_m'),
                    map_key="map_opt_genel",
                    w1_rows=res.get('W_jkb'),
                    w2_rows=res.get('W_klb'),
                    w3_rows=res.get('W_kmb'),
                    height=430,
                )

                if st.session_state.senaryo_gecmisi:
                    st.markdown("#### Senaryo Karşılaştırma Geçmişi")
                    st.dataframe(pd.DataFrame(st.session_state.senaryo_gecmisi), use_container_width=True, hide_index=True)

            # --------------------------------------------------
            # SEKMELER 1..5: DÖNEM 1 .. DÖNEM 5
            # --------------------------------------------------
            for t_no in range(1, 6):
                with ana_sekmeler[t_no]:
                    dv = donemler_sozluk.get(t_no) or donemler_sozluk.get(str(t_no))
                    if dv is None:
                        st.warning(f"Dönem {t_no} için optimizasyon çıktısı bulunamadı.")
                        continue

                    x1_t = dv.get("x1") or {}
                    x2_t = dv.get("x2") or {}
                    x3_t = dv.get("x3") or {}
                    x4_t = dv.get("x4") or {}

                    f1, f2, f3 = st.columns(3)
                    f1.metric(f"Dönem {t_no} Net Kârı", f"${float(dv.get('net_kar', 0.0)):,.0f}")
                    f2.metric(f"Dönem {t_no} Geliri", f"${float(dv.get('gelir', 0.0)):,.0f}")
                    f3.metric(f"Dönem {t_no} Maliyeti", f"${float(dv.get('maliyet', 0.0)):,.0f}")

                    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
                    t_c1.metric("Aktif Toplama (x1)", f"{sum(x1_t.values())}")
                    t_c2.metric("Aktif İşleme (x2)", f"{sum(x2_t.values())}")
                    t_c3.metric("Aktif İkinci Ömür (x3)", f"{sum(x3_t.values())}")
                    t_c4.metric("Aktif Geri Dönüşüm (x4)", f"{sum(x4_t.values())}")

                    st.markdown(f"#### Dönem {t_no} Karar Değişkenleri")
                    dt1, dt2, dt3, dt4, dt5, dt6 = st.tabs([
                        "Aktif Tesisler (x1..x4)",
                        "Toplama (Z)",
                        "Tesis Akışları (W1..W4)",
                        "Pazar Satışları (V)",
                        "Stoklar (I1..I4)",
                        "Hammadde Üretimi (δ)",
                    ])

                    with dt1:
                        cx1, cx2, cx3, cx4 = st.columns(4)
                        with cx1:
                            st.markdown("**Toplama (x1)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [{"İl": il_adini_normalize_et(k), "Durum": "Açık (1)" if int(round(v)) == 1 else "Kapalı (0)"} for k, v in x1_t.items()]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with cx2:
                            st.markdown("**İşleme (x2)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [{"İl": il_adini_normalize_et(k), "Durum": "Açık (1)" if int(round(v)) == 1 else "Kapalı (0)"} for k, v in x2_t.items()]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with cx3:
                            st.markdown("**İkinci Ömür (x3)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [{"İl": il_adini_normalize_et(k), "Durum": "Açık (1)" if int(round(v)) == 1 else "Kapalı (0)"} for k, v in x3_t.items()]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with cx4:
                            st.markdown("**Geri Dönüşüm (x4)**")
                            st.dataframe(
                                pd.DataFrame(
                                    [{"İl": il_adini_normalize_et(k), "Durum": "Açık (1)" if int(round(v)) == 1 else "Kapalı (0)"} for k, v in x4_t.items()]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                    with dt2:
                        _tablo_veya_bilgi(dv.get("Z"), f"Dönem {t_no} için toplama (Z) kaydı bulunmamaktadır.")

                    with dt3:
                        st.markdown("**W1 — Toplama → İşleme Akışları**")
                        _tablo_veya_bilgi(dv.get("W1"), f"Dönem {t_no} için W1 akışı bulunmamaktadır.")
                        st.markdown("**W2 — İşleme → İkinci Ömür Akışları**")
                        _tablo_veya_bilgi(dv.get("W2"), f"Dönem {t_no} için W2 akışı bulunmamaktadır.")
                        st.markdown("**W3 — İşleme → Geri Dönüşüm Akışları**")
                        _tablo_veya_bilgi(dv.get("W3"), f"Dönem {t_no} için W3 akışı bulunmamaktadır.")
                        st.markdown("**W4 — Bertaraf Akışları**")
                        _tablo_veya_bilgi(dv.get("W4"), f"Dönem {t_no} için W4 bertaraf kaydı bulunmamaktadır.")

                    with dt4:
                        _tablo_veya_bilgi(dv.get("V"), f"Dönem {t_no} için ikinci ömür pazar satışı (V) bulunmamaktadır.")

                    with dt5:
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.markdown("**I1 — Toplama Merkezi Stokları**")
                            _tablo_veya_bilgi(dv.get("I1"), f"Dönem {t_no} için I1 stoku bulunmamaktadır.")
                            st.markdown("**I3 — İşleme Merkezi Hücre Stokları**")
                            _tablo_veya_bilgi(dv.get("I3"), f"Dönem {t_no} için I3 stoku bulunmamaktadır.")
                        with col_i2:
                            st.markdown("**I2 — İşleme Merkezi Batarya Stokları**")
                            _tablo_veya_bilgi(dv.get("I2"), f"Dönem {t_no} için I2 stoku bulunmamaktadır.")
                            st.markdown("**I4 — İkinci Ömür Merkezi Stokları**")
                            _tablo_veya_bilgi(dv.get("I4"), f"Dönem {t_no} için I4 stoku bulunmamaktadır.")

                    with dt6:
                        _tablo_veya_bilgi(dv.get("delta"), f"Dönem {t_no} için hammadde üretimi (δ) kaydı bulunmamaktadır.")

                    st.markdown(f"#### Dönem {t_no} Aktif Tesisler Haritası")
                    _render_opt_tesis_haritasi(
                        x1_t,
                        x2_t,
                        x3_t,
                        x4_t,
                        map_key=f"map_opt_donem_{t_no}",
                        w1_rows=dv.get("W1"),
                        w2_rows=dv.get("W2"),
                        w3_rows=dv.get("W3"),
                        height=420,
                    )

# ==========================================
# ANA KONTROL MEKANİZMASI
# ==========================================
if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.current_view == 'dashboard':
        show_dashboard()
    elif st.session_state.current_view == 'module1':
        show_module1()
    elif st.session_state.current_view == 'module2':
        show_module2()
    render_progress_sidebar()
