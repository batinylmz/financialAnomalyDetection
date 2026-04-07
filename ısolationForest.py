
import time
import warnings
import numpy as np
import pandas as pd
from itertools import product

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from pyod.models.iforest import IForest

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────
# AYARLAR VE SÜTUN LİSTELERİ
# ───────────────────────────────────────────────────────
CSV_PATH = "financial_anomaly_benchmark_data (1).csv"

# 1. CSV'nin içinde zaten var olan ve okurken temizleyeceğimiz ham veriler
RAW_COLS = [
    "Open", "High", "Low", "Close",
    "Volume", "Returns", "Volume_Change", "Volatility_HighLow"
]

# 2. Modelin EĞİTİMDE kullanacağı yeni zeki özellikler (Kopyayı engellediğimiz liste)
FEATURE_COLS = [
    "Returns",
    "Rolling_Vol_5",
    "Rolling_Vol_20",
    "Volume_Spike_Ratio",
    "Price_MA_Ratio"
]

PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_samples" : ["auto", 256, 512, 1024],
    "max_features": [0.7, 0.8, 1.0]  # Ağaçların kullanacağı veri oranı
}


# ───────────────────────────────────────────────────────
# 0. VERİ YÜKLEME (Tarih formatı düzeltilmiş hali)
# ───────────────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Tarihi zorla formatlıyoruz
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Sadece ham sütunları (RAW_COLS) sayısal değere çeviriyoruz
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=RAW_COLS, inplace=True)
    print(f"  İlk Yüklenen kayıt : {len(df):,}")
    return df

# ───────────────────────────────────────────────────────
# YENİ: ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
# ───────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham fiyatlardan ziyade, modelin anomalileri anlayabileceği
    hareketli (rolling) istatistikler üretir.
    """
    df_new = df.copy()

    # Getirilerin kısa ve orta vadeli oynaklığı
    df_new['Rolling_Vol_5'] = df_new['Returns'].rolling(window=5).std()
    df_new['Rolling_Vol_20'] = df_new['Returns'].rolling(window=20).std()

    # Hacim Şoku (Son 10 güne kıyasla bugünkü hacim)
    df_new['Volume_Spike_Ratio'] = df_new['Volume'] / (df_new['Volume'].rolling(window=10).mean() + 1e-8)

    # Fiyat Şoku (Son 20 günlük ortalamadan sapma)
    df_new['Price_MA_Ratio'] = df_new['Close'] / (df_new['Close'].rolling(window=20).mean() + 1e-8)

    # Hareketli ortalamalar ilk günlerde hesaplanamayacağı için o boşlukları siliyoruz
    df_new.dropna(subset=FEATURE_COLS, inplace=True)
    print(f"  Özellik eklendikten sonra kayıt : {len(df_new):,}\n")
    return df_new

# ───────────────────────────────────────────────────────
# 1. ZAMANA GÖRE BÖLME  (%70 / %15 / %15)
# ───────────────────────────────────────────────────────
def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert df.index.is_monotonic_increasing, "Timestamp artan sırada olmalıdır!"
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(f"  {'Küme':<12} {'Kayıt':>7}  {'Oran':>6}  {'Başlangıç':<22} Bitiş")
    print(f"  {'─'*12} {'─'*7}  {'─'*6}  {'─'*22} {'─'*22}")
    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"  {name:<12} {len(part):>7,}  {len(part)/n*100:>5.1f}%  {str(part.index[0]):<22} {str(part.index[-1])}")
    print()
    return train, val, test

# ───────────────────────────────────────────────────────
# 2. ŞOK SKORU & ETİKETLER
# ───────────────────────────────────────────────────────
def create_shock_score(df: pd.DataFrame) -> pd.Series:
    # Gerçek şokları hala orijinal oynaklık ve hacimden belirliyoruz (Burası Hakem)
    return df["Volatility_HighLow"] * df["Volume_Change"].abs()


# YENİ VE DİNAMİK ETİKETLEME FONKSİYONU
def create_labels(shock_scores: pd.Series, z_threshold: float = 3.0) -> pd.Series:
    """
    Sabit %1 yerine, istatistiksel olarak ortalamadan
    3 standart sapma uzağa düşen gerçek şokları bulur.
    """
    mean_val = shock_scores.mean()
    std_val = shock_scores.std()

    # Her günün Z-Skorunu hesapla
    z_scores = (shock_scores - mean_val) / std_val

    # Z-Skoru 3'ten büyük olanları (aşırı şokları) 1 yap, diğerlerini 0
    return (z_scores > z_threshold).astype(int)
# ───────────────────────────────────────────────────────
# 3. MODEL DEĞERLENDİRME
# ───────────────────────────────────────────────────────
benchmark_results = []
def evaluate_anomaly_model(y_true, y_pred, y_scores, model_name, inference_time):
    pr_auc  = average_precision_score(y_true, y_scores)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    result = {
        "Model"         : model_name,
        "PR-AUC"        : round(pr_auc, 4),
        "F1-Score"      : round(f1, 4),
        "ROC-AUC"       : round(roc_auc, 4),
        "Inference (s)" : round(inference_time, 4),
    }
    benchmark_results.append(result)
    return result

# ───────────────────────────────────────────────────────
# 4. ISOLATION FOREST ─ GRID SEARCH
# ───────────────────────────────────────────────────────
def grid_search_iforest(X_train, X_val, y_val, param_grid=PARAM_GRID, contamination=0.01):
    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_params = {}
    best_pr_auc = -np.inf

    print(f"  {len(combos)} kombinasyon deneniyor...\n")
    for combo in combos:
        params = dict(zip(keys, combo))
        clf = IForest(n_estimators=params["n_estimators"], max_samples=params["max_samples"], contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X_train)
        val_scores = clf.decision_function(X_val)
        pr_auc = average_precision_score(y_val, val_scores)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_params = params.copy()
    return best_params, best_pr_auc

# ───────────────────────────────────────────────────────
# 5. FINAL MODEL
# ───────────────────────────────────────────────────────
def train_and_evaluate_best_iforest(X_train, X_test, y_test, best_params):
    clf = IForest(n_estimators=best_params["n_estimators"], max_samples=best_params["max_samples"], contamination=0.01, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    t0 = time.perf_counter()
    test_scores = clf.decision_function(X_test)
    inference_t = time.perf_counter() - t0

    threshold = np.percentile(test_scores, 99)
    y_pred = (test_scores >= threshold).astype(int)

    evaluate_anomaly_model(y_test, y_pred, test_scores, "Final IForest", inference_t)
    return clf

# ───────────────────────────────────────────────────────
# ANA AKIŞ (MAIN)
# ───────────────────────────────────────────────────────
def main():
    # 1. Veriyi Oku
    df = load_data(CSV_PATH)

    # 2. Yeni Özellikleri Üret (Hatamızın çözüldüğü nokta)
    df = add_engineered_features(df)

    # 3. Parçalara Ayır
    train, val, test = time_based_split(df)

    # 4. Etiketleri Oluştur
    for split_df in [val, test]:
        shock = create_shock_score(split_df)
        split_df["y_true"] = create_labels(shock)

    # 5. Verileri Numpy Dizilerine Çevir (DİKKAT: Artık sadece FEATURE_COLS kullanılıyor)
    X_train = train[FEATURE_COLS].values
    X_val   = val[FEATURE_COLS].values
    X_test  = test[FEATURE_COLS].values
    y_val   = val["y_true"].values
    y_test  = test["y_true"].values

    # 6. Eğit ve Test Et
    best_params, _ = grid_search_iforest(X_train, X_val, y_val)
    final_model = train_and_evaluate_best_iforest(X_train, X_test, y_test, best_params)

    results_df = pd.DataFrame(benchmark_results)
    print("\nBENCHMARK TABLOSU")
    print(results_df.to_string(index=False))
    return final_model, results_df

if __name__ == "__main__":
    final_model, results_df = main()