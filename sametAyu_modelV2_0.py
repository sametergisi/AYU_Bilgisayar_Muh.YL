# rolling_predict_all_models.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ─── Ayarlar ────────────────────────────────────────────────────────────────

CSV_FILE    = "alertrecord.csv"
DATE_COL    = "dataDate"
TARGET      = "ldr"          # temp, hum, gas, mic, ldr

RESAMPLE_TIME = "1h"
WINDOW        = 6             # 6 x 30 dk = 3 saat geçmiş
MA_WINDOW     = 3             # 3 x 30 dk = 1.5 saat moving average

LSTM_EPOCHS     = 20          # warm-start sayesinde 40'a gerek yok
LSTM_EPOCHS_UPD = 5           # sonraki adımlarda güncelleme epoch sayısı
LSTM_BATCH_SIZE = 16
LSTM_UNITS      = 16          # 32 yerine 16 — hız kazancı

ARIMA_ORDER = (2, 1, 2)       # (p, d, q)

# ─── Global model nesneleri (warm-start için) ───────────────────────────────

_lstm_model  = None
_lstm_scaler = None
_arima_fitted = None


# ─── Yardımcı metrikler ──────────────────────────────────────────────────────

def error_percent(real, pred):
    if pd.isna(real) or pd.isna(pred):
        return np.nan
    if real == 0:
        return np.nan
    return abs(real - pred) / abs(real) * 100


def accuracy_percent(real, pred):
    err = error_percent(real, pred)
    if pd.isna(err):
        return np.nan
    acc = 100 - err
    return max(acc, 0)


# ─── Veri yükleme ────────────────────────────────────────────────────────────

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

    diff_sec = df.index.to_series().diff().dt.total_seconds()
    gaps = diff_sec[diff_sec > 2]

    print("Toplam satır           :", len(df))
    print("2 sn'den büyük kopma   :", len(gaps))
    if len(gaps) > 0:
        print("En büyük kopma (sn)    :", gaps.max())
        print("İlk 5 kopma:\n", gaps.head())

    df    = df.resample("1s").mean()
    df    = df.interpolate(limit=10)
    df_30 = df.resample(RESAMPLE_TIME).mean().dropna()

    return df_30


# ─── Model: Moving Average ───────────────────────────────────────────────────

def moving_average_predict(train_series):
    return train_series.tail(MA_WINDOW).mean()


# ─── Model: Linear Regression ────────────────────────────────────────────────

def linear_regression_predict(train_series):
    data = pd.DataFrame({"y": train_series})
    for i in range(1, WINDOW + 1):
        data[f"lag_{i}"] = data["y"].shift(i)
    data = data.dropna()

    if len(data) < 5:
        return np.nan

    X = data.drop(columns=["y"])
    y = data["y"]

    model = LinearRegression()
    model.fit(X, y)

    last_values = train_series.tail(WINDOW).values[::-1]
    if len(last_values) < WINDOW:
        return np.nan

    X_next = pd.DataFrame(
        [last_values],
        columns=[f"lag_{i}" for i in range(1, WINDOW + 1)]
    )
    return model.predict(X_next)[0]


# ─── Model: LSTM (warm-start) ────────────────────────────────────────────────

def create_lstm_dataset(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def lstm_predict(train_series):
    global _lstm_model, _lstm_scaler

    if len(train_series) < WINDOW + 10:
        return np.nan

    values = train_series.values.reshape(-1, 1)
    _lstm_scaler = MinMaxScaler()
    scaled = _lstm_scaler.fit_transform(values)

    X, y = create_lstm_dataset(scaled, WINDOW)
    if len(X) < 10:
        return np.nan

    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)

    if _lstm_model is None:
        # İlk adım: sıfırdan eğit
        _lstm_model = Sequential([
            Input(shape=(WINDOW, 1)),
            LSTM(LSTM_UNITS),
            Dense(1)
        ])
        _lstm_model.compile(optimizer="adam", loss="mse")
        epochs = LSTM_EPOCHS
    else:
        # Sonraki adımlar: sadece güncelle
        epochs = LSTM_EPOCHS_UPD

    _lstm_model.fit(
        X, y,
        epochs=epochs,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop]
    )

    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    pred_scaled = _lstm_model.predict(last_window, verbose=0)[0][0]
    return _lstm_scaler.inverse_transform([[pred_scaled]])[0][0]


# ─── Model: ARIMA (apply ile hızlı güncelleme) ───────────────────────────────

def arima_predict(train_series):
    global _arima_fitted

    if len(train_series) < WINDOW + 10:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)

            if _arima_fitted is None:
                # İlk adım: sıfırdan fit et
                model = ARIMA(train_series.values, order=ARIMA_ORDER)
                _arima_fitted = model.fit()
            else:
                # Sonraki adımlar: mevcut parametrelerle yeni veriye uygula
                _arima_fitted = _arima_fitted.apply(train_series.values)

            return _arima_fitted.forecast(steps=1)[0]

    except Exception:
        _arima_fitted = None  # Hata olursa sıfırla, bir sonraki adımda yeniden fit et
        return np.nan


# ─── Rolling Backtest ────────────────────────────────────────────────────────

def rolling_backtest(series):
    global _lstm_model, _lstm_scaler, _arima_fitted

    # Warm-start modellerini sıfırla (tekrar çalıştırma güvenliği)
    _lstm_model   = None
    _lstm_scaler  = None
    _arima_fitted = None

    results = []
    start_index = WINDOW + 10

    for i in range(start_index, len(series)):
        train_series = series.iloc[:i]
        real = series.iloc[i]
        time = series.index[i]

        # MA, LR, ARIMA paralel; LSTM ayrı (global model paylaşımı nedeniyle)
        par_results = Parallel(n_jobs=3)(
            delayed(f)(train_series) for f in [
                moving_average_predict,
                linear_regression_predict,
                arima_predict,
            ]
        )
        ma_pred, lr_pred, arima_pred = par_results
        lstm_pred = lstm_predict(train_series)

        ma_err    = error_percent(real, ma_pred)
        lr_err    = error_percent(real, lr_pred)
        lstm_err  = error_percent(real, lstm_pred)
        arima_err = error_percent(real, arima_pred)

        ma_acc    = accuracy_percent(real, ma_pred)
        lr_acc    = accuracy_percent(real, lr_pred)
        lstm_acc  = accuracy_percent(real, lstm_pred)
        arima_acc = accuracy_percent(real, arima_pred)

        results.append({
            "time": time,
            "real": real,

            "ma_pred":    ma_pred,
            "lr_pred":    lr_pred,
            "lstm_pred":  lstm_pred,
            "arima_pred": arima_pred,

            "ma_error":    ma_err,
            "lr_error":    lr_err,
            "lstm_error":  lstm_err,
            "arima_error": arima_err,

            "ma_accuracy":    ma_acc,
            "lr_accuracy":    lr_acc,
            "lstm_accuracy":  lstm_acc,
            "arima_accuracy": arima_acc,
        })

        print(
            f"{time} | Real={real:.4f} | "
            f"MA={ma_pred:.4f} Acc={ma_acc:.2f}% | "
            f"LR={lr_pred:.4f} Acc={lr_acc:.2f}% | "
            f"LSTM={lstm_pred:.4f} Acc={lstm_acc:.2f}% | "
            f"ARIMA={arima_pred:.4f} Acc={arima_acc:.2f}%"
        )

    return pd.DataFrame(results)


# ─── Grafikler ───────────────────────────────────────────────────────────────

def draw_graphs(results):
    # Tahmin karşılaştırması
    plt.figure(figsize=(14, 6))
    plt.plot(results["time"], results["real"],       label="Gerçek",           linewidth=2)
    plt.plot(results["time"], results["ma_pred"],    label="Moving Average",   linestyle="--")
    plt.plot(results["time"], results["lr_pred"],    label="Linear Regression",linestyle="--")
    plt.plot(results["time"], results["lstm_pred"],  label="LSTM",             linestyle="--")
    plt.plot(results["time"], results["arima_pred"], label="ARIMA",            linestyle="--")
    plt.title(f"{TARGET} - Gerçek Değer ve Model Tahminleri")
    plt.xlabel("Zaman")
    plt.ylabel(TARGET)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Doğruluk zaman serisi
    plt.figure(figsize=(14, 6))
    plt.plot(results["time"], results["ma_accuracy"],    label="Moving Average")
    plt.plot(results["time"], results["lr_accuracy"],    label="Linear Regression")
    plt.plot(results["time"], results["lstm_accuracy"],  label="LSTM")
    plt.plot(results["time"], results["arima_accuracy"], label="ARIMA")
    plt.title(f"{TARGET} - Her 30 Dakikadaki Doğruluk Oranı")
    plt.xlabel("Zaman")
    plt.ylabel("Doğruluk (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Ortalama doğruluk bar
    avg_acc = {
        "Moving Average":    results["ma_accuracy"].mean(),
        "Linear Regression": results["lr_accuracy"].mean(),
        "LSTM":              results["lstm_accuracy"].mean(),
        "ARIMA":             results["arima_accuracy"].mean(),
    }
    plt.figure(figsize=(9, 5))
    bars = plt.bar(avg_acc.keys(), avg_acc.values(), color=["steelblue","orange","green","red"])
    plt.bar_label(bars, fmt="%.2f%%")
    plt.title(f"{TARGET} - Ortalama Doğruluk Karşılaştırması")
    plt.ylabel("Ortalama Doğruluk (%)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Ortalama hata bar
    avg_err = {
        "Moving Average":    results["ma_error"].mean(),
        "Linear Regression": results["lr_error"].mean(),
        "LSTM":              results["lstm_error"].mean(),
        "ARIMA":             results["arima_error"].mean(),
    }
    plt.figure(figsize=(9, 5))
    bars = plt.bar(avg_err.keys(), avg_err.values(), color=["steelblue","orange","green","red"])
    plt.bar_label(bars, fmt="%.2f%%")
    plt.title(f"{TARGET} - Ortalama Hata Karşılaştırması")
    plt.ylabel("Ortalama Hata (%)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# ─── Özet ────────────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 55)
    print("               ORTALAMA SONUÇLAR")
    print("=" * 55)
    print(f"{'Model':<22} {'Hata %':>10}   {'Doğruluk %':>10}")
    print("-" * 55)

    models = {
        "Moving Average":    ("ma_error",    "ma_accuracy"),
        "Linear Regression": ("lr_error",    "lr_accuracy"),
        "LSTM":              ("lstm_error",  "lstm_accuracy"),
        "ARIMA":             ("arima_error", "arima_accuracy"),
    }

    best_model = None
    best_acc   = -1

    for name, (err_col, acc_col) in models.items():
        avg_err = results[err_col].mean()
        avg_acc = results[acc_col].mean()
        print(f"{name:<22} {avg_err:>10.4f}   {avg_acc:>10.4f}")
        if avg_acc > best_acc:
            best_acc   = avg_acc
            best_model = name

    print("=" * 55)
    print(f"En iyi model: {best_model}  (Ort. Doğruluk: %{best_acc:.4f})")
    print("=" * 55)


# ─── Ana akış ────────────────────────────────────────────────────────────────

def main():
    df_30 = load_and_prepare_data(CSV_FILE)

    if TARGET not in df_30.columns:
        print(f"Hata: '{TARGET}' kolonu bulunamadı.")
        print("Mevcut kolonlar:", df_30.columns.tolist())
        return

    series = df_30[TARGET].dropna()

    print(f"\n30 dakikalık veri sayısı : {len(series)}")
    print(f"Tahmin edilen kolon      : {TARGET}")

    if len(series) < WINDOW + 20:
        print("Veri çok az. WINDOW değerini düşür veya daha fazla veri kullan.")
        return

    results = rolling_backtest(series)

    out_file = f"rolling_results_{TARGET}.csv"
    results.to_csv(out_file, index=False)
    print(f"\nSonuçlar kaydedildi: {out_file}")

    print_summary(results)
    draw_graphs(results)


if __name__ == "__main__":
    main()
