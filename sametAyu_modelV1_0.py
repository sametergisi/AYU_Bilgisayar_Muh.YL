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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


CSV_FILE = "alertrecord.csv"

DATE_COL = "dataDate"
TARGET = "temp"          # temp, hum, gas, mic, ldr

RESAMPLE_TIME = "30min"
WINDOW = 6               # 6 x 30 dk = 3 saat geçmiş
MA_WINDOW = 3            # 3 x 30 dk = 1.5 saat moving average

LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 8


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

    if acc < 0:
        acc = 0

    return acc


def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    df = df.set_index(DATE_COL)

    diff_sec = df.index.to_series().diff().dt.total_seconds()
    gaps = diff_sec[diff_sec > 2]

    print("Toplam satır:", len(df))
    print("2 saniyeden büyük kopma sayısı:", len(gaps))

    if len(gaps) > 0:
        print("En büyük kopma saniye:", gaps.max())
        print("İlk 5 kopma:")
        print(gaps.head())

    # 1 saniyeye oturt
    df = df.resample("1s").mean()

    # Küçük kopmaları doldur, büyük kopmaları komple yapay doldurma
    df = df.interpolate(limit=10)

    # 30 dakikalık ortalama
    df_30 = df.resample(RESAMPLE_TIME).mean().dropna()

    return df_30


def moving_average_predict(train_series):
    return train_series.tail(MA_WINDOW).mean()


def linear_regression_predict(train_series):
    data = pd.DataFrame()
    data["y"] = train_series

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

    pred = model.predict(X_next)[0]

    return pred


def create_lstm_dataset(values, window):
    X = []
    y = []

    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])

    return np.array(X), np.array(y)


def lstm_predict(train_series):
    if len(train_series) < WINDOW + 10:
        return np.nan

    values = train_series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = create_lstm_dataset(scaled, WINDOW)

    if len(X) < 10:
        return np.nan

    model = Sequential()
    model.add(Input(shape=(WINDOW, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop]
    )

    last_window = scaled[-WINDOW:]
    last_window = last_window.reshape(1, WINDOW, 1)

    pred_scaled = model.predict(last_window, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    return pred


def rolling_backtest(series):
    results = []

    start_index = WINDOW + 10

    for i in range(start_index, len(series)):
        train_series = series.iloc[:i]
        real = series.iloc[i]
        time = series.index[i]

        ma_pred = moving_average_predict(train_series)
        lr_pred = linear_regression_predict(train_series)
        lstm_pred = lstm_predict(train_series)

        ma_err = error_percent(real, ma_pred)
        lr_err = error_percent(real, lr_pred)
        lstm_err = error_percent(real, lstm_pred)

        ma_acc = accuracy_percent(real, ma_pred)
        lr_acc = accuracy_percent(real, lr_pred)
        lstm_acc = accuracy_percent(real, lstm_pred)

        results.append({
            "time": time,
            "real": real,

            "ma_pred": ma_pred,
            "lr_pred": lr_pred,
            "lstm_pred": lstm_pred,

            "ma_error": ma_err,
            "lr_error": lr_err,
            "lstm_error": lstm_err,

            "ma_accuracy": ma_acc,
            "lr_accuracy": lr_acc,
            "lstm_accuracy": lstm_acc,
        })

        print(
            f"{time} | "
            f"Real={real:.4f} | "
            f"MA={ma_pred:.4f} Acc={ma_acc:.2f}% | "
            f"LR={lr_pred:.4f} Acc={lr_acc:.2f}% | "
            f"LSTM={lstm_pred:.4f} Acc={lstm_acc:.2f}%"
        )

    return pd.DataFrame(results)


def draw_graphs(results):
    plt.figure(figsize=(14, 6))
    plt.plot(results["time"], results["real"], label="Gerçek")
    plt.plot(results["time"], results["ma_pred"], label="Moving Average")
    plt.plot(results["time"], results["lr_pred"], label="Linear Regression")
    plt.plot(results["time"], results["lstm_pred"], label="LSTM")
    plt.title(f"{TARGET} - Gerçek Değer ve Model Tahminleri")
    plt.xlabel("Zaman")
    plt.ylabel(TARGET)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(results["time"], results["ma_accuracy"], label="Moving Average Accuracy")
    plt.plot(results["time"], results["lr_accuracy"], label="Linear Regression Accuracy")
    plt.plot(results["time"], results["lstm_accuracy"], label="LSTM Accuracy")
    plt.title(f"{TARGET} - Her 30 Dakikadaki Doğruluk Oranı")
    plt.xlabel("Zaman")
    plt.ylabel("Doğruluk (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    avg_acc = {
        "Moving Average": results["ma_accuracy"].mean(),
        "Linear Regression": results["lr_accuracy"].mean(),
        "LSTM": results["lstm_accuracy"].mean()
    }

    plt.figure(figsize=(9, 5))
    plt.bar(avg_acc.keys(), avg_acc.values())
    plt.title(f"{TARGET} - Ortalama Doğruluk Karşılaştırması")
    plt.ylabel("Ortalama Doğruluk (%)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    avg_err = {
        "Moving Average": results["ma_error"].mean(),
        "Linear Regression": results["lr_error"].mean(),
        "LSTM": results["lstm_error"].mean()
    }

    plt.figure(figsize=(9, 5))
    plt.bar(avg_err.keys(), avg_err.values())
    plt.title(f"{TARGET} - Ortalama Hata Karşılaştırması")
    plt.ylabel("Ortalama Hata (%)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def print_summary(results):
    print("\n--- Ortalama Sonuçlar ---")

    print(f"Moving Average Ortalama Hata     : %{results['ma_error'].mean():.4f}")
    print(f"Linear Regression Ortalama Hata  : %{results['lr_error'].mean():.4f}")
    print(f"LSTM Ortalama Hata               : %{results['lstm_error'].mean():.4f}")

    print()

    print(f"Moving Average Ortalama Doğruluk     : %{results['ma_accuracy'].mean():.4f}")
    print(f"Linear Regression Ortalama Doğruluk  : %{results['lr_accuracy'].mean():.4f}")
    print(f"LSTM Ortalama Doğruluk               : %{results['lstm_accuracy'].mean():.4f}")

    best_model = {
        "Moving Average": results["ma_accuracy"].mean(),
        "Linear Regression": results["lr_accuracy"].mean(),
        "LSTM": results["lstm_accuracy"].mean()
    }

    best = max(best_model, key=best_model.get)

    print("\nEn iyi model:", best)


def main():
    df_30 = load_and_prepare_data(CSV_FILE)

    if TARGET not in df_30.columns:
        print(f"Hata: {TARGET} kolonu bulunamadı.")
        print("Mevcut kolonlar:", df_30.columns.tolist())
        return

    series = df_30[TARGET].dropna()

    print("\n30 dakikalık veri sayısı:", len(series))
    print("Tahmin edilen kolon:", TARGET)

    if len(series) < WINDOW + 20:
        print("Veri çok az. WINDOW değerini düşür veya daha fazla veri kullan.")
        return

    results = rolling_backtest(series)

    results.to_csv(f"rolling_results_{TARGET}.csv", index=False)
    print(f"\nSonuçlar kaydedildi: rolling_results_{TARGET}.csv")

    print_summary(results)
    draw_graphs(results)


if __name__ == "__main__":
    main()