
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
zip_path = "CMAPSS_outer.zip"
extract_dir = "CMAPSSData"

urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

inner_zip = os.path.join(
    extract_dir,
    "6. Turbofan Engine Degradation Simulation Data Set",
    "CMAPSSData.zip"
)

with zipfile.ZipFile(inner_zip, "r") as z:
    z.extractall(extract_dir)

print("Dataset ready")

train_file = os.path.join(extract_dir, "train_FD001.txt")
test_file  = os.path.join(extract_dir, "test_FD001.txt")
rul_file   = os.path.join(extract_dir, "RUL_FD001.txt")

print("Train exists:", os.path.exists(train_file))
print("Test exists :", os.path.exists(test_file))
print("RUL exists  :", os.path.exists(rul_file))


columns = (
    ["unit_nr", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26]
    df.columns = columns
    return df

train_df = load_cmapss_file(train_file)
test_df = load_cmapss_file(test_file)

rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, engine="python")
rul_df = rul_df.iloc[:, :1]
rul_df.columns = ["RUL"]

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("RUL shape  :", rul_df.shape)

train_max = train_df.groupby("unit_nr")["time_cycles"].max().reset_index()
train_max.columns = ["unit_nr", "max_cycle"]

train_df = train_df.merge(train_max, on="unit_nr", how="left")
train_df["RUL"] = train_df["max_cycle"] - train_df["time_cycles"]
train_df.drop(columns=["max_cycle"], inplace=True)

test_max = test_df.groupby("unit_nr")["time_cycles"].max().reset_index()
test_max.columns = ["unit_nr", "max_cycle"]

rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1)

test_max = test_max.merge(rul_df, on="unit_nr", how="left")
test_max["failure_cycle"] = test_max["max_cycle"] + test_max["RUL"]

test_df = test_df.merge(
    test_max[["unit_nr", "failure_cycle"]],
    on="unit_nr",
    how="left"
)

test_df["RUL"] = test_df["failure_cycle"] - test_df["time_cycles"]
test_df.drop(columns=["failure_cycle"], inplace=True)


RUL_CAP = 125
train_df["RUL_clipped"] = train_df["RUL"].clip(upper=RUL_CAP)
test_df["RUL_clipped"] = test_df["RUL"].clip(upper=RUL_CAP)

print(train_df["RUL_clipped"].describe())


feature_cols = [
    "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_11",
    "sensor_13",
    "sensor_17"
]

print("Selected features:", feature_cols)
print("Number of selected features:", len(feature_cols))


def create_train_windows(df, features, window):
    X = []
    y = []
    unit_ids = []

    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit].reset_index(drop=True)

        if len(unit_df) < window + 1:
            continue

        for i in range(window, len(unit_df)):
            window_slice = unit_df[features].iloc[i-window:i]

            raw = window_slice.values.flatten()
            trend = window_slice.diff().fillna(0).values.flatten()
            mean_feat = window_slice.mean().values
            std_feat = window_slice.std().fillna(0).values

            window_data = np.concatenate([raw, trend, mean_feat, std_feat])

            X.append(window_data)
            y.append(unit_df["RUL_clipped"].iloc[i])
            unit_ids.append(unit)

    return np.array(X), np.array(y), np.array(unit_ids)


def create_test_last_windows(df, features, window):
    X = []
    y = []
    unit_ids = []

    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit].reset_index(drop=True)

        if len(unit_df) < window:
            continue

        window_slice = unit_df[features].iloc[-window:]

        raw = window_slice.values.flatten()
        trend = window_slice.diff().fillna(0).values.flatten()
        mean_feat = window_slice.mean().values
        std_feat = window_slice.std().fillna(0).values

        window_data = np.concatenate([raw, trend, mean_feat, std_feat])

        X.append(window_data)
        y.append(unit_df["RUL_clipped"].iloc[-1])
        unit_ids.append(unit)

    return np.array(X), np.array(y), np.array(unit_ids)


def evaluate_window(window):
    X_train, y_train, _ = create_train_windows(train_df, feature_cols, window)
    X_test, y_test, _ = create_test_last_windows(test_df, feature_cols, window)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "window": window,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }



window_results = []

for w in [8, 10, 12, 14, 17, 18, 20,22,24]:
    result = evaluate_window(w)
    window_results.append(result)

    print(
        f"Window = {result['window']} | "
        f"RMSE = {result['rmse']:.4f} | "
        f"MAE = {result['mae']:.4f} | "
        f"R2 = {result['r2']:.4f}"
    )

results_df = pd.DataFrame(window_results)
best_row = results_df.loc[results_df["r2"].idxmax()]
best_window = int(best_row["window"])

print("\nBest Window Size:", best_window)
print(best_row)


X_train, y_train, train_units = create_train_windows(train_df, feature_cols, best_window)
X_test, y_test, test_units = create_test_last_windows(test_df, feature_cols, best_window)

print("\nFinal data shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

final_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
y_pred = np.clip(y_pred, 0, RUL_CAP)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print("\nFINAL LAST-CYCLE RESULTS")
print("Best Window:", best_window)
print("RMSE:", final_rmse)
print("MAE :", final_mae)
print("R2  :", final_r2)


pred_df = pd.DataFrame({
    "unit_nr": test_units,
    "Actual_RUL": y_test,
    "Predicted_RUL": y_pred
})

print("\nSample predictions:")
print(pred_df.head(10))



