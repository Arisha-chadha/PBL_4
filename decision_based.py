
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

def classify_health(rul):
    if rul > 80:
        return "Healthy"
    elif rul > 30:
        return "Warning"
    else:
        return "Replace"

def create_train_windows_with_health(df, features, window):
    X = []
    y_rul = []
    y_health = []
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

            rul_target = unit_df["RUL_clipped"].iloc[i]
            health_target = classify_health(rul_target)

            X.append(window_data)
            y_rul.append(rul_target)
            y_health.append(health_target)
            unit_ids.append(unit)

    return np.array(X), np.array(y_rul), np.array(y_health), np.array(unit_ids)


def create_test_last_windows_with_health(df, features, window):
    X = []
    y_rul = []
    y_health = []
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

        rul_target = unit_df["RUL_clipped"].iloc[-1]
        health_target = classify_health(rul_target)

        X.append(window_data)
        y_rul.append(rul_target)
        y_health.append(health_target)
        unit_ids.append(unit)

    return np.array(X), np.array(y_rul), np.array(y_health), np.array(unit_ids)

WINDOW = 24   # use your best window

X_train_cls, y_train_rul, y_train_health, train_units_cls = create_train_windows_with_health(
    train_df, feature_cols, WINDOW
)

X_test_cls, y_test_rul, y_test_health, test_units_cls = create_test_last_windows_with_health(
    test_df, feature_cols, WINDOW
)

print("Classification train shape:", X_train_cls.shape)
print("Classification test shape :", X_test_cls.shape)

from sklearn.preprocessing import StandardScaler

scaler_cls = StandardScaler()
X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)


label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train_health)
y_test_enc = label_encoder.transform(y_test_health)

print("Classes:", label_encoder.classes_)

base_clf = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Calibrate probabilities for better confidence values
clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
clf.fit(X_train_cls, y_train_enc)


y_pred_enc = clf.predict(X_test_cls)
y_pred = label_encoder.inverse_transform(y_pred_enc)

probs = clf.predict_proba(X_test_cls)
confidence = probs.max(axis=1)

print("\nLAST-CYCLE HEALTH CLASSIFICATION REPORT")
print(classification_report(y_test_health, y_pred))

print("Accuracy:", accuracy_score(y_test_health, y_pred))
print("Macro F1:", f1_score(y_test_health, y_pred, average="macro"))
print("Weighted F1:", f1_score(y_test_health, y_pred, average="weighted"))


cm = confusion_matrix(y_test_health, y_pred, labels=["Healthy", "Warning", "Replace"])
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Health Classification Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Healthy", "Warning", "Replace"])
plt.yticks(tick_marks, ["Healthy", "Warning", "Replace"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

def decision_rule(pred_label, conf):
    if conf < 0.60:
        return "Inspect"
    elif pred_label == "Replace":
        return "Replace Immediately"
    elif pred_label == "Warning":
        return "Monitor Closely"
    else:
        return "Normal Operation"

output_df = pd.DataFrame({
    "unit_nr": test_units_cls,
    "actual_health": y_test_health,
    "predicted_health": y_pred,
    "confidence": confidence,
    "actual_rul": y_test_rul
})

output_df["decision"] = [
    decision_rule(p, c) for p, c in zip(output_df["predicted_health"], output_df["confidence"])
]

print("\nSample decision output:")
print(output_df.head(10))


plt.figure(figsize=(6, 4))
plt.hist(confidence, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()

mask = output_df["actual_health"].isin(["Warning", "Replace"])
early_warning_acc = (output_df.loc[mask, "predicted_health"] == output_df.loc[mask, "actual_health"]).mean()

print("Early warning accuracy:", early_warning_acc)


