"""
FAST + ACCURATE DTW Pose-Gait Attendance (Demo Final)

Improvements vs your current version:
- Much faster recognition using rolling/sliding window (no need to walk many times)
- Person-level matching using multiple samples (Arisha_1, Arisha_2 -> person "Arisha")
- Still safe: DTW threshold + gap check + stable confirmations
- Built-in RESET database option
- Privacy-preserving: stores ONLY pose-gait features, no face, no raw video saved

Run:
    python gait_attendance_dtw_final.py

Keys:
    ESC -> quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import shutil
from datetime import datetime

# ===================== CONFIG =====================
TIME_LIMIT_SECONDS = 10 * 60
REGISTER_SECONDS = 8

# ---------- FAST recognition ----------
# Sliding window: evaluate every CHECK_EVERY_SECONDS using last WINDOW_SECONDS of frames.
WINDOW_SECONDS = 1.4              # collect ~1.4s gait window
CHECK_EVERY_SECONDS = 0.6         # run recognition every ~0.6s
MIN_ROLL_FRAMES = 16              # minimum frames required inside window

# DTW speed
TARGET_LEN = 45                   # resample length for DTW
DTW_BAND = 7                      # narrower band => faster

# Safety / accuracy
DTW_THRESHOLD = 0.44              # if too many UNKNOWNs -> 0.46
MIN_GAP = 0.07                    # if wrong matches -> 0.09
STABLE_DECISIONS = 2              # need same person 2 times in a row

COOLDOWN_SECONDS = 2
LEAVE_RESET_FRAMES = 12
UNKNOWN_NAME = "UNKNOWN"

DB_DIR = "gait_db"
ATT_FILE = "attendance.csv"
# ==================================================

os.makedirs(DB_DIR, exist_ok=True)

# ---------- MediaPipe (faster) ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, smooth_landmarks=True)  # 0 = fastest
mp_draw = mp.solutions.drawing_utils

# ---------- Camera (speed) ----------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not cap.isOpened():
    print("Camera not opened. Close Zoom/Teams/Chrome camera tabs and retry.")
    raise SystemExit


# ================= GAIT FEATURES (10 per frame) =================
def _xy(res, idx: int) -> np.ndarray:
    lm = res.pose_landmarks.landmark[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.arccos(cosv))

def extract_frame_features(res, prev_state):
    """
    Privacy-preserving gait features (no face / no raw storage)
    10 features:
    1 l_knee, 2 r_knee, 3 l_hip, 4 r_hip,
    5 ankle_motion, 6 hip_motion, 7 ankle_dist,
    8 torso_lean, 9 knee_sym, 10 stride_x
    """
    ls = _xy(res, 11); rs = _xy(res, 12)
    lh = _xy(res, 23); rh = _xy(res, 24)
    lk = _xy(res, 25); rk = _xy(res, 26)
    la = _xy(res, 27); ra = _xy(res, 28)

    shoulder_w = np.linalg.norm(ls - rs) + 1e-9

    l_knee = _angle(lh, lk, la)
    r_knee = _angle(rh, rk, ra)
    l_hip  = _angle(ls, lh, lk)
    r_hip  = _angle(rs, rh, rk)

    ankle_motion = 0.0
    hip_motion = 0.0
    hip_mid = (lh + rh) / 2.0

    if prev_state is not None:
        prev_la, prev_ra, prev_hip_mid = prev_state
        ankle_motion = (np.linalg.norm(la - prev_la) + np.linalg.norm(ra - prev_ra)) / (2.0 * shoulder_w)
        hip_motion = np.linalg.norm(hip_mid - prev_hip_mid) / shoulder_w

    ankle_dist = np.linalg.norm(la - ra) / shoulder_w
    shoulder_mid = (ls + rs) / 2.0
    torso_vec = shoulder_mid - hip_mid
    torso_lean = float(np.arctan2(torso_vec[0], torso_vec[1] + 1e-9))
    knee_sym = abs(l_knee - r_knee)
    stride_x = abs(la[0] - ra[0])

    feats = np.array([
        l_knee, r_knee,
        l_hip, r_hip,
        ankle_motion, hip_motion,
        ankle_dist, torso_lean,
        knee_sym, stride_x
    ], dtype=np.float32)

    return feats, (la, ra, hip_mid)


# ================= SEQUENCE UTILS =================
def list_db_samples():
    return sorted([fn[:-4] for fn in os.listdir(DB_DIR) if fn.endswith(".npz")])

def base_name(sample):
    return sample.split("_")[0] if "_" in sample else sample

def resample_sequence(seq: np.ndarray, target_len: int):
    T, D = seq.shape
    if T == target_len:
        return seq.astype(np.float32)
    if T < 2:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, seq[:, d])
    return out

def znorm_rows(seq: np.ndarray):
    mu = seq.mean(axis=0, keepdims=True)
    sd = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mu) / sd

def cosine_dist(u, v):
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-9
    return float(1.0 - (np.dot(u, v) / denom))

def dtw_distance(A, B, band=7):
    T = A.shape[0]
    INF = 1e9
    dp = np.full((T + 1, T + 1), INF, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, T + 1):
        j_start = max(1, i - band)
        j_end = min(T, i + band)
        for j in range(j_start, j_end + 1):
            cost = cosine_dist(A[i - 1], B[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[T, T] / (2.0 * T))

def save_sample(name, seq):
    np.savez_compressed(os.path.join(DB_DIR, f"{name}.npz"), seq=seq.astype(np.float32))

def load_sample(name):
    data = np.load(os.path.join(DB_DIR, f"{name}.npz"))
    return data["seq"].astype(np.float32)

def reset_database():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(ATT_FILE):
        os.remove(ATT_FILE)
    print("Database cleared.")


# ================= MODE =================
print("\n===== GAIT ATTENDANCE (DTW) =====")
print("1) Register new sample")
print("2) Attendance mode")
print("3) RESET database (delete all samples)")
mode = input("Choose (1/2/3): ").strip()

if mode == "3":
    reset_database()
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

register = (mode == "1")

# ================= REGISTRATION =================
if register:
    sample_name = input("Enter sample name (Arisha_1, Arisha_2, Friend_1...): ").strip()
    print(f"Registering '{sample_name}' for {REGISTER_SECONDS}s.")
    print("Walk SIDEWAYS. Full body visible. 6–10 steps. Normal speed.")

    frames = []
    prev_state = None
    t0 = time.time()

    while time.time() - t0 < REGISTER_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            feats, prev_state = extract_frame_features(res, prev_state)
            frames.append(feats)

        cv2.putText(frame, f"REGISTERING: {sample_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Gait Register", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < 30:
        print("Not enough frames. Improve lighting + full body + walk longer.")
        raise SystemExit

    seq = znorm_rows(resample_sequence(np.stack(frames, axis=0), TARGET_LEN))
    save_sample(sample_name, seq)

    print("Saved:", sample_name)
    print("DB samples:", list_db_samples())
    raise SystemExit


# ================= ATTENDANCE =================
db_samples = list_db_samples()
if len(db_samples) < 2:
    print("Register at least 2 samples first.")
    raise SystemExit

# Group samples by person
person_to_samples = {}
for s in db_samples:
    person_to_samples.setdefault(base_name(s), []).append(s)

db_sequences = {s: load_sample(s) for s in db_samples}

print("\nAttendance mode ON (10 minutes).")
print("Walk sideways for ~2–3 seconds (it checks continuously).")
print("Press ESC to stop.\n")

start_time = time.time()
attendance = []

locked = False
no_pose_count = 0
last_mark_time = 0

candidate = None
candidate_decisions = 0

roll = []                # list of (timestamp, feature_vector)
prev_state = None
last_check_time = 0

def decide_identity_person_level(live_seq):
    """
    Person-level matching:
    For each person, take min DTW distance over their multiple samples (Arisha_1, Arisha_2).
    Then compute best person, second person, and gap.
    """
    person_scores = []
    for person, samples in person_to_samples.items():
        best_for_person = 999.0
        for s in samples:
            d = dtw_distance(live_seq, db_sequences[s], band=DTW_BAND)
            if d < best_for_person:
                best_for_person = d
        person_scores.append((person, best_for_person))

    person_scores.sort(key=lambda x: x[1])
    best_person, best_d = person_scores[0]
    second_d = person_scores[1][1] if len(person_scores) > 1 else 999.0
    gap = second_d - best_d

    confident = (best_d < DTW_THRESHOLD) and (gap > MIN_GAP)
    return best_person, best_d, second_d, gap, confident

while True:
    if time.time() - start_time > TIME_LIMIT_SECONDS:
        print("Time Over. Attendance closed.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    # No pose detected
    if not res.pose_landmarks:
        no_pose_count += 1
        prev_state = None
        roll = []
        candidate, candidate_decisions = None, 0

        if no_pose_count >= LEAVE_RESET_FRAMES:
            locked = False

        cv2.putText(frame, "No pose detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Gait Attendance", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    no_pose_count = 0
    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    feats, prev_state = extract_frame_features(res, prev_state)
    now = time.time()
    roll.append((now, feats))

    # Keep only last WINDOW_SECONDS
    cutoff = now - WINDOW_SECONDS
    roll = [(t, f) for (t, f) in roll if t >= cutoff]

    label_line = "Collecting gait..."

    # Check every CHECK_EVERY_SECONDS
    if (now - last_check_time) >= CHECK_EVERY_SECONDS and len(roll) >= MIN_ROLL_FRAMES:
        last_check_time = now

        live_seq = np.stack([f for (_, f) in roll], axis=0)
        live_seq = znorm_rows(resample_sequence(live_seq, TARGET_LEN))

        best_person, best_d, second_d, gap, confident = decide_identity_person_level(live_seq)
        pred = best_person if confident else UNKNOWN_NAME

        # Stability logic (prevents random single-frame matches)
        if confident and not locked:
            if candidate == pred:
                candidate_decisions += 1
            else:
                candidate = pred
                candidate_decisions = 1
        else:
            candidate = None
            candidate_decisions = 0

        # Mark attendance
        if (candidate is not None and candidate_decisions >= STABLE_DECISIONS
                and (now - last_mark_time) > COOLDOWN_SECONDS):

            if not any(a[0] == candidate for a in attendance):
                now_str = datetime.now().strftime("%H:%M:%S")
                attendance.append([candidate, now_str, "Present", round(best_d, 3), round(gap, 3)])
                print("ATTENDANCE MARKED:", candidate, "DTW=", round(best_d, 3), "GAP=", round(gap, 3))

            locked = True
            last_mark_time = now
            candidate, candidate_decisions = None, 0

        if confident:
            label_line = f"Match: {best_person} dtw={best_d:.2f} gap={gap:.2f} [{candidate_decisions}/{STABLE_DECISIONS}]"
        else:
            label_line = f"UNKNOWN dtw={best_d:.2f} gap={gap:.2f}"

    cv2.putText(frame, "DTW Pose-Gait Attendance | Privacy Preserving", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, label_line, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(frame, f"Locked={locked} (leave frame to unlock)", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gait Attendance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(attendance, columns=["Name", "Time", "Status", "DTW_Distance", "Gap"])
df.to_csv(ATT_FILE, index=False)

print("Attendance saved at:", os.path.abspath(ATT_FILE))
print(df if len(df) else "No attendance marked.")
