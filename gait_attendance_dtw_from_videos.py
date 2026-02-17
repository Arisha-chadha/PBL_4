import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
import argparse
from glob import glob
from collections import Counter

# ===================== CONFIG =====================
WINDOW_SECONDS = 1.4
CHECK_EVERY_SECONDS = 0.6
MIN_ROLL_FRAMES = 16

TARGET_LEN = 45
DTW_BAND = 7

# ✅ GaHu dataset-friendly (marks attendance reliably)
DTW_THRESHOLD = 0.42      # was 0.38 (too strict for this dataset)
MIN_GAP = 0.008           # was 0.012 (still safe-ish, but allows marking)
UNKNOWN_NAME = "UNKNOWN"

# ✅ Voting tuned for jittery dataset (2 confident hits in last 7 checks)
VOTE_WINDOW = 7
VOTE_MIN_WINS = 2

COOLDOWN_SECONDS = 4

DB_DIR = "dataset"
ATT_FILE = "attendance.csv"

# DB extraction robustness
MIN_POSE_FRAMES = 16
START_OFFSET_SECONDS = 1.5
# ==================================================

os.makedirs(DB_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, smooth_landmarks=True)
mp_draw = mp.solutions.drawing_utils


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
    SAME 10 features as webcam code + stride_x normalized
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

    stride_x = abs(la[0] - ra[0]) / shoulder_w

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


# ================= VIDEO -> FEATURE SEQ =================
def extract_sequence_from_video(video_path: str, start_seconds=0.0, max_seconds=None, every_n=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    if start_seconds and start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    frames = []
    prev_state = None
    frame_idx = 0
    max_frames = int(max_seconds * fps) if max_seconds is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx % every_n) != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            feats, prev_state = extract_frame_features(res, prev_state)
            frames.append(feats)
        else:
            prev_state = None

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    if len(frames) < MIN_POSE_FRAMES:
        return None

    seq = np.stack(frames, axis=0)
    seq = znorm_rows(resample_sequence(seq, TARGET_LEN))
    return seq


# ================= BUILD DB =================
def build_db_from_folder(
    dataset_root: str,
    pattern="*.avi",
    reset=False,
    segments_per_video=4,
    segment_seconds=10.0,
    segment_stride_seconds=10.0,
    limit_people=None,
    limit_per_person=1,
):
    if reset:
        reset_database()

    video_files = []
    video_files.extend(glob(os.path.join(dataset_root, "**", pattern), recursive=True))
    video_files.extend(glob(os.path.join(dataset_root, "**", pattern.upper()), recursive=True))
    video_files = sorted(set(video_files))

    if not video_files:
        raise RuntimeError(f"No videos found under: {dataset_root} (pattern={pattern})")

    def infer_person_id(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split("_")[0] if "_" in stem else stem

    person_to_videos = {}
    for vf in video_files:
        pid = infer_person_id(vf)
        person_to_videos.setdefault(pid, []).append(vf)

    persons = sorted(person_to_videos.keys())
    if limit_people is not None:
        persons = persons[:limit_people]

    saved = 0
    for pid in persons:
        vids = person_to_videos[pid][:max(1, limit_per_person)]
        seg_global = 0
        for vp in vids:
            for sidx in range(segments_per_video):
                seg_global += 1
                start_t = START_OFFSET_SECONDS + sidx * segment_stride_seconds
                sample_name = f"{pid}_{seg_global}"
                print(f"[DB] {sample_name} <= {vp}  (t={start_t:.1f}s..{start_t+segment_seconds:.1f}s)")

                seq = extract_sequence_from_video(vp, start_seconds=start_t, max_seconds=segment_seconds, every_n=1)
                if seq is None:
                    print("   -> skipped (not enough pose frames)")
                    continue

                save_sample(sample_name, seq)
                saved += 1

    print(f"\nDone. Saved {saved} samples.")
    print("DB samples:", list_db_samples())


# ================= MATCHING (ALL PEOPLE) =================
def decide_identity_person_level(live_seq, person_to_samples, db_sequences):
    """
    Person-level scoring:
    - DTW against each sample for that person
    - take mean of best 2 distances (stable)
    """
    person_scores = []
    for person, samples in person_to_samples.items():
        ds = [dtw_distance(live_seq, db_sequences[s], band=DTW_BAND) for s in samples]
        ds.sort()
        best_for_person = float(np.mean(ds[:2])) if len(ds) >= 2 else float(ds[0])
        person_scores.append((person, best_for_person))

    person_scores.sort(key=lambda x: x[1])
    best_person, best_d = person_scores[0]
    second_d = person_scores[1][1] if len(person_scores) > 1 else 999.0
    gap = second_d - best_d

    confident = (best_d < DTW_THRESHOLD) and (gap > MIN_GAP)
    return best_person, best_d, second_d, gap, confident


# ================= ATTENDANCE =================
def run_attendance_on_video(video_path: str, display=True, max_seconds=None):
    db_samples = list_db_samples()
    if len(db_samples) < 2:
        raise RuntimeError("Build DB first (need at least 2 samples).")

    # group samples by person
    person_to_samples = {}
    for s in db_samples:
        person_to_samples.setdefault(base_name(s), []).append(s)

    # load sequences ONLY for existing DB samples
    db_sequences = {s: load_sample(s) for s in db_samples}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    attendance = []
    last_mark_t = -1e9

    roll = []
    prev_state = None
    last_check_t = -1e9

    # ✅ confidence-only vote buffer
    recent_conf_preds = []

    frame_idx = 0
    max_frames = int(max_seconds * fps) if max_seconds is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            prev_state = None
            roll = []
            recent_conf_preds = []
            label_line = "No pose detected"
        else:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            feats, prev_state = extract_frame_features(res, prev_state)
            roll.append((t, feats))

            cutoff = t - WINDOW_SECONDS
            roll = [(tt, ff) for (tt, ff) in roll if tt >= cutoff]

            label_line = "Collecting gait..."

            if (t - last_check_t) >= CHECK_EVERY_SECONDS and len(roll) >= MIN_ROLL_FRAMES:
                last_check_t = t

                live_seq = np.stack([f for (_, f) in roll], axis=0)
                live_seq = znorm_rows(resample_sequence(live_seq, TARGET_LEN))

                best_person, best_d, second_d, gap, confident = decide_identity_person_level(
                    live_seq, person_to_samples, db_sequences
                )

                print(f"[DBG] best={best_person} d={best_d:.3f} second={second_d:.3f} gap={gap:.3f} confident={confident}")

                # ✅ Vote ONLY on confident hits
                if confident:
                    recent_conf_preds.append(best_person)
                else:
                    recent_conf_preds.append(None)

                if len(recent_conf_preds) > VOTE_WINDOW:
                    recent_conf_preds = recent_conf_preds[-VOTE_WINDOW:]

                valid = [p for p in recent_conf_preds if p is not None]
                winner, wins = None, 0
                if valid:
                    c = Counter(valid)
                    winner, wins = c.most_common(1)[0]

                # ✅ Mark quickly once it gets 2 confident hits in last 7 checks
                if (winner is not None and wins >= VOTE_MIN_WINS and (t - last_mark_t) > COOLDOWN_SECONDS):
                    if not any(a[0] == winner for a in attendance):
                        now_str = datetime.now().strftime("%H:%M:%S")
                        attendance.append([winner, now_str, "Present", round(best_d, 3), round(gap, 3), wins])
                        print("ATTENDANCE MARKED:", winner, "DTW=", round(best_d, 3),
                              "GAP=", round(gap, 3), f"CONF_VOTE={wins}/{VOTE_WINDOW}")

                    last_mark_t = t
                    recent_conf_preds = []

                label_line = f"best={best_person} d={best_d:.2f} gap={gap:.2f} | conf_vote={winner}:{wins}/{VOTE_WINDOW}"

        if display:
            cv2.putText(frame, "DTW Pose-Gait Attendance (ALL PEOPLE)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, label_line, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Gait Attendance (Video)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    df = pd.DataFrame(attendance, columns=["Name", "Time", "Status", "DTW_Distance", "Gap", "ConfVoteWins"])
    df.to_csv(ATT_FILE, index=False)
    print("Attendance saved at:", os.path.abspath(ATT_FILE))
    print(df if len(df) else "No attendance marked.")


# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reset", "build_db", "attendance"], required=True)
    ap.add_argument("--dataset_root", type=str, default=None)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--pattern", type=str, default="*.avi")

    ap.add_argument("--segments_per_video", type=int, default=4)
    ap.add_argument("--segment_seconds", type=float, default=10.0)
    ap.add_argument("--segment_stride_seconds", type=float, default=10.0)

    ap.add_argument("--limit_people", type=int, default=None)
    ap.add_argument("--limit_per_person", type=int, default=1)

    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=None)
    args = ap.parse_args()

    if args.mode == "reset":
        reset_database()
        return

    if args.mode == "build_db":
        if not args.dataset_root:
            raise SystemExit("Provide --dataset_root")
        build_db_from_folder(
            dataset_root=args.dataset_root,
            pattern=args.pattern,
            reset=False,
            segments_per_video=args.segments_per_video,
            segment_seconds=args.segment_seconds,
            segment_stride_seconds=args.segment_stride_seconds,
            limit_people=args.limit_people,
            limit_per_person=args.limit_per_person,
        )
        return

    if args.mode == "attendance":
        if not args.video:
            raise SystemExit("Provide --video")
        run_attendance_on_video(
            video_path=args.video,
            display=(not args.no_display),
            max_seconds=args.max_seconds
        )
        return

if __name__ == "__main__":
    main()
