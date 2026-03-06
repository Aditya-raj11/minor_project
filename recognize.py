"""
recognize.py — Real-Time Face Recognition & Attendance
========================================================
Uses ArcFace embeddings + cosine similarity for identification.
Marks attendance automatically with duplicate prevention.
"""

import os
import multiprocessing
import threading
import queue

# ── Pin ALL thread pools to physical core count BEFORE TF/DeepFace loads ──
CPU_COUNT = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"]     = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["TF_NUM_INTRAOP_THREADS"]   = str(CPU_COUNT)
os.environ["TF_NUM_INTEROP_THREADS"]   = str(CPU_COUNT)

import cv2
import numpy as np
import sqlite3
import time
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine
import tempfile
import math

# Tell OpenCV to use all cores too
cv2.setNumThreads(CPU_COUNT)

def _init_env():
    for d in ["face_data", "models", "database", "exports"]:
        os.makedirs(d, exist_ok=True)

_init_env()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()

DB_PATH = "database/attendance.db"
MODEL_PATH = "models/embeddings.npy"

# Tuned thresholds for ArcFace (cosine distance — LOWER = more similar)
RECOGNITION_THRESHOLD = 0.40   # below this = recognized (tighter = fewer false positives)
MARGIN_THRESHOLD      = 0.08   # best match must beat 2nd-best by at least this margin
CONFIDENCE_THRESHOLD  = 0.92   # MTCNN detection confidence

# ─────────────────────────────────────────────
# LOAD MODEL & USER DATA
# ─────────────────────────────────────────────

def load_embeddings():
    if not os.path.exists(MODEL_PATH):
        return None
    return np.load(MODEL_PATH, allow_pickle=True).item()

def get_user_name(user_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM users WHERE user_code = ?", (user_code,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else user_code

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_code, name FROM users")
    rows = c.fetchall()
    conn.close()
    return {code: name for code, name in rows}

# ─────────────────────────────────────────────
# RECOGNITION LOGIC
# ─────────────────────────────────────────────

def _build_mean_embeddings(embeddings_db):
    """
    Pre-compute a single mean embedding vector per user.
    Averaging over all samples makes the representative far more stable
    and prevents a single bad (e.g. glasses-heavy) frame from dominating.
    """
    mean_db = {}
    for user_code, embs in embeddings_db.items():
        arr = np.array(embs)          # shape: (N, 512)
        mean_vec = arr.mean(axis=0)   # shape: (512,)
        # L2-normalise so cosine distance is meaningful
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        mean_db[user_code] = mean_vec
    return mean_db

# Public alias so gui.py and other modules can import without the underscore
build_mean_embeddings = _build_mean_embeddings


def mask_glasses_region(face_img, left_eye, right_eye):
    """
    Given a 224×224 face image and the pixel coordinates of both eyes
    (already transformed into that image's coordinate space), draw an
    elliptical mask over the glasses / eye-frame region and fill it with
    the surrounding skin tone.

    This removes spectacle-frame features BEFORE ArcFace sees the image,
    so two people wearing the same glasses will no longer produce similar
    embeddings.
    """
    masked = face_img.copy()
    h, w  = masked.shape[:2]

    lx, ly = int(left_eye[0]),  int(left_eye[1])
    rx, ry = int(right_eye[0]), int(right_eye[1])

    # Clamp to image bounds
    lx = max(2, min(w - 2, lx))
    rx = max(2, min(w - 2, rx))
    ly = max(2, min(h - 2, ly))
    ry = max(2, min(h - 2, ry))

    eye_dist = max(1, np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2))

    # Ellipse centre: midpoint between eyes, nudged slightly upward
    cx = (lx + rx) // 2
    cy = int((ly + ry) / 2 - eye_dist * 0.05)

    # Axes: wide to cover full frame width, tall to cover frame thickness
    half_w = int(eye_dist * 0.72)
    half_h = int(eye_dist * 0.28)
    half_w = max(half_w, 4)
    half_h = max(half_h, 4)

    # Sample fill colour from the CHEEK area (below the glasses) for natural look
    cheek_y = min(h - 1, ry + int(eye_dist * 0.45))
    cheek_x = cx
    sample_y1 = max(0, cheek_y - 5)
    sample_y2 = min(h,  cheek_y + 5)
    sample_x1 = max(0, cheek_x - 5)
    sample_x2 = min(w,  cheek_x + 5)
    skin_patch = masked[sample_y1:sample_y2, sample_x1:sample_x2]
    if skin_patch.size > 0:
        fill_color = tuple(int(c) for c in cv2.mean(skin_patch)[:3])
    else:
        fill_color = (180, 150, 130)   # fallback neutral skin tone

    # Draw the ellipse mask and fill
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, (cx, cy), (half_w, half_h), 0, 0, 360, 255, -1)
    masked[ellipse_mask == 255] = fill_color

    return masked



# ─────────────────────────────────────────────
# FACE VALIDATION HELPERS  (used by gui.py WebcamEngine + _reg_thread)
# ─────────────────────────────────────────────

_CONF_THRESHOLD = 0.97
_BLUR_THRESHOLD = 80

def is_valid_face_shape(x, y, w, h, frame_shape):
    fh, fw = frame_shape[:2]
    if w < 70 or h < 70:                     return False, "too small"
    if w > int(fh * 0.85):                   return False, "too large"
    aspect = w / max(h, 1)
    if aspect < 0.60 or aspect > 1.55:       return False, "bad aspect"
    margin = 10
    if x < margin or y < margin:             return False, "edge"
    if (x+w) > (fw-margin) or (y+h) > (fh-margin): return False, "edge"
    return True, "ok"

def has_valid_landmarks(result):
    kp     = result.get("keypoints", {})
    needed = ["left_eye","right_eye","nose","mouth_left","mouth_right"]
    if not all(k in kp for k in needed): return False, "no landmarks"
    le  = np.array(kp["left_eye"],   dtype=float)
    re  = np.array(kp["right_eye"],  dtype=float)
    nos = np.array(kp["nose"],       dtype=float)
    ml  = np.array(kp["mouth_left"], dtype=float)
    mr  = np.array(kp["mouth_right"],dtype=float)
    angle = abs(math.degrees(math.atan2(re[1]-le[1], re[0]-le[0])))
    if angle > 45:                            return False, "tilt"
    eye_cy = (le[1] + re[1]) / 2
    if nos[1] <= eye_cy:                      return False, "nose up"
    if (ml[1]+mr[1])/2 <= nos[1]:            return False, "mouth up"
    x, y, w, h = result["box"]
    ed = float(np.linalg.norm(re - le))
    if ed < w*0.18 or ed > w*0.85:           return False, "eye dist"
    mw = float(np.linalg.norm(mr - ml))
    if mw < 10:                               return False, "mouth"
    eto = nos[1] - eye_cy
    ntm = (ml[1]+mr[1])/2 - nos[1]
    if eto < 3 or ntm < 3:                   return False, "flat"
    r = eto / max(ntm, 1)
    if r < 0.2 or r > 4.5:                   return False, "ratio"
    return True, "ok"

def is_sharp_enough(face_bgr, threshold=_BLUR_THRESHOLD):
    gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold, float(score)

def detect_valid_faces(frame, detector, conf_thr=_CONF_THRESHOLD):
    """
    Run MTCNN, then apply 4-layer validation filter.
    Returns (valid_list, rejected_list).
    valid_list items: (result, face_224, x, y, w, h, conf)
    """
    if frame is None or frame.size == 0:
        return [], []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = detector.detect_faces(rgb)
    except Exception as e:
        print(f"[WARN] MTCNN Error: {e}")
        return [], []
    valid, rejected = [], []
    for r in results:
        x, y, w, h = r["box"]
        conf        = r["confidence"]
        x, y = max(0, x), max(0, y)
        if conf < conf_thr:                  rejected.append((x,y,w,h)); continue
        ok, _=is_valid_face_shape(x,y,w,h,frame.shape)
        if not ok:                           rejected.append((x,y,w,h)); continue
        ok, _=has_valid_landmarks(r)
        if not ok:                           rejected.append((x,y,w,h)); continue
        pad = int(0.12*max(w,h))
        x1=max(0,x-pad); y1=max(0,y-pad)
        x2=min(frame.shape[1],x+w+pad); y2=min(frame.shape[0],y+h+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:                   rejected.append((x,y,w,h)); continue
        ok, _=is_sharp_enough(crop)
        if not ok:                           rejected.append((x,y,w,h)); continue
        valid.append((r, cv2.resize(crop,(224,224)), x, y, w, h, conf))
    return valid, rejected


def identify_face(face_img, embeddings_db, mean_db=None):
    """
    Given a face image, compute ArcFace embedding and find best match.

    Strategy (spectacles-robust):
      1. Compare query against the MEAN embedding of each user (not every
         individual sample).  Averaging suppresses appearance-specific
         artefacts like glasses frames.
      2. Require the best match to be within RECOGNITION_THRESHOLD.
      3. Require the best match to beat the 2nd-best by at least
         MARGIN_THRESHOLD — if two users look similar (e.g. same glasses),
         neither is reported and the face is returned as UNKNOWN.

    Returns (user_code, distance, confidence_pct) or (None, 1.0, 0)
    """
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "face_tmp.jpg")
        cv2.imwrite(tmp_path, face_img)

        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="ArcFace",
            detector_backend="skip",   # already cropped by MTCNN
            enforce_detection=False,
            align=True                 # align=True normalises pose → glasses matter less
        )
        if not result:
            return None, 1.0, 0

        query_embedding = np.array(result[0]['embedding'])
        # L2-normalise query
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding = query_embedding / q_norm

        # Use pre-built mean-embedding lookup if available (much faster in live loop)
        if mean_db is None:
            mean_db = _build_mean_embeddings(embeddings_db)

        # --- Compare against every user's mean vector ---
        distances = {}   # user_code → cosine distance
        for user_code, mean_emb in mean_db.items():
            distances[user_code] = cosine(query_embedding, mean_emb)

        # Sort by distance ascending
        sorted_users = sorted(distances.items(), key=lambda x: x[1])

        best_match, best_dist = sorted_users[0]

        # ── Gate 1: absolute threshold ────────────────────────────────────
        if best_dist >= RECOGNITION_THRESHOLD:
            return None, best_dist, 0

        # ── Gate 2: margin check  ─────────────────────────────────────────
        # If there's a 2nd candidate, verify we're sufficiently separated.
        # This catches the "same glasses" case where two people sit too
        # close together in embedding space.
        if len(sorted_users) > 1:
            second_dist = sorted_users[1][1]
            margin = second_dist - best_dist
            if margin < MARGIN_THRESHOLD:
                # Too ambiguous — report unknown rather than guess wrong
                return None, best_dist, 0

        confidence = max(0, (1 - best_dist / RECOGNITION_THRESHOLD)) * 100
        confidence = min(confidence, 99.9)

        return best_match, best_dist, confidence

    except Exception:
        return None, 1.0, 0

# ─────────────────────────────────────────────
# ATTENDANCE MARKING
# ─────────────────────────────────────────────

def mark_attendance(user_code):
    """
    Mark attendance for today. Returns True if marked, False if already marked.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE user_code = ?", (user_code,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "User not found"

    user_id = row[0]
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    try:
        c.execute(
            "INSERT INTO attendance (user_id, date, time, status) VALUES (?, ?, ?, 'PRESENT')",
            (user_id, today, now_time)
        )
        conn.commit()
        conn.close()
        return True, "Marked"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Already marked today"

def get_today_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
        SELECT u.name, u.user_code, a.time
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (today,))
    rows = c.fetchall()
    conn.close()
    return rows

# ─────────────────────────────────────────────
# CAMERA READER THREAD  (producer)
# ─────────────────────────────────────────────

class CameraReader(threading.Thread):
    """
    Continuously reads frames from the webcam in a background thread.
    The main thread pulls the latest frame from the queue without waiting
    on camera I/O, so MTCNN + ArcFace inference never stalls on cap.read().
    """
    def __init__(self, cap, maxsize=2):
        super().__init__(daemon=True)
        self.cap   = cap
        self.queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Drop stale frames — always keep the freshest one
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        """Block until a frame is available, return (True, frame)."""
        try:
            return True, self.queue.get(timeout=2.0)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()



# ─────────────────────────────────────────────
# MAIN RECOGNITION LOOP
# ─────────────────────────────────────────────

def start_attendance():
    console.print("\n[bold blue]═══ Starting Attendance Session ═══[/]\n")

    embeddings_db = load_embeddings()
    if not embeddings_db:
        console.print("[bold red]✗ No trained model found! Register users and train first.[/]")
        return

    users = get_all_users()

    # Pre-build mean embeddings ONCE for the whole session (spectacles-robust)
    mean_db = _build_mean_embeddings(embeddings_db)

    console.print(f"  [green]✓ Model loaded:[/] {len(embeddings_db)} registered user(s)")
    console.print(f"  [cyan]ℹ[/] [dim]Press [bold]Q[/bold] to stop the session[/]\n")

    detector = MTCNN()

    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): break

    if not cap or not cap.isOpened():
        console.print("[bold red]✗ Cannot open webcam on any index![/]")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)   # minimise internal camera buffer lag

    # ── Start background camera reader ────────────────────────────────────
    reader = CameraReader(cap)
    reader.start()
    console.print(f"  [dim cyan]ℹ Using {CPU_COUNT} CPU core(s) for inference[/]\n")

    # Session state
    marked_today     = set()
    session_log      = []          # list of (name, user_code, time_str)
    last_recognition = {}          # user_code → timestamp, to avoid spam
    COOLDOWN         = 3.0         # seconds between recognition attempts per face

    # Pre-populate already marked today
    for name, code, t in get_today_attendance():
        marked_today.add(code)

    frame_count = 0
    fps_start   = time.time()
    fps         = 0

    console.print("[bold green]  📷 Webcam running... Looking for faces[/]\n")

    while True:
        ret, frame = reader.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detect faces (MTCNN, full resolution) ──────────────────────
        results = detector.detect_faces(rgb)

        # Filter low-confidence detections and pre-process all face crops
        valid_results = []
        face_imgs     = []
        for result in results:
            x, y, w, h   = result['box']
            det_conf     = result['confidence']
            if det_conf < CONFIDENCE_THRESHOLD:
                continue
            x, y = max(0, x), max(0, y)
            pad  = int(0.15 * max(w, h))
            x1   = max(0, x - pad)
            y1   = max(0, y - pad)
            x2   = min(frame.shape[1], x + w + pad)
            y2   = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, (224, 224))
            # Glasses masking
            kp     = result.get('keypoints', {})
            crop_w = max(1, x2 - x1)
            crop_h = max(1, y2 - y1)
            if 'left_eye' in kp and 'right_eye' in kp:
                le = ((kp['left_eye'][0]  - x1) * 224.0 / crop_w,
                      (kp['left_eye'][1]  - y1) * 224.0 / crop_h)
                re = ((kp['right_eye'][0] - x1) * 224.0 / crop_w,
                      (kp['right_eye'][1] - y1) * 224.0 / crop_h)
                face_resized = mask_glasses_region(face_resized, le, re)
            valid_results.append((result, x, y, w, h))
            face_imgs.append(face_resized)

        # ── Identify faces one by one ───────────────────────────────
        now = time.time()
        for i, (result, x, y, w, h) in enumerate(valid_results):
            user_code, dist, conf_pct = identify_face(
                face_imgs[i], embeddings_db, mean_db
            )

            label = "UNKNOWN"
            box_color = (0, 60, 220)   # blue for unknown
            sub_label = f"Dist: {dist:.3f}"

            if user_code:
                name = users.get(user_code, user_code)
                last_seen = last_recognition.get(user_code, 0)

                if now - last_seen > COOLDOWN:
                    last_recognition[user_code] = now
                    marked, msg = mark_attendance(user_code)

                    if marked:
                        marked_today.add(user_code)
                        session_log.append((name, user_code, datetime.now().strftime("%H:%M:%S")))
                        console.print(
                            f"  [bold green]✅ MARKED:[/] [white]{name}[/] "
                            f"[dim]({user_code})[/] — [cyan]{datetime.now().strftime('%H:%M:%S')}[/]"
                        )

                if user_code in marked_today:
                    box_color = (0, 220, 80)   # green = present
                    label = f"{name}"
                    sub_label = f"✓ Present  {conf_pct:.1f}%"
                else:
                    box_color = (0, 200, 255)  # yellow = recognized but not yet marked
                    label = f"{name}"
                    sub_label = f"Conf: {conf_pct:.1f}%"
            else:
                sub_label = f"Unknown  dist:{dist:.3f}"

            # ── Draw UI on frame ──────────────────────
            # Bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 2)

            # Corner accents
            cl = 18
            for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                cv2.line(display, (px, py), (px + dx*cl, py), (255,255,255), 3)
                cv2.line(display, (px, py), (px, py + dy*cl), (255,255,255), 3)

            # Label background
            label_y = max(y - 10, 30)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(display, (x, label_y - lh - 8), (x + lw + 10, label_y + 4), box_color, -1)
            cv2.putText(display, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(display, sub_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # ── HUD ───────────────────────────────────────
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]

        # Top bar
        cv2.rectangle(display, (0, 0), (w_frame, 55), (15, 15, 15), -1)
        cv2.putText(display, "SMART ATTENDANCE SYSTEM", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        cv2.putText(display, datetime.now().strftime("%d %b %Y  %H:%M:%S"),
                    (w_frame - 320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Bottom bar
        cv2.rectangle(display, (0, h_frame - 45), (w_frame, h_frame), (15, 15, 15), -1)
        cv2.putText(display, f"FPS: {fps:.1f}", (20, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)
        cv2.putText(display, f"Present today: {len(marked_today)}  |  Registered: {len(users)}",
                    (130, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, "Press Q to stop", (w_frame - 200, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

        cv2.imshow("Smart Attendance System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()

    # Session summary
    console.print("\n")
    if session_log:
        table = Table(title="📋 Session Attendance Log", box=box.ROUNDED, border_style="green")
        table.add_column("Name", style="bold white")
        table.add_column("ID", style="cyan")
        table.add_column("Time Marked", style="green")

        for name, code, t in session_log:
            table.add_row(name, code, t)

        console.print(table)
    else:
        console.print("[yellow]  No new attendance marked in this session.[/]")

    console.print(f"\n  [bold green]Session ended.[/] Total present today: [cyan]{len(marked_today)}[/]\n")
