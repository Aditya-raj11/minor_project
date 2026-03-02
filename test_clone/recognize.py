"""
recognize.py — Real-Time Face Recognition & Attendance
========================================================
Uses ArcFace embeddings + cosine similarity for identification.
Marks attendance automatically with duplicate prevention.
"""

import os
import cv2
import numpy as np
import sqlite3
import time
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine

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
RECOGNITION_THRESHOLD = 0.52   # below this = recognized
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

def identify_face(face_img, embeddings_db):
    """
    Given a face image, compute ArcFace embedding and find best match.
    Returns (user_code, distance, confidence_pct) or (None, 1.0, 0)
    """
    try:
        tmp_path = "/tmp/face_tmp.jpg"
        cv2.imwrite(tmp_path, face_img)

        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="ArcFace",
            detector_backend="skip",   # already cropped
            enforce_detection=False,
            align=False
        )
        if not result:
            return None, 1.0, 0

        query_embedding = np.array(result[0]['embedding'])

        best_match = None
        best_dist = 1.0

        for user_code, stored_embeddings in embeddings_db.items():
            for stored_emb in stored_embeddings:
                dist = cosine(query_embedding, stored_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_match = user_code

        confidence = max(0, (1 - best_dist / RECOGNITION_THRESHOLD)) * 100
        confidence = min(confidence, 99.9)

        if best_dist < RECOGNITION_THRESHOLD:
            return best_match, best_dist, confidence
        return None, best_dist, 0

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
# MAIN RECOGNITION LOOP
# ─────────────────────────────────────────────

def start_attendance():
    console.print("\n[bold blue]═══ Starting Attendance Session ═══[/]\n")

    embeddings_db = load_embeddings()
    if not embeddings_db:
        console.print("[bold red]✗ No trained model found! Register users and train first.[/]")
        return

    users = get_all_users()
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Session state
    marked_today = set()
    session_log = []          # list of (name, user_code, time_str)
    last_recognition = {}     # user_code → timestamp, to avoid spam
    COOLDOWN = 3.0            # seconds between recognition attempts per face

    # Pre-populate already marked today
    for name, code, t in get_today_attendance():
        marked_today.add(code)

    frame_count = 0
    fps_start = time.time()
    fps = 0

    console.print("[bold green]  📷 Webcam running... Looking for faces[/]\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detect faces ──────────────────────────────
        results = detector.detect_faces(rgb)

        for result in results:
            x, y, w, h = result['box']
            det_conf = result['confidence']

            if det_conf < CONFIDENCE_THRESHOLD:
                continue

            x, y = max(0, x), max(0, y)
            pad = int(0.15 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (224, 224))

            # ── Identify ──────────────────────────────
            now = time.time()
            user_code, dist, conf_pct = identify_face(face_resized, embeddings_db)

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
