"""
register.py — Face Registration Module
========================================
Captures multi-angle face samples using MTCNN and trains 
ArcFace embeddings with a beautiful progress display.
"""

import os
import cv2
import numpy as np
import sqlite3
import shutil
import time
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn, TaskProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm

console = Console()

DB_PATH = "database/attendance.db"
FACE_DATA_DIR = "face_data"
MODEL_PATH = "models/embeddings.npy"
SAMPLES_REQUIRED = 80  # number of face samples to capture

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    for d in ["face_data", "models", "database", "exports"]:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            user_code   TEXT UNIQUE NOT NULL,
            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            date     DATE NOT NULL,
            time     TIME NOT NULL,
            status   TEXT DEFAULT 'PRESENT',
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date)
        )
    """)
    conn.commit()
    conn.close()

def user_exists(user_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE user_code = ?", (user_code,))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_user(name, user_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (name, user_code) VALUES (?, ?)", (name, user_code))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id

def delete_user_from_db(user_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM users WHERE user_code = ?", (user_code,))
    row = c.fetchone()
    if row:
        c.execute("DELETE FROM attendance WHERE user_id = ?", (row[0],))
        c.execute("DELETE FROM users WHERE user_code = ?", (user_code,))
        conn.commit()
    conn.close()
    return row

# ─────────────────────────────────────────────
# FACE CAPTURE
# ─────────────────────────────────────────────

def capture_face_samples(name: str, user_code: str, samples_needed: int = SAMPLES_REQUIRED):
    """
    Opens webcam, detects face with MTCNN, saves cropped samples.
    Shows a live progress bar while capturing.
    """
    save_dir = os.path.join(FACE_DATA_DIR, user_code)
    os.makedirs(save_dir, exist_ok=True)

    detector = MTCNN()
    
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): break
        
    if not cap or not cap.isOpened():
        console.print("[bold red]✗ Cannot open webcam on any index![/]")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    console.print(Panel(
        f"[bold cyan]📸 Starting face capture for:[/] [bold white]{name}[/] [dim]({user_code})[/]\n\n"
        f"[yellow]Instructions:[/]\n"
        f"  • Look straight at the camera first\n"
        f"  • Slowly turn your head [cyan]left[/] and [cyan]right[/]\n"
        f"  • Tilt slightly [cyan]up[/] and [cyan]down[/]\n"
        f"  • Keep face well-lit\n\n"
        f"[dim]Press [bold]Q[/bold] to cancel anytime[/]",
        title="[bold blue]Face Registration[/]",
        border_style="blue"
    ))

    input("\n  [Press ENTER to begin capturing...]")

    count = 0
    last_capture_time = 0
    CAPTURE_INTERVAL = 0.12  # seconds between captures

    with Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold blue]Capturing faces..."),
        BarColumn(bar_width=45, style="cyan", complete_style="bold green", finished_style="bold green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("capture", total=samples_needed)

        while count < samples_needed:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            now = time.time()

            # Detect face
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb)

            face_detected = False
            for result in results:
                x, y, w, h = result['box']
                conf = result['confidence']

                # Only accept high-confidence detections
                if conf < 0.92:
                    continue

                face_detected = True
                x, y = max(0, x), max(0, y)

                # Draw bounding box
                color = (0, 255, 100) if now - last_capture_time > CAPTURE_INTERVAL else (0, 200, 255)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                # Draw corner accents
                corner_len = 20
                cv2.line(display, (x, y), (x+corner_len, y), (255,255,255), 3)
                cv2.line(display, (x, y), (x, y+corner_len), (255,255,255), 3)
                cv2.line(display, (x+w, y), (x+w-corner_len, y), (255,255,255), 3)
                cv2.line(display, (x+w, y), (x+w, y+corner_len), (255,255,255), 3)

                # Confidence label
                cv2.putText(display, f"Conf: {conf:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)

                # Capture at interval
                if now - last_capture_time > CAPTURE_INTERVAL:
                    # Add padding to face crop
                    pad = int(0.2 * max(w, h))
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)

                    face_crop = frame[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_crop, (224, 224))

                    img_path = os.path.join(save_dir, f"img_{count:04d}.jpg")
                    cv2.imwrite(img_path, face_resized)

                    count += 1
                    last_capture_time = now
                    progress.update(task, advance=1)

            # Status overlay
            status_color = (0, 255, 100) if face_detected else (0, 100, 255)
            status_text = f"DETECTED ({count}/{samples_needed})" if face_detected else "NO FACE — Position your face"
            cv2.putText(display, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            # Progress bar on frame
            bar_w = int((count / samples_needed) * (frame.shape[1] - 40))
            cv2.rectangle(display, (20, frame.shape[0]-30), (frame.shape[1]-20, frame.shape[0]-10), (50,50,50), -1)
            cv2.rectangle(display, (20, frame.shape[0]-30), (20+bar_w, frame.shape[0]-10), (0,220,100), -1)
            cv2.putText(display, f"{int((count/samples_needed)*100)}%",
                        (frame.shape[1]//2 - 20, frame.shape[0]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow(f"Registering: {name}", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                console.print("[yellow]⚠ Capture cancelled by user.[/]")
                cap.release()
                cv2.destroyAllWindows()
                return False

    cap.release()
    cv2.destroyAllWindows()
    console.print(f"\n[bold green]  ✓ Captured {count} face samples successfully![/]\n")
    return True

# ─────────────────────────────────────────────
# TRAINING — Generate ArcFace Embeddings
# ─────────────────────────────────────────────

def train_model():
    """
    Generates ArcFace face embeddings for all registered users.
    Shows detailed progress with per-user stats.
    """
    console.print(Panel(
        "[bold cyan]🧠 Training Face Recognition Model[/]\n"
        "[dim]Generating ArcFace embeddings for all registered users...[/]",
        border_style="bright_magenta"
    ))

    users = [d for d in os.listdir(FACE_DATA_DIR)
             if os.path.isdir(os.path.join(FACE_DATA_DIR, d))]

    if not users:
        console.print("[red]✗ No face data found. Register users first.[/]")
        return False

    embeddings_db = {}  # { user_code: [embedding_vectors] }
    total_images = sum(
        len([f for f in os.listdir(os.path.join(FACE_DATA_DIR, u)) if f.endswith('.jpg')])
        for u in users
    )

    console.print(f"\n  [cyan]Found:[/] {len(users)} user(s), {total_images} total images\n")

    overall_success = 0
    overall_failed = 0

    with Progress(
        SpinnerColumn(spinner_name="arc", style="bright_magenta"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40, style="magenta", complete_style="bold bright_magenta"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        # Overall task
        overall_task = progress.add_task(
            "[bold white]Overall Training Progress", total=total_images
        )

        for user_code in users:
            user_dir = os.path.join(FACE_DATA_DIR, user_code)
            images = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]

            user_task = progress.add_task(
                f"[cyan]  → {user_code}", total=len(images)
            )

            user_embeddings = []
            success_count = 0
            fail_count = 0

            for img_file in images:
                img_path = os.path.join(user_dir, img_file)
                try:
                    # Extract ArcFace embedding
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        detector_backend="mtcnn",
                        enforce_detection=True,
                        align=True
                    )
                    if result:
                        user_embeddings.append(np.array(result[0]['embedding']))
                        success_count += 1
                        overall_success += 1
                except Exception:
                    # Skip blurry/bad frames
                    fail_count += 1
                    overall_failed += 1

                progress.update(user_task, advance=1)
                progress.update(overall_task, advance=1)

            if user_embeddings:
                embeddings_db[user_code] = user_embeddings
                progress.update(
                    user_task,
                    description=f"[green]  ✓ {user_code} — {success_count} embeddings | {fail_count} skipped"
                )
            else:
                progress.update(
                    user_task,
                    description=f"[red]  ✗ {user_code} — No valid faces found"
                )

    # Save embeddings
    np.save(MODEL_PATH, embeddings_db)

    # Training summary
    accuracy_estimate = (overall_success / max(1, overall_success + overall_failed)) * 100

    summary = Table(box=box.ROUNDED, border_style="bright_green", show_header=True)
    summary.add_column("Metric", style="bold white")
    summary.add_column("Value", style="bold green")

    summary.add_row("Users Trained",         str(len(embeddings_db)))
    summary.add_row("Images Processed",      str(overall_success + overall_failed))
    summary.add_row("Valid Embeddings",       str(overall_success))
    summary.add_row("Rejected (bad frames)", str(overall_failed))
    summary.add_row("Embedding Quality",     f"{accuracy_estimate:.1f}%")
    summary.add_row("Model",                 "ArcFace (512-dim vectors)")
    summary.add_row("Detector",              "MTCNN")
    summary.add_row("Model saved to",        MODEL_PATH)

    console.print("\n")
    console.print(Panel(summary, title="[bold bright_green]✅ Training Complete[/]", border_style="bright_green"))
    console.print("\n")

    return True

# ─────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def register_user():
    init_db()

    console.print("\n[bold blue]═══ Register New User ═══[/]\n")

    name = Prompt.ask("  [cyan]Enter full name[/]").strip()
    if not name:
        console.print("[red]✗ Name cannot be empty.[/]")
        return

    user_code = Prompt.ask("  [cyan]Enter ID / Roll Number[/]").strip().upper()
    if not user_code:
        console.print("[red]✗ ID cannot be empty.[/]")
        return

    if user_exists(user_code):
        console.print(f"[red]✗ User with ID '{user_code}' already exists![/]")
        return

    console.print(f"\n  [white]Registering:[/] [bold cyan]{name}[/] [dim]({user_code})[/]\n")

    # Step 1: Capture samples
    success = capture_face_samples(name, user_code)
    if not success:
        shutil.rmtree(os.path.join(FACE_DATA_DIR, user_code), ignore_errors=True)
        return

    # Step 2: Save to DB
    save_user(name, user_code)
    console.print(f"  [green]✓ User saved to database[/]")

    # Step 3: Retrain model
    console.print(f"\n  [cyan]Retraining model with new user data...[/]\n")
    train_model()

    console.print(f"[bold green]🎉 Registration complete for {name}![/]\n")


def delete_user():
    init_db()
    console.print("\n[bold red]═══ Delete User ═══[/]\n")

    user_code = Prompt.ask("  [cyan]Enter user ID to delete[/]").strip().upper()
    row = delete_user_from_db(user_code)

    if not row:
        console.print(f"[red]✗ No user found with ID '{user_code}'[/]")
        return

    face_dir = os.path.join(FACE_DATA_DIR, user_code)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    console.print(f"[green]✓ Deleted user: {row[1]} ({user_code})[/]")

    # Retrain without deleted user
    if os.path.exists(FACE_DATA_DIR) and os.listdir(FACE_DATA_DIR):
        console.print("\n  [cyan]Retraining model...[/]\n")
        train_model()
    else:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        console.print("[yellow]  No users remaining — model cleared.[/]")
