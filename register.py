"""
register.py — Training Module (GUI-only)
==========================================
Generates ArcFace embeddings for all registered face data.
All user interaction (capture, registration, deletion) runs through gui.py.
"""

import os
import numpy as np
from deepface import DeepFace

FACE_DATA_DIR = "face_data"
MODEL_PATH    = "models/embeddings.npy"


def train_model_with_callback(progress_cb=None):
    """
    Generate ArcFace embeddings for every registered user.

    progress_cb(pct, acc, success, fail) is called after every image so
    gui.py can update its progress bar.  align=True matches recognition.
    """
    users = [d for d in os.listdir(FACE_DATA_DIR)
             if os.path.isdir(os.path.join(FACE_DATA_DIR, d))]
    if not users:
        return False

    total_images = sum(
        len([f for f in os.listdir(os.path.join(FACE_DATA_DIR, u)) if f.endswith('.jpg')])
        for u in users
    )

    embeddings_db   = {}
    overall_success = 0
    overall_failed  = 0
    processed       = 0

    for user_code in users:
        user_dir = os.path.join(FACE_DATA_DIR, user_code)
        images   = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
        user_embeddings = []

        for img_file in images:
            img_path = os.path.join(user_dir, img_file)
            try:
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="skip",
                    enforce_detection=False,
                    align=True          # must match recognition pipeline
                )
                if result:
                    user_embeddings.append(np.array(result[0]['embedding']))
                    overall_success += 1
            except Exception:
                overall_failed += 1

            processed += 1
            if progress_cb:
                pct = processed / max(1, total_images)
                acc = (overall_success / max(1, overall_success + overall_failed)) * 100
                progress_cb(pct, acc, overall_success, overall_failed)

        if user_embeddings:
            embeddings_db[user_code] = user_embeddings

    np.save(MODEL_PATH, embeddings_db)
    return True
