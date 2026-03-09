"""
recognize.py — Face Recognition Engine (GUI-only)
===================================================
Provides ArcFace embedding matching, face validation helpers,
and glasses-masking utilities.  All UI runs through gui.py.
"""

import os
import multiprocessing
import math
import tempfile

# ── Pin ALL thread pools to physical core count BEFORE TF/DeepFace loads ──
CPU_COUNT = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"]     = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["TF_NUM_INTRAOP_THREADS"]   = str(CPU_COUNT)
os.environ["TF_NUM_INTEROP_THREADS"]   = str(CPU_COUNT)

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Tell OpenCV to use all cores too
cv2.setNumThreads(CPU_COUNT)

for _d in ["face_data", "models", "database", "exports"]:
    os.makedirs(_d, exist_ok=True)

MODEL_PATH = "models/embeddings.npy"

# Tuned thresholds for ArcFace (cosine distance — LOWER = more similar)
RECOGNITION_THRESHOLD = 0.42   # below this = recognized
MARGIN_THRESHOLD      = 0.06   # best must beat 2nd-best by this much (prevents swap)
CONFIDENCE_THRESHOLD  = 0.92   # MTCNN detection confidence

# ─────────────────────────────────────────────
# LOAD MODEL & USER DATA
# ─────────────────────────────────────────────

def load_embeddings():
    """Load saved embeddings from disk (used by gui.py at session start)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return np.load(MODEL_PATH, allow_pickle=True).item()

# ─────────────────────────────────────────────
# RECOGNITION LOGIC
# ─────────────────────────────────────────────

def _build_mean_embeddings(embeddings_db):
    """
    Pre-compute a single representative embedding per user.

    1. L2-normalise every raw embedding.
    2. Compute the median vector (robust to outliers).
    3. Prune samples that are > 2.5× the median cosine-distance away.
    4. Average the remaining clean samples → final representative.

    This makes close users (e.g. adi vs mam) much more separable.
    """
    mean_db = {}
    for user_code, embs in embeddings_db.items():
        arr = np.array(embs)                             # (N, 512)
        # L2-normalise each sample
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr_n = arr / norms

        # Robust centre: median
        med = np.median(arr_n, axis=0)
        med = med / (np.linalg.norm(med) or 1)

        # Cosine distances to median
        dists = np.array([cosine(v, med) for v in arr_n])
        med_dist = np.median(dists)
        cutoff = max(med_dist * 2.5, 0.15)

        # Keep only inliers
        clean = arr_n[dists <= cutoff]
        if len(clean) < 3:
            clean = arr_n  # fallback: keep all if too few survive

        mean_vec = clean.mean(axis=0)
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
        pad = int(0.20*max(w,h))
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

    Two-stage strategy:
      Stage 1 — Mean-distance shortlist (fast):
        Compare query against the pruned-mean embedding of each user.
      Stage 2 — Sample-level verification (disambiguates close users):
        If the top-2 means are close (margin < 0.10), compare the query
        against ALL individual samples of the top-2 users.  The user
        whose samples are closer ON AVERAGE wins.  This prevents swaps
        between users who look similar at the mean level.

    Returns (user_code, distance, confidence_pct) or (None, 1.0, 0)
    """
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "face_tmp.jpg")
        cv2.imwrite(tmp_path, face_img)

        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        if not result:
            return None, 1.0, 0

        query_embedding = np.array(result[0]['embedding'])
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding = query_embedding / q_norm

        if mean_db is None:
            mean_db = _build_mean_embeddings(embeddings_db)

        # ── Stage 1: mean-distance ranking ────────────────────────────
        distances = {}
        for user_code, mean_emb in mean_db.items():
            distances[user_code] = cosine(query_embedding, mean_emb)

        sorted_users = sorted(distances.items(), key=lambda x: x[1])
        best_code, best_dist = sorted_users[0]

        # Gate 1: absolute threshold
        if best_dist >= RECOGNITION_THRESHOLD:
            return None, best_dist, 0

        # ── Stage 2: sample-level verification for close pairs ────────
        if len(sorted_users) > 1:
            second_code, second_dist = sorted_users[1]
            mean_margin = second_dist - best_dist

            if mean_margin < MARGIN_THRESHOLD:
                # Too ambiguous at the mean level → reject
                return None, best_dist, 0

            # If margin is moderate (< 0.10), double-check with samples
            if mean_margin < 0.10:
                def _avg_sample_dist(code):
                    embs = embeddings_db.get(code, [])
                    if not embs:
                        return 1.0
                    arr = np.array(embs)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    arr_n = arr / norms
                    # Use top-10 closest samples (most representative)
                    sample_dists = np.array([cosine(query_embedding, s) for s in arr_n])
                    top_k = min(10, len(sample_dists))
                    return float(np.sort(sample_dists)[:top_k].mean())

                d1 = _avg_sample_dist(best_code)
                d2 = _avg_sample_dist(second_code)

                if d2 - d1 < 0.03:
                    # Still too close at sample level → unknown
                    return None, best_dist, 0
                if d1 > d2:
                    # Samples say second user is actually closer → unknown
                    return None, best_dist, 0

        confidence = max(0, (1 - best_dist / RECOGNITION_THRESHOLD)) * 100
        confidence = min(confidence, 99.9)
        return best_code, best_dist, confidence

    except Exception:
        return None, 1.0, 0
