# 🎓 Smart Attendance System
> High-accuracy face recognition attendance using **ArcFace + MTCNN** with real-time progress tracking

---

## 🧠 Tech Stack

| Component | Library | Why |
|---|---|---|
| Face Detection | `MTCNN` | Multi-scale, handles angles, best detection accuracy |
| Face Recognition | `DeepFace` + `ArcFace` | 99.4%+ benchmark accuracy, 512-dim embeddings |
| Similarity | Cosine Distance (`scipy`) | Better than Euclidean for high-dim embeddings |
| UI / Progress | `rich` | Beautiful terminal progress bars & tables |
| Database | `SQLite3` | Zero-config, local, no server needed |
| Webcam | `OpenCV` | Real-time frame capture & display |

---

## 📁 Project Structure

```
smart_attendance_system/
│
├── main.py               # Entry point & menu
├── register.py           # Face capture + ArcFace training with progress bars
├── recognize.py          # Real-time recognition + attendance marking
├── view_attendance.py    # Viewer, filter, export
├── requirements.txt
│
├── face_data/
│   └── <USER_CODE>/      # 80 face images per person
│       ├── img_0000.jpg
│       └── ...
│
├── models/
│   └── embeddings.npy    # ArcFace embedding vectors (all users)
│
├── database/
│   └── attendance.db     # SQLite — users + attendance tables
│
└── exports/
    └── *.csv             # Exported attendance files
```

---

## ⚙️ Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

> ⚠️ First run downloads ArcFace model weights (~500MB). Requires internet.

---

## 🚀 How to Use

### 1. Register a User
- Run `main.py` → Option 1
- Enter name and ID
- Webcam opens — capture 80 face images (auto, ~10 seconds)
- Model automatically retrains with ArcFace embeddings
- Training progress shown with per-user stats

### 2. Take Attendance
- Run `main.py` → Option 2
- Webcam opens — recognized faces are highlighted in **green**
- Attendance is auto-marked with duplicate prevention (one per day)
- Session summary printed when you quit

### 3. View Attendance
- Run `main.py` → Option 3
- Filter by date range or user ID
- Export to CSV

---

## 🎯 Accuracy Tips

- Capture samples in **good lighting**
- Move head slightly **left/right/up/down** during registration
- Keep the recognition threshold at `0.35` (cosine distance)
- More samples = better accuracy (80 is recommended, 100+ is ideal)

---

## 🗃️ Database Schema

```sql
-- Users
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    user_code     TEXT UNIQUE NOT NULL,
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Attendance
CREATE TABLE attendance (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER NOT NULL,
    date     DATE NOT NULL,
    time     TIME NOT NULL,
    status   TEXT DEFAULT 'PRESENT',
    UNIQUE(user_id, date),          -- prevents duplicate entries
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

## 📊 Training Output Example

```
╭─ Training Complete ───────────────────────────────╮
│ Users Trained          │ 5                        │
│ Images Processed       │ 400                      │
│ Valid Embeddings       │ 387                      │
│ Rejected (bad frames)  │ 13                       │
│ Embedding Quality      │ 96.8%                    │
│ Model                  │ ArcFace (512-dim vectors)│
│ Detector               │ MTCNN                    │
╰───────────────────────────────────────────────────╯
```

---

## 🔒 Privacy

- All data stored **locally** — no cloud
- Delete a user via Option 5 → removes face data + DB records + retrains model
