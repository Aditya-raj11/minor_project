"""
gui.py — Smart Attendance System  (v4 — Clean UI)
===================================================
Design direction: Warm, human, approachable — like a modern school app.
Light background, soft shadows, rounded cards, friendly sans-serif type.
No technical jargon exposed to users.

Performance: 3-thread engine from v3 fully retained.
New: Clear Attendance button (with date picker) for testing/reset.
"""

import multiprocessing as _mp, os as _os
_CPU = _mp.cpu_count()
_os.environ["OMP_NUM_THREADS"]        = str(_CPU)
_os.environ["OPENBLAS_NUM_THREADS"]   = str(_CPU)
_os.environ["MKL_NUM_THREADS"]        = str(_CPU)
_os.environ["TF_NUM_INTRAOP_THREADS"] = str(_CPU)
_os.environ["TF_NUM_INTEROP_THREADS"] = str(_CPU)

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import cv2, numpy as np, sqlite3
import threading, queue, os, shutil, time
import serial, serial.tools.list_ports
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from collections import deque
import multiprocessing

cv2.setNumThreads(multiprocessing.cpu_count())


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM MODULES  (recognition logic lives in recognize.py / register.py)
# ─────────────────────────────────────────────────────────────────────────────
from recognize import mask_glasses_region, identify_face, build_mean_embeddings
from register  import train_model_with_callback


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (technical — not shown in UI)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH         = "database/attendance.db"
FACE_DATA_DIR   = "face_data"
MODEL_PATH      = "models/embeddings.npy"
SAMPLES_NEEDED  = 120
CONF_THRESHOLD  = 0.97
RECOG_THRESHOLD = 0.42
BLUR_THRESHOLD  = 80
COOLDOWN_SEC    = 3.0
CAM_W, CAM_H    = 640, 480
DISPLAY_FPS     = 120
DETECT_EVERY    = 1
INFER_COOLDOWN  = 0.1

CAPTURE_ROUNDS = [
    (25, "Round 1 of 5", "Look straight at the camera"),
    (25, "Round 2 of 5", "Slowly turn your head left and right"),
    (25, "Round 3 of 5", "Pull your hair back or change style"),
    (25, "Round 4 of 5", "Tilt your head slightly up and down"),
    (20, "Round 5 of 5", "Any variation — glasses on/off, different expression"),
]

# ─────────────────────────────────────────────────────────────────────────────
# WARM CLEAN PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    # Backgrounds
    "bg":        "#F5F4F0",   # warm off-white
    "bg2":       "#EDECEA",   # slightly darker warm
    "sidebar":   "#FFFFFF",   # pure white sidebar
    "card":      "#FFFFFF",   # white cards
    "card_warm": "#FFFBF5",   # very warm white

    # Borders & dividers
    "border":    "#E8E6E0",
    "border2":   "#D8D5CE",

    # Brand / accent — warm indigo
    "accent":    "#5B6AF0",
    "accent_lt": "#EEF0FE",
    "accent_dk": "#4553D4",

    # Status colors
    "green":     "#22C55E",
    "green_lt":  "#F0FDF4",
    "red":       "#EF4444",
    "red_lt":    "#FFF1F1",
    "amber":     "#F59E0B",
    "amber_lt":  "#FFFBEB",

    # Typography
    "text":      "#1C1917",   # near-black warm
    "text2":     "#6B7280",   # medium grey
    "text3":     "#9CA3AF",   # light grey
    "text_inv":  "#FFFFFF",

    # Nav selected
    "nav_sel":   "#F0F2FF",
    "nav_dot":   "#5B6AF0",
}

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

FONT_H1   = ("Georgia", 22, "bold")
FONT_H2   = ("Georgia", 17, "bold")
FONT_H3   = ("Georgia", 13, "bold")
FONT_BODY = ("Helvetica Neue", 12)
FONT_SM   = ("Helvetica Neue", 10)
FONT_XS   = ("Helvetica Neue", 9)
FONT_NUM  = ("Georgia", 32, "bold")
FONT_NUM2 = ("Georgia", 20, "bold")

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION + DATABASE  (logic lives in their own modules)
# ─────────────────────────────────────────────────────────────────────────────
from recognize import (
    mask_glasses_region, identify_face, build_mean_embeddings,
    detect_valid_faces, is_valid_face_shape, has_valid_landmarks, is_sharp_enough,
)
from register  import train_model_with_callback
from database  import (
    init_db, get_users, get_today_attendance, get_attendance_filtered,
    mark_attendance_db, save_user_db, delete_user_db, delete_all_users_db,
    user_exists, clear_attendance_db, get_all_users_dict,
)

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY RING  (restyled warm)
# ─────────────────────────────────────────────────────────────────────────────
class AccuracyRing(tk.Canvas):
    def __init__(self,parent,size=130,bg=None,**kwargs):
        bg=bg or C["card"]
        super().__init__(parent,width=size,height=size,bg=bg,highlightthickness=0,**kwargs)
        self.size=size; self.value=0.0; self.target=0.0; self._draw(0)

    def set_value(self,v):
        self.target=max(0.0,min(100.0,float(v))); self._animate()

    def _animate(self):
        d=self.target-self.value
        if abs(d)>0.4:
            self.value+=d*0.14; self._draw(self.value); self.after(16,self._animate)
        else:
            self.value=self.target; self._draw(self.value)

    def _draw(self,val):
        self.delete("all")
        s=self.size; pad=16; cx=cy=s//2
        # Shadow ring
        self.create_arc(pad+1,pad+1,s-pad+1,s-pad+1,start=90,extent=360,
                        style=tk.ARC,outline="#E5E3DD",width=10)
        # Track
        self.create_arc(pad,pad,s-pad,s-pad,start=90,extent=360,
                        style=tk.ARC,outline=C["border"],width=10)
        # Value arc
        if val>1:
            col=(C["green"] if val>=75 else C["amber"] if val>=50 else C["red"])
            self.create_arc(pad,pad,s-pad,s-pad,start=90,extent=-int(3.6*val),
                            style=tk.ARC,outline=col,width=10)
        col=(C["green"] if val>=75 else C["amber"] if val>=50 else
             (C["red"] if val>1 else C["text3"]))
        self.create_text(cx,cy-8,text=f"{int(val)}%",
                         fill=col,font=("Georgia",18,"bold"))
        self.create_text(cx,cy+12,text="match",
                         fill=C["text3"],font=("Helvetica Neue",9))

# ─────────────────────────────────────────────────────────────────────────────
# WEBCAM ENGINE  (unchanged 3-thread architecture from v3)
# ─────────────────────────────────────────────────────────────────────────────
class WebcamEngine:
    def __init__(self,on_frame,on_recognized,on_fps,embeddings_db,all_users):
        self.on_frame=on_frame; self.on_recognized=on_recognized; self.on_fps=on_fps
        self.embeddings_db=embeddings_db; self.all_users=all_users
        self.frame_q=queue.Queue(maxsize=2); self.result_q=queue.Queue(maxsize=4)
        self.running=False; self.cap=None
        self._overlay_lock=threading.Lock()
        self._overlay_boxes=[]; self._overlay_rej=[]
        self._fps_deque=deque(maxlen=30)
        self._mean_db=None   # built once from embeddings_db, reset on model reload
        self._conf_smooth={}  # user_code → EMA-smoothed confidence (reduces fluctuation)

    def start(self):
        self.running=True
        threading.Thread(target=self._capture_loop,daemon=True).start()
        threading.Thread(target=self._infer_loop,daemon=True).start()
        threading.Thread(target=self._display_loop,daemon=True).start()
        return True

    def stop(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None

    def _capture_loop(self):
        for idx in [0, 1, 2]:
            self.cap=cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if self.cap.isOpened(): break
        if not self.cap or not self.cap.isOpened():
            self.running=False
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS,30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        
        while self.running:
            ret,frame=self.cap.read()
            if not ret: time.sleep(0.01); continue
            if self.frame_q.full():
                try: self.frame_q.get_nowait()
                except queue.Empty: pass
            try: self.frame_q.put_nowait(frame)
            except queue.Full: pass

    def _infer_loop(self):
        try:
            from mtcnn import MTCNN
            from deepface import DeepFace
            from scipy.spatial.distance import cosine
        except ImportError: return
        detector=MTCNN(); last_infer={}; frame_count=0
        while self.running:
            try: frame=self.frame_q.get(timeout=0.5)
            except queue.Empty: continue
            frame_count+=1
            if frame_count%DETECT_EVERY!=0: continue
            valid_faces,rejected=detect_valid_faces(frame,detector)
            boxes=[]; now=time.time()
            def compute_iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
                yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = boxA[2] * boxA[3]
                boxBArea = boxB[2] * boxB[3]
                iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
                return iou

            if not hasattr(self, "_face_cache"): self._face_cache = []
            self._face_cache = [c for c in self._face_cache if now - c['time'] < INFER_COOLDOWN * 2]
            
            new_cache = []
            available_cache = list(self._face_cache)
            
            for result,face_224,x,y,w,h,conf in valid_faces:
                user_code=None; conf_pct=0.0; best_dist=1.0
                current_box = (x, y, w, h)
                
                best_iou = 0
                matched_idx = -1
                for idx, c in enumerate(available_cache):
                    iou = compute_iou(current_box, c['box'])
                    if iou > best_iou:
                        best_iou = iou
                        matched_idx = idx
                        
                needs_inference = True
                
                if best_iou > 0.4 and matched_idx != -1:
                    cached_data = available_cache.pop(matched_idx)
                    if now - cached_data['last_infer'] < INFER_COOLDOWN:
                        user_code = cached_data['code']
                        conf_pct = cached_data['conf']
                        needs_inference = False
                        new_cache.append({
                            'box': current_box, 'code': user_code, 'conf': conf_pct,
                            'time': now, 'last_infer': cached_data['last_infer']
                        })
                
                if needs_inference:
                    try:
                        # ── Glasses mask (from recognize.py) ───────────────────────
                        face_for_emb = face_224.copy()
                        kp = result.get('keypoints', {})
                        if 'left_eye' in kp and 'right_eye' in kp:
                            rx2,ry2,rw2,rh2 = result['box']
                            rx2,ry2 = max(0,rx2), max(0,ry2)
                            pad2 = int(0.12*max(rw2,rh2))
                            x1c  = max(0,rx2-pad2); y1c = max(0,ry2-pad2)
                            x2c  = min(frame.shape[1],rx2+rw2+pad2)
                            y2c  = min(frame.shape[0],ry2+rh2+pad2)
                            cw2  = max(1,x2c-x1c); ch2 = max(1,y2c-y1c)
                            le2  = ((kp['left_eye'][0] -x1c)*224.0/cw2,
                                    (kp['left_eye'][1] -y1c)*224.0/ch2)
                            re2  = ((kp['right_eye'][0]-x1c)*224.0/cw2,
                                    (kp['right_eye'][1]-y1c)*224.0/ch2)
                            face_for_emb = mask_glasses_region(face_for_emb, le2, re2)

                        # ── Identify (from recognize.py) ─────────────────────
                        if self._mean_db is None:
                            self._mean_db = build_mean_embeddings(self.embeddings_db)
                        user_code, _dist, conf_pct = identify_face(
                            face_for_emb, self.embeddings_db, self._mean_db
                        )
                    except Exception as e:
                        print(f"[DEBUG ERR] Recognition error: {e}")
                    
                    if user_code:
                        # ── EMA smoothing: 70% old + 30% new ──
                        EMA = 0.30
                        prev = self._conf_smooth.get(user_code, conf_pct)
                        conf_pct = (1 - EMA) * prev + EMA * conf_pct
                        self._conf_smooth[user_code] = conf_pct
                        new_cache.append({
                            'box': current_box, 'code': user_code, 'conf': conf_pct,
                            'time': now, 'last_infer': now
                        })
                        try: self.result_q.put_nowait((user_code, conf_pct))
                        except queue.Full: pass

                name=self.all_users.get(user_code,None) if user_code else None
                boxes.append((x,y,w,h,name,conf_pct,user_code))
                
            self._face_cache = new_cache
            with self._overlay_lock:
                self._overlay_boxes=boxes; self._overlay_rej=rejected

    def _display_loop(self):
        interval=1.0/DISPLAY_FPS; last_t=time.time()
        while self.running:
            now=time.time()
            delta=now-last_t
            if delta<interval: time.sleep(interval-delta)
            fps=1.0/max(0.001,time.time()-last_t); last_t=time.time()
            self._fps_deque.append(fps)
            avg_fps=sum(self._fps_deque)/len(self._fps_deque)
            if not self.cap or not self.cap.isOpened(): continue
            ret,frame=self.cap.read()
            if not ret: continue
            display=frame.copy()
            with self._overlay_lock:
                boxes=list(self._overlay_boxes)
                rej=list(self._overlay_rej)

            # Draw face boxes — clean rounded style
            for (x,y,w,h,name,cp,uc) in boxes:
                if name:
                    col=(0,180,60); label=name
                    sub=f"{cp:.0f}% match"
                else:
                    col=(180,60,60); label="Unknown"; sub=""
                # Rounded corners only
                cl=18
                for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                    cv2.line(display,(px,py),(px+dx*cl,py),col,3)
                    cv2.line(display,(px,py),(px,py+dy*cl),col,3)
                # Soft label pill
                ly=max(y-12,22)
                (lw2,lh2),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
                cv2.rectangle(display,(x,ly-lh2-6),(x+lw2+12,ly+4),col,-1)
                cv2.putText(display,label,(x+6,ly),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
                if sub:
                    cv2.putText(display,sub,(x,y+h+18),cv2.FONT_HERSHEY_SIMPLEX,0.50,col,1)

            # Minimal HUD — just time
            h2,w2=display.shape[:2]
            cv2.putText(display,datetime.now().strftime("%H:%M:%S"),
                        (w2-100,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)

            self.on_fps(f"{avg_fps:.0f} fps")
            self.on_frame(display)

    def get_result(self):
        try: return self.result_q.get_nowait()
        except queue.Empty: return None

    def update_overlay_mark(self,marked_codes):
        with self._overlay_lock:
            updated=[]
            for (x,y,w,h,name,cp,uc) in self._overlay_boxes:
                updated.append((x,y,w,h,name,cp,uc))
            self._overlay_boxes=updated

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def card(parent, padx=0, pady=0, **kwargs):
    """White rounded card with soft shadow effect."""
    inner=tk.Frame(parent,bg=C["card"],
                   highlightbackground=C["border"],
                   highlightthickness=1, **kwargs)
    return inner,inner

def label(parent,text,style="body",fg=None,**kwargs):
    fonts={"h1":FONT_H1,"h2":FONT_H2,"h3":FONT_H3,
           "body":FONT_BODY,"sm":FONT_SM,"xs":FONT_XS,
           "num":FONT_NUM,"num2":FONT_NUM2}
    colors={"h1":C["text"],"h2":C["text"],"h3":C["text"],
            "body":C["text2"],"sm":C["text2"],"xs":C["text3"],
            "num":C["accent"],"num2":C["accent"]}
    return tk.Label(parent,text=text,font=fonts.get(style,FONT_BODY),
                    fg=fg or colors.get(style,C["text2"]),
                    bg=parent["bg"],**kwargs)

def pill_btn(parent,text,color,hover,command,width=130,height=38,text_color="white"):
    return ctk.CTkButton(parent,text=text,width=width,height=height,
                          font=("Helvetica Neue",12),
                          fg_color=color,hover_color=hover,
                          text_color=text_color,corner_radius=20,
                          command=command)

def ghost_btn(parent,text,command,width=130,height=38):
    return ctk.CTkButton(parent,text=text,width=width,height=height,
                          font=("Helvetica Neue",12),
                          fg_color="transparent",
                          hover_color=C["bg2"],
                          text_color=C["text2"],
                          border_color=C["border2"],
                          border_width=1,
                          corner_radius=20,
                          command=command)

def divider(parent):
    tk.Frame(parent,bg=C["border"],height=1).pack(fill="x",padx=20,pady=12)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAR ATTENDANCE DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class ClearAttendanceDialog(tk.Toplevel):
    def __init__(self,parent,on_cleared):
        super().__init__(parent)
        self.on_cleared=on_cleared
        self.title("Clear Attendance")
        self.geometry("420x400")
        self.resizable(False,False)
        self.configure(bg=C["bg"])
        self.grab_set()  # modal

        # Center on parent
        self.transient(parent)
        px=parent.winfo_x()+parent.winfo_width()//2-210
        py=parent.winfo_y()+parent.winfo_height()//2-200
        self.geometry(f"+{px}+{py}")

        self._build()

    def _build(self):
        main=tk.Frame(self,bg=C["card"],
                      highlightbackground=C["border"],highlightthickness=1)
        main.pack(fill="both",expand=True,padx=16,pady=16)

        # Icon + title
        top=tk.Frame(main,bg=C["card"]); top.pack(fill="x",padx=24,pady=(24,0))
        tk.Label(top,text="🗑",font=("Helvetica Neue",28),bg=C["card"]).pack(side="left",padx=(0,12))
        title_f=tk.Frame(top,bg=C["card"]); title_f.pack(side="left")
        tk.Label(title_f,text="Clear Attendance",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(anchor="w")
        tk.Label(title_f,text="Remove records for testing purposes",
                 font=FONT_SM,fg=C["text3"],bg=C["card"]).pack(anchor="w")

        tk.Frame(main,bg=C["border"],height=1).pack(fill="x",padx=24,pady=16)

        # Options
        self.mode=tk.StringVar(value="all")
        opts_f=tk.Frame(main,bg=C["card"]); opts_f.pack(fill="x",padx=24)

        for val,lbl_text,sub in [
            ("today","Clear Today's Records","Removes attendance for today only"),
            ("date","Clear Specific Date","Enter a date below"),
            ("all","Clear All Records","⚠ Removes everything — use carefully"),
        ]:
            row=tk.Frame(opts_f,bg=C["card"]); row.pack(fill="x",pady=4)
            rb=tk.Radiobutton(row,variable=self.mode,value=val,
                               bg=C["card"],activebackground=C["card"],
                               fg=C["text"],selectcolor=C["accent"],
                               font=FONT_BODY)
            rb.pack(side="left")
            tk.Label(row,text=lbl_text,font=FONT_BODY,fg=C["text"],bg=C["card"]).pack(side="left")
            tk.Label(row,text=f"  — {sub}",font=FONT_SM,fg=C["text3"],bg=C["card"]).pack(side="left")

        # Date entry (shown only when mode=date)
        date_f=tk.Frame(main,bg=C["card"]); date_f.pack(fill="x",padx=24,pady=(8,0))
        tk.Label(date_f,text="Date (YYYY-MM-DD):",font=FONT_SM,fg=C["text2"],bg=C["card"]).pack(side="left")
        self.date_entry=ctk.CTkEntry(date_f,width=140,height=32,
                                      font=("Helvetica Neue",11),
                                      fg_color=C["bg"],border_color=C["border2"],
                                      text_color=C["text"],
                                      placeholder_text=datetime.now().strftime("%Y-%m-%d"))
        self.date_entry.pack(side="left",padx=8)

        # Buttons
        btn_f=tk.Frame(main,bg=C["card"]); btn_f.pack(fill="x",padx=24,pady=(20,24))
        ghost_btn(btn_f,"Cancel",self.destroy,width=100).pack(side="left")
        pill_btn(btn_f,"Clear Records",C["red"],"#CC2222",self._do_clear,width=140).pack(side="right")

    def _do_clear(self):
        mode=self.mode.get()
        if mode=="today":
            date=datetime.now().strftime("%Y-%m-%d")
            label_str="today"
        elif mode=="date":
            date=self.date_entry.get().strip()
            if not date:
                messagebox.showwarning("Missing Date","Please enter a date.", parent=self); return
            label_str=f"date {date}"
        else:
            date=None
            label_str="ALL dates"

        confirm=messagebox.askyesno("Confirm",
            f"Clear attendance for {label_str}?\nThis cannot be undone.", parent=self)
        if not confirm: return

        count=clear_attendance_db(date)
        self.on_cleared(count,label_str)
        self.destroy()

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE SERIAL LINK
# ─────────────────────────────────────────────────────────────────────────────

def get_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class SmartAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        init_db()
        self.title("Attendance")
        self.geometry("1360x840")
        self.minsize(1100,720)
        self.configure(fg_color=C["bg"])

        self.engine=None
        self.capture_active=False
        self.embeddings_db=None
        self.all_users={}
        self.marked_today=set()
        self.current_tab="home"
        self._last_display=0.0
        self._display_interval=1.0/30
        self._imgtk_att=None

        self.arduino_serial = None
        self.arduino_port = tk.StringVar(value="None")
        
        # Build UI layout immediately

        self._build_ui()
        self._refresh_users_list()
        self._refresh_attendance_table()
        self._update_stats()
        self._clock_tick()
        self._poll_results()

    # ══════════════════════════════════════════════════════════════════════════
    # UI SKELETON
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar=tk.Frame(self,bg=C["sidebar"],width=230)
        sidebar.pack(side="left",fill="y"); sidebar.pack_propagate(False)
        tk.Frame(self,bg=C["border"],width=1).pack(side="left",fill="y")

        # Logo area
        logo_f=tk.Frame(sidebar,bg=C["sidebar"])
        logo_f.pack(fill="x",padx=24,pady=(28,8))
        tk.Label(logo_f,text="Attendance",font=("Georgia",20,"bold"),
                 fg=C["text"],bg=C["sidebar"]).pack(anchor="w")
        self.clock_lbl=tk.Label(logo_f,text="",font=FONT_SM,
                                 fg=C["text3"],bg=C["sidebar"])
        self.clock_lbl.pack(anchor="w",pady=(2,0))

        tk.Frame(sidebar,bg=C["border"],height=1).pack(fill="x",padx=24,pady=16)

        # Nav items
        self.nav_btns={}
        nav_items=[
            ("home",    "🏠", "Home"),
            ("register","👤", "Add Person"),
            ("attend",  "📷", "Take Attendance"),
            ("records", "📋", "Records"),
            ("people",  "👥", "People"),
        ]
        for key,icon,lbl_txt in nav_items:
            self.nav_btns[key]=self._nav_btn(sidebar,icon,lbl_txt,key)

        tk.Frame(sidebar,bg=C["border"],height=1).pack(fill="x",padx=24,pady=16)

        # Stat cards in sidebar
        self.stat_reg   =self._sidebar_stat(sidebar,"0","Registered")
        self.stat_today =self._sidebar_stat(sidebar,"0","Present Today")
        self.stat_total =self._sidebar_stat(sidebar,"0","Total Records")

        # ── Content ──────────────────────────────────────────────────────────
        self.content=tk.Frame(self,bg=C["bg"])
        self.content.pack(side="left",fill="both",expand=True)

        self.pages={
            "home":     self._build_home_page(),
            "register": self._build_register_page(),
            "attend":   self._build_attend_page(),
            "records":  self._build_records_page(),
            "people":   self._build_people_page(),
        }
        self._switch_tab("home")

    def _nav_btn(self,parent,icon,label_txt,key):
        f=tk.Frame(parent,bg=C["sidebar"],cursor="hand2")
        f.pack(fill="x",padx=12,pady=2)

        def enter(e):
            if self.current_tab!=key: f.configure(bg=C["bg2"])
        def leave(e):
            if self.current_tab!=key: f.configure(bg=C["sidebar"])
        def click(e): self._switch_tab(key)

        inner=tk.Frame(f,bg=f["bg"]); inner.pack(fill="x",padx=12,pady=10)
        icon_lbl=tk.Label(inner,text=icon,font=("Helvetica Neue",14),
                           fg=C["text"],bg=inner["bg"])
        icon_lbl.pack(side="left")
        txt_lbl=tk.Label(inner,text=f"  {label_txt}",font=FONT_BODY,
                          fg=C["text2"],bg=inner["bg"])
        txt_lbl.pack(side="left")

        for w in [f,inner,icon_lbl,txt_lbl]:
            w.bind("<Enter>",enter); w.bind("<Leave>",leave); w.bind("<Button-1>",click)

        # Indicator dot
        self._dot=None
        return f

    def _sidebar_stat(self,parent,val,lbl_txt):
        f=tk.Frame(parent,bg=C["sidebar"]); f.pack(fill="x",padx=24,pady=6)
        row=tk.Frame(f,bg=C["sidebar"]); row.pack(fill="x")
        num=tk.Label(row,text=val,font=("Georgia",26,"bold"),fg=C["accent"],bg=C["sidebar"])
        num.pack(side="left")
        tk.Label(f,text=lbl_txt,font=FONT_XS,fg=C["text3"],bg=C["sidebar"]).pack(anchor="w")
        return num

    def _switch_tab(self,key):
        if self.engine and key!="attend":
            self._stop_webcam()
        self.current_tab=key
        for k,page in self.pages.items(): page.pack_forget()
        self.pages[key].pack(fill="both",expand=True)

        # Update nav highlight
        for k,btn in self.nav_btns.items():
            is_sel=(k==key)
            bg=C["nav_sel"] if is_sel else C["sidebar"]
            btn.configure(bg=bg,highlightbackground=C["nav_sel"] if is_sel else C["sidebar"],
                          highlightthickness=1 if is_sel else 0)
            for w in btn.winfo_children():
                w.configure(bg=bg)
                for ww in w.winfo_children(): ww.configure(bg=bg)

        if key=="records": self._refresh_attendance_table()
        elif key=="people": self._refresh_users_list()
        elif key=="attend": self._load_model_async()
        elif key=="home": self._update_stats()

    # ══════════════════════════════════════════════════════════════════════════
    # HOME PAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_home_page(self):
        page=tk.Frame(self.content,bg=C["bg"])

        # Header
        hdr=tk.Frame(page,bg=C["bg"]); hdr.pack(fill="x",padx=36,pady=(32,0))
        tk.Label(hdr,text="Good morning 👋",font=FONT_H1,fg=C["text"],bg=C["bg"]).pack(anchor="w")
        tk.Label(hdr,text="Here's what's happening today.",
                 font=FONT_BODY,fg=C["text2"],bg=C["bg"]).pack(anchor="w",pady=(4,0))

        # Stats row
        stats=tk.Frame(page,bg=C["bg"]); stats.pack(fill="x",padx=36,pady=24)

        for title,key,icon,color in [
            ("Registered","reg","👤",C["accent"]),
            ("Present Today","today","✅",C["green"]),
            ("Total Records","total","📊",C["amber"]),
        ]:
            _,card_inner=card(stats); card_inner.pack(side="left",fill="both",expand=True,padx=(0,12),ipady=20,ipadx=16)
            tk.Label(card_inner,text=icon,font=("Helvetica Neue",22),bg=C["card"]).pack(anchor="w",padx=20,pady=(16,4))
            num_lbl=tk.Label(card_inner,text="0",font=("Georgia",34,"bold"),fg=color,bg=C["card"])
            num_lbl.pack(anchor="w",padx=20)
            tk.Label(card_inner,text=title,font=FONT_SM,fg=C["text3"],bg=C["card"]).pack(anchor="w",padx=20,pady=(2,16))
            setattr(self,f"home_stat_{key}",num_lbl)

        # Today's log
        _,log_card=card(page); log_card.pack(fill="both",expand=True,padx=36,pady=(0,32))

        log_hdr=tk.Frame(log_card,bg=C["card"]); log_hdr.pack(fill="x",padx=20,pady=(16,8))
        tk.Label(log_hdr,text="Today's Attendance",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(side="left")
        pill_btn(log_hdr,"Take Attendance",C["accent"],C["accent_dk"],
                  lambda:self._switch_tab("attend"),width=148).pack(side="right")

        tk.Frame(log_card,bg=C["border"],height=1).pack(fill="x",padx=20,pady=(0,8))

        self.home_log=tk.Listbox(log_card,bg=C["card"],fg=C["text2"],
                                  selectbackground=C["accent_lt"],
                                  font=("Helvetica Neue",11),
                                  bd=0,highlightthickness=0,
                                  activestyle="none",relief="flat",
                                  selectforeground=C["text"])
        vsb=tk.Scrollbar(log_card,orient="vertical",command=self.home_log.yview,
                          bg=C["bg2"],troughcolor=C["card"])
        self.home_log.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y",pady=4)
        self.home_log.pack(fill="both",expand=True,padx=12,pady=(0,12))

        return page

    def _refresh_home_log(self):
        rows=get_today_attendance()
        self.home_log.delete(0,tk.END)
        if not rows:
            self.home_log.insert(0,"   No attendance / users yet")
            return
        for name,code,in_t,out_t,status,conf in rows:
            if status == "ABSENT":
                self.home_log.insert(tk.END, f"   ✗  {name:<22}  {code:<14}  {'--':<8}   ABSENT")
            else:
                t_str = f"{in_t}-{out_t}" if out_t else str(in_t)
                self.home_log.insert(tk.END, f"   ✓  {name:<22}  {code:<14}  {t_str:<16}   {conf:.0f}% match")

    # ══════════════════════════════════════════════════════════════════════════
    # REGISTER PAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_register_page(self):
        page=tk.Frame(self.content,bg=C["bg"])

        hdr=tk.Frame(page,bg=C["bg"]); hdr.pack(fill="x",padx=36,pady=(32,0))
        tk.Label(hdr,text="Add New Person",font=FONT_H1,fg=C["text"],bg=C["bg"]).pack(anchor="w")
        tk.Label(hdr,text="We'll scan their face across 5 short rounds to make recognition more reliable.",
                 font=FONT_BODY,fg=C["text2"],bg=C["bg"]).pack(anchor="w",pady=(4,0))

        body=tk.Frame(page,bg=C["bg"]); body.pack(fill="both",expand=True,padx=36,pady=20)

        # ── Left form ────────────────────────────────────────────────────────
        _,left=card(body); left.pack(side="left",fill="y",padx=(0,16),ipadx=0)
        left.configure(width=340); left.pack_propagate(False)

        tk.Label(left,text="Person Details",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(anchor="w",padx=24,pady=(22,16))

        tk.Label(left,text="Full Name",font=FONT_SM,fg=C["text2"],bg=C["card"]).pack(anchor="w",padx=24)
        self.reg_name=ctk.CTkEntry(left,height=42,font=("Helvetica Neue",12),
                                    fg_color=C["bg"],border_color=C["border2"],
                                    text_color=C["text"],corner_radius=10,
                                    placeholder_text="e.g. Arjun Sharma")
        self.reg_name.pack(fill="x",padx=24,pady=(4,14))

        tk.Label(left,text="Roll / Employee ID",font=FONT_SM,fg=C["text2"],bg=C["card"]).pack(anchor="w",padx=24)
        self.reg_code=ctk.CTkEntry(left,height=42,font=("Helvetica Neue",12),
                                    fg_color=C["bg"],border_color=C["border2"],
                                    text_color=C["text"],corner_radius=10,
                                    placeholder_text="e.g. CS2024001")
        self.reg_code.pack(fill="x",padx=24,pady=(4,20))

        tk.Frame(left,bg=C["border"],height=1).pack(fill="x",padx=24)

        # Round indicator
        self.round_lbl=tk.Label(left,text="—",font=FONT_BODY,fg=C["text3"],
                                 bg=C["card"],wraplength=270,justify="left")
        self.round_lbl.pack(anchor="w",padx=24,pady=(14,4))

        # Capture progress
        tk.Label(left,text="Photo Progress",font=FONT_SM,fg=C["text2"],bg=C["card"]).pack(anchor="w",padx=24,pady=(10,4))
        prog_row=tk.Frame(left,bg=C["card"]); prog_row.pack(fill="x",padx=24)
        self.cap_pct_lbl=tk.Label(prog_row,text="0%",font=("Georgia",30,"bold"),
                                   fg=C["accent"],bg=C["card"])
        self.cap_pct_lbl.pack(side="left")
        self.cap_count_lbl=tk.Label(prog_row,text=f"  of {SAMPLES_NEEDED} photos",
                                     font=FONT_SM,fg=C["text3"],bg=C["card"])
        self.cap_count_lbl.pack(side="left",pady=6)
        self.cap_bar=ctk.CTkProgressBar(left,height=6,progress_color=C["accent"],
                                         fg_color=C["border"],corner_radius=3)
        self.cap_bar.set(0); self.cap_bar.pack(fill="x",padx=24,pady=(4,4))
        self.cap_status_lbl=tk.Label(left,text="Ready when you are",
                                      font=FONT_SM,fg=C["text3"],bg=C["card"])
        self.cap_status_lbl.pack(anchor="w",padx=24,pady=(0,14))

        tk.Frame(left,bg=C["border"],height=1).pack(fill="x",padx=24)

        # Training result
        tk.Label(left,text="Recognition Quality",font=FONT_SM,fg=C["text2"],bg=C["card"]).pack(anchor="w",padx=24,pady=(14,6))
        self.train_ring=AccuracyRing(left,size=120,bg=C["card"])
        self.train_ring.pack(padx=24)
        self.train_status_lbl=tk.Label(left,text="Will appear after scan",font=FONT_SM,
                                        fg=C["text3"],bg=C["card"],wraplength=270,justify="center")
        self.train_status_lbl.pack(padx=24,pady=(4,8))

        self.train_bar=ctk.CTkProgressBar(left,height=6,progress_color=C["green"],
                                           fg_color=C["border"],corner_radius=3)
        self.train_bar.set(0); self.train_bar.pack(fill="x",padx=24,pady=(0,4))
        self.train_detail_lbl=tk.Label(left,text="",font=FONT_XS,fg=C["text3"],bg=C["card"])
        self.train_detail_lbl.pack(anchor="w",padx=24,pady=(0,16))

        self.reg_btn=pill_btn(left,"Start Registration",C["accent"],C["accent_dk"],
                               self._start_registration,width=260,height=44)
        self.reg_btn.pack(padx=24,pady=(4,24))

        # ── Right camera ─────────────────────────────────────────────────────
        _,right=card(body); right.pack(side="left",fill="both",expand=True)

        cam_hdr=tk.Frame(right,bg=C["card"]); cam_hdr.pack(fill="x",padx=20,pady=(16,8))
        tk.Label(cam_hdr,text="Camera",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(side="left")
        self.reg_cam_status=tk.Label(cam_hdr,text="Off",font=FONT_SM,fg=C["text3"],bg=C["card"])
        self.reg_cam_status.pack(side="right")
        tk.Frame(right,bg=C["border"],height=1).pack(fill="x",padx=20,pady=(0,0))

        self.reg_cam_lbl=tk.Label(right,bg="#F0EDE8",
                                   text="📷\n\nCamera will start\nwhen you begin",
                                   font=FONT_BODY,fg=C["text3"],justify="center")
        self.reg_cam_lbl.pack(fill="both",expand=True,padx=1,pady=1)

        tips_f=tk.Frame(right,bg=C["bg2"]); tips_f.pack(fill="x")
        tk.Label(tips_f,text="💡  Tip: Good lighting makes a big difference. Try hair up AND down during the rounds.",
                 font=FONT_XS,fg=C["text2"],bg=C["bg2"],pady=8,padx=16).pack(side="left")

        return page

    # ══════════════════════════════════════════════════════════════════════════
    # ATTENDANCE PAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_attend_page(self):
        page=tk.Frame(self.content,bg=C["bg"])

        hdr=tk.Frame(page,bg=C["bg"]); hdr.pack(fill="x",padx=36,pady=(32,0))
        tk.Label(hdr,text="Take Attendance",font=FONT_H1,fg=C["text"],bg=C["bg"]).pack(anchor="w")
        self.attend_date_lbl=tk.Label(hdr,text=datetime.now().strftime("%A, %d %B %Y"),
                                       font=FONT_BODY,fg=C["text2"],bg=C["bg"])
        self.attend_date_lbl.pack(anchor="w",pady=(4,0))

        body=tk.Frame(page,bg=C["bg"]); body.pack(fill="both",expand=True,padx=36,pady=20)

        # ── Camera card ──────────────────────────────────────────────────────
        _,cam_card=card(body); cam_card.pack(side="left",fill="both",expand=True,padx=(0,16))

        cam_hdr=tk.Frame(cam_card,bg=C["card"]); cam_hdr.pack(fill="x",padx=20,pady=(14,8))
        tk.Label(cam_hdr,text="Live Camera",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(side="left")
        self.att_cam_status=tk.Label(cam_hdr,text="● Offline",font=FONT_SM,fg=C["text3"],bg=C["card"])
        self.att_cam_status.pack(side="right")

        tk.Frame(cam_card,bg=C["border"],height=1).pack(fill="x",padx=20)

        cam_body=tk.Frame(cam_card,bg=C["border"])
        cam_body.pack(fill="both",expand=True,padx=20)
        cam_body.pack_propagate(False)

        self.att_cam_lbl=tk.Label(cam_body,bg="#F0EDE8",
                                   text="📷\n\nClick Start to begin",
                                   font=FONT_BODY,fg=C["text3"],justify="center")
        self.att_cam_lbl.pack(fill="both",expand=True,padx=1,pady=1)

        ctrl=tk.Frame(cam_card,bg=C["card"]); ctrl.pack(fill="x",padx=20,pady=14)
        self.start_btn=pill_btn(ctrl,"▶  Start",C["green"],"#16A34A",self._start_webcam,width=110)
        self.start_btn.pack(side="left")
        self.stop_btn=pill_btn(ctrl,"■  Stop",C["red"],"#DC2626",self._stop_webcam,width=110)
        self.stop_btn.configure(state="disabled"); self.stop_btn.pack(side="left",padx=10)
        self.fps_lbl=tk.Label(ctrl,text="",font=FONT_XS,fg=C["text3"],bg=C["card"])
        self.fps_lbl.pack(side="right")

        # ── Right panel ──────────────────────────────────────────────────────
        right=tk.Frame(body,bg=C["bg"],width=300)
        right.pack(side="left",fill="y"); right.pack_propagate(False)

        # Accuracy card
        _,acc_card=card(right); acc_card.pack(fill="x",pady=(0,12))
        tk.Label(acc_card,text="Match Score",font=FONT_SM,fg=C["text2"],bg=C["card"],pady=12).pack()
        self.live_ring=AccuracyRing(acc_card,size=140,bg=C["card"])
        self.live_ring.pack(pady=(2,6))
        self.recog_name_lbl=tk.Label(acc_card,text="—",font=FONT_H3,fg=C["text"],bg=C["card"])
        self.recog_name_lbl.pack(pady=(0,2))
        self.recog_code_lbl=tk.Label(acc_card,text="Waiting for face...",
                                      font=FONT_SM,fg=C["text3"],bg=C["card"])
        self.recog_code_lbl.pack(pady=(0,14))

        # Today log card
        _,log_card=card(right); log_card.pack(fill="both",expand=True)
        log_hdr=tk.Frame(log_card,bg=C["card"]); log_hdr.pack(fill="x",padx=14,pady=(14,8))
        tk.Label(log_hdr,text="Marked Today",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(side="left")
        self.today_count_lbl=tk.Label(log_hdr,text="0",font=("Georgia",16,"bold"),
                                       fg=C["accent"],bg=C["card"])
        self.today_count_lbl.pack(side="right")
        tk.Frame(log_card,bg=C["border"],height=1).pack(fill="x",padx=14)
        self.log_listbox=tk.Listbox(log_card,bg=C["card"],fg=C["text2"],
                                     selectbackground=C["accent_lt"],
                                     font=("Helvetica Neue",10),
                                     bd=0,highlightthickness=0,
                                     activestyle="none",relief="flat")
        vsb=tk.Scrollbar(log_card,orient="vertical",command=self.log_listbox.yview,
                          bg=C["bg2"],troughcolor=C["card"])
        self.log_listbox.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y")
        self.log_listbox.pack(fill="both",expand=True,padx=6,pady=6)

        # ── Arduino Control Card ─────────────────────────────────────────────
        _,ard_card=card(right); ard_card.pack(fill="x", side="bottom", pady=(12, 0))
        tk.Label(ard_card,text="Hardware Interface",font=FONT_H3,fg=C["text"],bg=C["card"]).pack(pady=(12,4))
        tk.Frame(ard_card,bg=C["border"],height=1).pack(fill="x",padx=14)
        
        # COM port dropdown & refresh
        cport_frame = tk.Frame(ard_card, bg=C["card"])
        cport_frame.pack(fill="x", padx=14, pady=8)
        
        self.com_dropdown = ttk.Combobox(cport_frame, textvariable=self.arduino_port, state="readonly", width=12)
        self.com_dropdown['values'] = get_serial_ports()
        self.com_dropdown.pack(side="left")
        if self.com_dropdown['values']: self.com_dropdown.current(0)
        
        def refresh_ports():
            ports = get_serial_ports()
            self.com_dropdown['values'] = ports
            if ports and self.arduino_port.get() not in ports:
                self.com_dropdown.current(0)
                
        pill_btn(cport_frame,"⟳",C["accent"],C["accent_dk"],refresh_ports,width=28,height=22).pack(side="right")
        
        # Connect Button & Status
        conn_frame = tk.Frame(ard_card, bg=C["card"])
        conn_frame.pack(fill="x", padx=14, pady=(0, 12))
        
        self.ard_status_lbl = tk.Label(conn_frame,text="Disconnected",font=FONT_XS,fg=C["red"],bg=C["card"])
        self.ard_status_lbl.pack(side="right", pady=5)
        
        self.ard_conn_btn = pill_btn(conn_frame,"Connect",C["border2"],C["green"],self._toggle_arduino,width=90,height=26, text_color=C["text2"])
        self.ard_conn_btn.pack(side="left")

        return page

    def _toggle_arduino(self):
        if self.arduino_serial is None:
            # Try to connect
            port = self.arduino_port.get()
            if not port or port == "None":
                messagebox.showerror("Port Error", "Select a valid COM port.", parent=self)
                return
            try:
                self.arduino_serial = serial.Serial(port, 9600, timeout=1)
                self.ard_status_lbl.configure(text="Connected", fg=C["green"])
                self.ard_conn_btn.configure(bg=C["green"])
                for child in self.ard_conn_btn.winfo_children(): child.configure(bg=C["green"], fg="white", text="Disconnect")
                self.ard_conn_btn._orig_bg = C["green"]
            except Exception as e:
                messagebox.showerror("Serial Error", f"Failed to connect on {port}:\n\n{str(e)}", parent=self)
        else:
            # Disconnect
            try: self.arduino_serial.close()
            except: pass
            self.arduino_serial = None
            self.ard_status_lbl.configure(text="Disconnected", fg=C["red"])
            self.ard_conn_btn.configure(bg=C["border2"])
            for child in self.ard_conn_btn.winfo_children(): child.configure(bg=C["border2"], fg=C["text2"], text="Connect")
            self.ard_conn_btn._orig_bg = C["border2"]

    # ══════════════════════════════════════════════════════════════════════════
    # RECORDS PAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_records_page(self):
        page=tk.Frame(self.content,bg=C["bg"])

        hdr=tk.Frame(page,bg=C["bg"]); hdr.pack(fill="x",padx=36,pady=(32,0))
        tk.Label(hdr,text="Attendance Records",font=FONT_H1,fg=C["text"],bg=C["bg"]).pack(side="left")

        # Action buttons
        btn_row=tk.Frame(hdr,bg=C["bg"]); btn_row.pack(side="right")
        pill_btn(btn_row,"Export CSV",C["accent"],C["accent_dk"],
                  self._export_csv,width=120).pack(side="right",padx=(8,0))
        pill_btn(btn_row,"🗑  Clear",C["red"],"#DC2626",
                  self._open_clear_dialog,width=110,height=38).pack(side="right")

        # Filters
        _,filt=card(page); filt.pack(fill="x",padx=36,pady=(16,0))
        filt_inner=tk.Frame(filt,bg=C["card"]); filt_inner.pack(fill="x",padx=16,pady=12)

        def flbl(t):
            tk.Label(filt_inner,text=t,font=FONT_SM,fg=C["text2"],bg=C["card"],padx=6).pack(side="left")

        flbl("From")
        self.filter_from=ctk.CTkEntry(filt_inner,width=130,height=34,
                                       font=("Helvetica Neue",11),fg_color=C["bg"],
                                       border_color=C["border2"],text_color=C["text"],
                                       corner_radius=8,placeholder_text="YYYY-MM-DD")
        self.filter_from.pack(side="left",padx=(0,12))
        flbl("To")
        self.filter_to=ctk.CTkEntry(filt_inner,width=130,height=34,
                                     font=("Helvetica Neue",11),fg_color=C["bg"],
                                     border_color=C["border2"],text_color=C["text"],
                                     corner_radius=8,placeholder_text="YYYY-MM-DD")
        self.filter_to.pack(side="left",padx=(0,12))
        flbl("Person ID")
        self.filter_user=ctk.CTkEntry(filt_inner,width=130,height=34,
                                       font=("Helvetica Neue",11),fg_color=C["bg"],
                                       border_color=C["border2"],text_color=C["text"],
                                       corner_radius=8,placeholder_text="leave blank = all")
        self.filter_user.pack(side="left",padx=(0,12))
        pill_btn(filt_inner,"Search",C["accent"],C["accent_dk"],
                  self._refresh_attendance_table,width=90,height=34).pack(side="left")
        self.rec_count_lbl=tk.Label(filt_inner,text="",font=FONT_SM,fg=C["text3"],bg=C["card"],padx=10)
        self.rec_count_lbl.pack(side="right")

        # Table
        _,tf=card(page); tf.pack(fill="both",expand=True,padx=36,pady=14)
        self._style_tree()
        cols=("Name","Person ID","Date","In - Out","Status","Match %")
        self.tree=ttk.Treeview(tf,columns=cols,show="headings",style="Clean.Treeview")
        for col,w in zip(cols,[200,130,110,140,100,100]):
            self.tree.heading(col,text=col); self.tree.column(col,width=w,anchor="center")
        vsb=ttk.Scrollbar(tf,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); self.tree.pack(fill="both",expand=True)
        return page

    def _style_tree(self):
        sty=ttk.Style(); sty.theme_use("default")
        sty.configure("Clean.Treeview",
                       background=C["card"],foreground=C["text2"],
                       fieldbackground=C["card"],rowheight=36,
                       font=("Helvetica Neue",11),
                       borderwidth=0,relief="flat")
        sty.configure("Clean.Treeview.Heading",
                       background=C["bg2"],foreground=C["text"],
                       font=("Helvetica Neue",11,"bold"),relief="flat",
                       borderwidth=0)
        sty.map("Clean.Treeview",
                background=[("selected",C["accent_lt"])],
                foreground=[("selected",C["text"])])

    # ══════════════════════════════════════════════════════════════════════════
    # PEOPLE PAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_people_page(self):
        page=tk.Frame(self.content,bg=C["bg"])
        hdr=tk.Frame(page,bg=C["bg"]); hdr.pack(fill="x",padx=36,pady=(32,0))
        tk.Label(hdr,text="Registered People",font=FONT_H1,fg=C["text"],bg=C["bg"]).pack(side="left")
        pill_btn(hdr,"+ Add Person",C["accent"],C["accent_dk"],
                  lambda:self._switch_tab("register"),width=130).pack(side="right",padx=(8,0))
        ghost_btn(hdr,"Remove Selected",self._delete_selected_user,width=140).pack(side="right")
        
        # New "Remove All" button
        ghost_btn(hdr,"Remove All",self._delete_all_users,width=120).pack(side="right",padx=(0,8))

        _,tf=card(page); tf.pack(fill="both",expand=True,padx=36,pady=20)
        cols=("Name","Person ID","Date Added","Photos","Days Present")
        self.users_tree=ttk.Treeview(tf,columns=cols,show="headings",style="Clean.Treeview")
        for col,w in zip(cols,[220,140,160,120,120]):
            self.users_tree.heading(col,text=col); self.users_tree.column(col,width=w,anchor="center")
        vsb=ttk.Scrollbar(tf,orient="vertical",command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); self.users_tree.pack(fill="both",expand=True)
        return page

    # ══════════════════════════════════════════════════════════════════════════
    # REGISTRATION LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _start_registration(self):
        name=self.reg_name.get().strip(); code=self.reg_code.get().strip().upper()
        if not name or not code:
            messagebox.showerror("Missing Info","Please enter both Name and ID."); return
        if user_exists(code):
            messagebox.showerror("Already Exists",f"Someone with ID '{code}' is already registered."); return
        self.reg_btn.configure(state="disabled",text="Registering…")
        self.capture_active=True
        self.cap_bar.set(0); self.cap_pct_lbl.configure(text="0%")
        self.reg_cam_status.configure(text="● Live",fg=C["green"])
        self.round_lbl.configure(text="Starting camera…",fg=C["text3"])
        threading.Thread(target=self._reg_thread,args=(name,code),daemon=True).start()

    def _reg_thread(self,name,code):
        try:
            from mtcnn import MTCNN
        except ImportError:
            self.after(0,lambda:messagebox.showerror("Missing Package",
                "Please install MTCNN:\n\npip install mtcnn"))
            self.after(0,self._reset_reg_ui); return

        save_dir=os.path.join(FACE_DATA_DIR,code); os.makedirs(save_dir,exist_ok=True)
        
        cap = None
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened(): break
            
        if not cap or not cap.isOpened():
            self.after(0,lambda:messagebox.showerror("Camera Error","Cannot open webcam on any index."))
            self.after(0,self._reset_reg_ui); return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

        detector=MTCNN(); total_done=0; INTERVAL=0.10

        for _,(round_samples,round_tag,round_instr) in enumerate(CAPTURE_ROUNDS):
            if not self.capture_active: break
            self.after(0,lambda t=round_tag,i=round_instr:
                       self.round_lbl.configure(text=f"{t} — {i}",fg=C["accent"]))

            # Instruction screen (3 seconds)
            deadline=time.time()+3.0
            while time.time()<deadline and self.capture_active:
                ret,frame=cap.read()
                if not ret: break
                d=frame.copy(); h2,w2=d.shape[:2]
                overlay=d.copy()
                cv2.rectangle(overlay,(0,h2//2-70),(w2,h2//2+70),(240,234,224),-1)
                cv2.addWeighted(overlay,0.85,d,0.15,0,d)
                cv2.putText(d,round_tag,(40,h2//2-28),cv2.FONT_HERSHEY_SIMPLEX,0.90,(91,106,240),2)
                cv2.putText(d,round_instr,(40,h2//2+14),cv2.FONT_HERSHEY_SIMPLEX,0.60,(60,60,60),1)
                cv2.putText(d,f"Starting in {int(deadline-time.time())+1}s…",
                            (40,h2//2+48),cv2.FONT_HERSHEY_SIMPLEX,0.55,(34,197,94),1)
                self._push_reg_frame(d); time.sleep(0.05)

            round_done=0; last_cap=0
            while round_done<round_samples and self.capture_active:
                ret,frame=cap.read()
                if not ret: break
                d=frame.copy(); now=time.time()
                valid,rejected=detect_valid_faces(frame,detector)
                detected=False
                for result,face_224,x,y,w,h,conf in valid:
                    detected=True
                    cv2.rectangle(d,(x,y),(x+w,y+h),(91,106,240),2)
                    cl=16
                    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                        cv2.line(d,(px,py),(px+dx*cl,py),(255,255,255),3)
                        cv2.line(d,(px,py),(px,py+dy*cl),(255,255,255),3)
                    if now-last_cap>INTERVAL:
                        # Apply glasses mask before saving training image
                        face_to_save = face_224.copy()
                        kp = result.get('keypoints', {})
                        if 'left_eye' in kp and 'right_eye' in kp:
                            r2 = result
                            x2,y2,w2,h2 = r2['box']
                            x2,y2 = max(0,x2), max(0,y2)
                            pad2  = int(0.12*max(w2,h2))
                            x1_c  = max(0, x2-pad2); y1_c = max(0, y2-pad2)
                            x2_c  = min(frame.shape[1], x2+w2+pad2)
                            y2_c  = min(frame.shape[0], y2+h2+pad2)
                            cw    = max(1, x2_c-x1_c); ch = max(1, y2_c-y1_c)
                            le = ((kp['left_eye'][0]  - x1_c)*224.0/cw,
                                  (kp['left_eye'][1]  - y1_c)*224.0/ch)
                            re = ((kp['right_eye'][0] - x1_c)*224.0/cw,
                                  (kp['right_eye'][1] - y1_c)*224.0/ch)
                            face_to_save = mask_glasses_region(face_to_save, le, re)
                        cv2.imwrite(os.path.join(save_dir,f"img_{total_done+round_done:04d}.jpg"), face_to_save)
                        round_done+=1; last_cap=now
                total_vis=total_done+round_done
                bw=int((total_vis/SAMPLES_NEEDED)*(d.shape[1]-40))
                cv2.rectangle(d,(20,d.shape[0]-24),(d.shape[1]-20,d.shape[0]-8),(220,218,212),-1)
                cv2.rectangle(d,(20,d.shape[0]-24),(20+bw,d.shape[0]-8),(91,106,240),-1)
                col2=(34,197,94) if detected else (156,163,175)
                cv2.putText(d,f"{total_vis}/{SAMPLES_NEEDED} photos captured",
                            (20,34),cv2.FONT_HERSHEY_SIMPLEX,0.62,col2,2)
                self._push_reg_frame(d)
                self.after(0,lambda v=total_vis:self._update_cap_ui(v,"Scanning…"))
            total_done+=round_done

        cap.release(); self.capture_active=False
        if total_done<SAMPLES_NEEDED*0.7:
            self.after(0,lambda:messagebox.showwarning("Not Enough Photos",
                f"Only got {total_done} photos. Please try again with better lighting."))
            shutil.rmtree(save_dir,ignore_errors=True)
            self.after(0,self._reset_reg_ui); return
        self.after(0,lambda:self.cap_status_lbl.configure(
            text=f"✓ {total_done} photos taken. Building recognition model…",fg=C["green"]))
        try: save_user_db(name,code)
        except: pass
        self._train_arcface(name,code)

    def _push_reg_frame(self,frame):
        img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        wl=self.reg_cam_lbl.winfo_width(); hl=self.reg_cam_lbl.winfo_height()
        if wl>10 and hl>10: img=img.resize((wl,hl),Image.BILINEAR)
        imgtk=ImageTk.PhotoImage(img)
        self.after(0,lambda i=imgtk:self._set_reg_cam(i))

    def _set_reg_cam(self,imgtk):
        self.reg_cam_lbl.configure(image=imgtk,text=""); self.reg_cam_lbl.image=imgtk

    def _update_cap_ui(self,count,status):
        pct=count/SAMPLES_NEEDED
        self.cap_bar.set(pct)
        self.cap_pct_lbl.configure(text=f"{int(pct*100)}%")
        self.cap_count_lbl.configure(text=f"  of {SAMPLES_NEEDED} photos  [{count}]")
        self.cap_status_lbl.configure(text=status,fg=C["text3"])

    def _train_arcface(self,name,code):
        """Delegates to train_model_with_callback() in register.py."""
        self.after(0,lambda:self.train_status_lbl.configure(
            text="Building recognition model…",fg=C["text3"]))
        state = [0.0, 0.0, 0, 0]   # pct, acc, success, fail

        def cb(pct, acc, s, f):
            state[:] = [pct, acc, s, f]
            self.after(0, lambda p=pct,a=acc,ss=s,ff=f:
                       self._update_train_ui(p, a, ss, ff))

        train_model_with_callback(cb)
        self.after(0, lambda: self._finish_training(
            state[1], int(state[2]), int(state[3]), name))

    def _update_train_ui(self,pct,acc,success,fail):
        self.train_bar.set(pct); self.train_ring.set_value(acc)
        col=C["green"] if acc>=85 else C["amber"] if acc>=65 else C["red"]
        self.train_status_lbl.configure(text=f"{int(pct*100)}% done  ·  {success} photos processed",fg=col)
        self.train_detail_lbl.configure(text=f"Quality: {acc:.1f}%   Skipped: {fail} blurry photos")

    def _finish_training(self,acc,success,fail,name):
        self.train_bar.set(1.0); self.train_ring.set_value(acc)
        col=C["green"] if acc>=85 else C["amber"] if acc>=65 else C["red"]
        self.train_status_lbl.configure(
            text=f"Done! {success} photos used, quality {acc:.1f}%",fg=col)
        self.train_detail_lbl.configure(text=f"{fail} blurry photos were skipped automatically")
        self.reg_cam_status.configure(text="● Done",fg=C["green"])
        self.round_lbl.configure(text="Registration complete ✓",fg=C["green"])
        self._reset_reg_ui(done=True)
        self._refresh_users_list(); self._update_stats()
        messagebox.showinfo("All Done! 🎉",
            f"{name} has been registered.\n\nRecognition quality: {acc:.1f}%\nPhotos used: {success}")

    def _reset_reg_ui(self,done=False):
        self.reg_btn.configure(state="normal",text="Start Registration")
        if not done:
            self.cap_bar.set(0); self.cap_pct_lbl.configure(text="0%")
            self.reg_cam_status.configure(text="Off",fg=C["text3"])
            self.round_lbl.configure(text="—",fg=C["text3"])

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE ATTENDANCE
    # ══════════════════════════════════════════════════════════════════════════

    def _load_model_async(self):
        def _load():
            if not os.path.exists(MODEL_PATH): return
            try:
                self.embeddings_db=np.load(MODEL_PATH,allow_pickle=True).item()
                conn=sqlite3.connect(DB_PATH); c=conn.cursor()
                c.execute("SELECT user_code,name FROM users")
                self.all_users={code:name for code,name in c.fetchall()}
                conn.close()
                for _,code,*_ in get_today_attendance(): self.marked_today.add(code)
                # Invalidate mean-embedding cache so it rebuilds with new data
                if self.engine:
                    self.engine._mean_db = None
            except: pass
        threading.Thread(target=_load,daemon=True).start()

    def _start_webcam(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("No Model","Please register at least one person first."); return
        self.engine=WebcamEngine(
            on_frame=self._on_att_frame,
            on_recognized=self._update_live_display,
            on_fps=lambda f:self.after(0,lambda v=f:self.fps_lbl.configure(text=v)),
            embeddings_db=self.embeddings_db,
            all_users=self.all_users,
        )
        self.engine.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.att_cam_status.configure(text="● Live",fg=C["green"])
        self._refresh_today_log()
        self._poll_results()

    def _stop_webcam(self):
        if self.engine: self.engine.stop(); self.engine=None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.att_cam_status.configure(text="● Offline",fg=C["text3"])
        self.att_cam_lbl.configure(image="",
                                    text="📷\n\nClick Start to begin",
                                    font=FONT_BODY)

    def _on_att_frame(self,frame):
        now=time.time()
        if now-self._last_display<self._display_interval: return
        self._last_display=now
        wl=self.att_cam_lbl.winfo_width(); hl=self.att_cam_lbl.winfo_height()
        if wl<10 or hl<10: return
        img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img=img.resize((wl,hl),Image.BILINEAR)
        imgtk=ImageTk.PhotoImage(img)
        self._imgtk_att=imgtk
        self.after(0,lambda i=imgtk:self._set_att_cam(i))

    def _set_att_cam(self,imgtk):
        self.att_cam_lbl.configure(image=imgtk,text=""); self.att_cam_lbl.image=imgtk

    def _poll_results(self):
        if self.engine:
            if time.time() < getattr(self, "_ignore_recognition_until", 0):
                while self.engine.get_result() is not None: pass
                self.after(100,self._poll_results)
                return
            for _ in range(10):
                res=self.engine.get_result()
                if res is None: break
                user_code,conf_pct=res
                name=self.all_users.get(user_code,user_code)
                self._update_live_display(conf_pct,name,user_code)
                
                # ------ Hardware Trigger ------
                if self.arduino_serial:
                    try:
                        cmd = b'1' if user_code != "Unknown" else b'0'
                        self.arduino_serial.write(cmd)
                    except Exception as e:
                        print("Serial write error:", e)
                # ------------------------------
                
                marked=mark_attendance_db(user_code,conf_pct)
                if marked:
                    self.marked_today.add(user_code)
                    if self.engine: self.engine.update_overlay_mark(self.marked_today)
                    self._refresh_today_log(); self._update_stats()
        self.after(100,self._poll_results)

    def _update_live_display(self,conf,name,code):
        self.live_ring.set_value(conf)
        self.recog_name_lbl.configure(text=name)
        status="✓ Marked present" if code in self.marked_today else "Recognising…"
        color=C["green"] if code in self.marked_today else C["accent"]
        self.recog_code_lbl.configure(text=f"{code}  ·  {status}",fg=color)

    def _refresh_today_log(self):
        rows=get_today_attendance()
        self.log_listbox.delete(0,tk.END)
        for name,code,in_t,out_t,status,conf in rows:
            time_str = f"{in_t} - {out_t}" if out_t else f"{in_t} (IN)"
            self.log_listbox.insert(0,f"  ✓  {name:<22}  {time_str}   {conf:.0f}%")
        self.today_count_lbl.configure(text=str(len(rows)))

    # ══════════════════════════════════════════════════════════════════════════
    # CLEAR ATTENDANCE
    # ══════════════════════════════════════════════════════════════════════════

    def _open_clear_dialog(self):
        ClearAttendanceDialog(self,self._on_attendance_cleared)

    def _on_attendance_cleared(self,count,label_str):
        self.marked_today.clear()
        if self.engine: self.engine.update_overlay_mark(self.marked_today)
        self._ignore_recognition_until = time.time() + 3.0
        self._refresh_attendance_table()
        self._update_stats()
        if hasattr(self,"home_log"): self._refresh_home_log()
        if hasattr(self,"log_listbox"): self._refresh_today_log()
        messagebox.showinfo("Cleared",
            f"Removed {count} record(s) for {label_str}.")

    # ══════════════════════════════════════════════════════════════════════════
    # RECORDS & PEOPLE
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_attendance_table(self):
        df=getattr(self,"filter_from",None); dt=getattr(self,"filter_to",None); du=getattr(self,"filter_user",None)
        rows=get_attendance_filtered(
            df.get().strip() or None if df else None,
            dt.get().strip() or None if dt else None,
            du.get().strip() or None if du else None)
        for item in self.tree.get_children(): self.tree.delete(item)
        for name,code,date,in_t,out_t,status,conf in rows:
            time_str = f"{in_t} - {out_t}" if out_t else f"{in_t} --"
            tag="present" if status=="PRESENT" else "absent"
            self.tree.insert("","end",values=(name,code,date,time_str,status,f"{conf}%"),tags=(tag,))
        self.tree.tag_configure("present",foreground=C["green"])
        self.tree.tag_configure("absent",foreground=C["red"])
        if hasattr(self,"rec_count_lbl"):
            self.rec_count_lbl.configure(text=f"{len(rows)} records")

    def _refresh_users_list(self):
        if not hasattr(self,"users_tree"): return
        for item in self.users_tree.get_children(): self.users_tree.delete(item)
        for uid,name,code,reg_at in get_users():
            fd=os.path.join(FACE_DATA_DIR,code)
            smp=len([f for f in os.listdir(fd) if f.endswith(".jpg")]) if os.path.exists(fd) else 0
            conn=sqlite3.connect(DB_PATH); c=conn.cursor()
            c.execute("SELECT COUNT(*) FROM attendance WHERE user_id=?",(uid,))
            days=c.fetchone()[0]; conn.close()
            self.users_tree.insert("","end",values=(name,code,reg_at[:16],smp,days))

    def _delete_selected_user(self):
        sel=self.users_tree.selection()
        if not sel: messagebox.showinfo("Select Someone","Please select a person first.", parent=self); return
        vals=self.users_tree.item(sel[0])["values"]; name,code=vals[0],vals[1]
        if not messagebox.askyesno("Remove Person",
                f"Remove {name} ({code})?\n\nThis will delete all their photos and attendance records.", parent=self): return
        delete_user_db(code)
        fd=os.path.join(FACE_DATA_DIR,code)
        if os.path.exists(fd): shutil.rmtree(fd)
        self._refresh_users_list(); self._update_stats()
        messagebox.showinfo("Removed",f"{name} has been removed.", parent=self)

    def _delete_all_users(self):
        if not get_users():
            messagebox.showinfo("Empty","There are no users to delete.", parent=self); return
        if not messagebox.askyesno("Remove ALL Users",
                "Are you sure you want to remove ALL registered people?\n\nThis will permanently delete EVERYONE'S photos and attendance records. This cannot be undone.", parent=self): return
        delete_all_users_db()
        for root, dirs, files in os.walk(FACE_DATA_DIR, topdown=False):
            for name in files:
                if name != "_infer_face.jpg":
                    try: os.remove(os.path.join(root, name))
                    except: pass
            for name in dirs:
                try: os.rmdir(os.path.join(root, name))
                except: pass
        self._refresh_users_list(); self._update_stats()
        messagebox.showinfo("Removed","All users have been removed.", parent=self)

    def _export_csv(self):
        df=self.filter_from.get().strip() or None
        dt=self.filter_to.get().strip() or None
        du=self.filter_user.get().strip() or None
        rows=get_attendance_filtered(df,dt,du)
        if not rows: messagebox.showinfo("Nothing to Export","No records match your filter."); return
        path=filedialog.asksaveasfilename(defaultextension=".csv",
             filetypes=[("CSV","*.csv")],
             initialfile=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv")
        if path:
            with open(path,"w") as f:
                f.write("Name,Person ID,Date,Time,Status,Match %\n")
                for row in rows: f.write(",".join(str(v) for v in row)+"\n")
            messagebox.showinfo("Exported",f"Saved {len(rows)} records to:\n{path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STATS & CLOCK
    # ══════════════════════════════════════════════════════════════════════════

    def _update_stats(self):
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        today=datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT COUNT(*) FROM users"); reg=c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM attendance WHERE date=?",(today,)); present=c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM attendance"); total=c.fetchone()[0]
        conn.close()
        self.stat_reg.configure(text=str(reg))
        self.stat_today.configure(text=str(present))
        self.stat_total.configure(text=str(total))
        if hasattr(self,"home_stat_reg"):
            self.home_stat_reg.configure(text=str(reg))
            self.home_stat_today.configure(text=str(present))
            self.home_stat_total.configure(text=str(total))
            self._refresh_home_log()

    def _clock_tick(self):
        now=datetime.now()
        hour=now.hour
        greeting=("Good morning" if 5<=hour<12 else
                  "Good afternoon" if 12<=hour<17 else "Good evening")
        self.clock_lbl.configure(text=now.strftime("%a, %d %b  %H:%M"))
        # Update home greeting if visible
        self.after(1000,self._clock_tick)

    def on_close(self):
        self.capture_active=False
        if self.engine: self.engine.stop()
        self.destroy()

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    app=SmartAttendanceApp()
    app.protocol("WM_DELETE_WINDOW",app.on_close)
    app.mainloop()
