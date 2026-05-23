"""
YOLOv8 Threat Detector — Live Webcam + Red Box Alerts
======================================================
Run:  streamlit run app.py
Pip:  pip install streamlit ultralytics opencv-python numpy
"""

import streamlit as st
import cv2
import numpy as np
import time, threading, sys, logging, queue
from collections import deque
from datetime import datetime

for _n in ["streamlit.runtime.scriptrunner_utils.script_run_context",
           "streamlit.runtime.scriptrunner", "streamlit"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

st.set_page_config(page_title="Threat Detector", page_icon="🔪",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

html,body,[class*="css"]{font-family:'Barlow Condensed',sans-serif;background:#0a0c10;color:#c8d6e5;}
.stApp{background:#0a0c10;}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2d3d;}
[data-testid="stSidebar"] *{font-family:'Barlow Condensed',sans-serif;}
h1,h2,h3{font-family:'Barlow Condensed',sans-serif;letter-spacing:.08em;text-transform:uppercase;}
[data-testid="metric-container"]{background:#0d1117!important;border:1px solid #1e2d3d;border-radius:4px;padding:12px 16px;}

.stButton>button{font-family:'Share Tech Mono',monospace;font-size:.85rem;letter-spacing:.12em;
  text-transform:uppercase;border-radius:2px;border:1px solid #2a6496;
  background:#0d1117;color:#5bc0de;transition:all .15s;}
.stButton>button:hover{background:#2a6496;color:#fff;}
div[data-testid="column"]:nth-child(1) .stButton>button{border-color:#27ae60;color:#2ecc71;}
div[data-testid="column"]:nth-child(1) .stButton>button:hover{background:#27ae60;color:#fff;}
div[data-testid="column"]:nth-child(2) .stButton>button{border-color:#c0392b;color:#e74c3c;}
div[data-testid="column"]:nth-child(2) .stButton>button:hover{background:#c0392b;color:#fff;}

.alert-card{background:#1a0505;border:1px solid #c0392b;border-left:4px solid #e74c3c;
  border-radius:4px;padding:12px 16px;margin-bottom:8px;
  font-family:'Share Tech Mono',monospace;animation:slidein .3s ease;}
@keyframes slidein{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}
.alert-card .alabel{font-size:1rem;color:#e74c3c;font-weight:bold;letter-spacing:.1em;}
.alert-card .ats{font-size:.7rem;color:#5d7a99;margin-top:4px;}
.alert-card .aconf{font-size:.75rem;color:#e8a87c;margin-top:2px;}

.pill{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:.7rem;
  letter-spacing:.12em;padding:4px 12px;border-radius:20px;text-transform:uppercase;}
.pill-live{background:#0d2b1a;color:#2ecc71;border:1px solid #27ae60;}
.pill-idle{background:#141a25;color:#5d7a99;border:1px solid #2a3a4a;}
.pill-alert{background:#2b0d0d;color:#e74c3c;border:1px solid #c0392b;
            animation:blink .5s step-start infinite;}

@keyframes blink{50%{opacity:0}}
hr{border-color:#1e2d3d;}
[data-baseweb="tag"]{background:#1e2d3d!important;}

.empty-log{background:#0d1117;border:1px solid #1e2d3d;border-radius:4px;
  padding:30px;text-align:center;font-family:'Share Tech Mono',monospace;
  font-size:.75rem;color:#3a4a5a;letter-spacing:.1em;}

.feed-label{font-family:'Share Tech Mono',monospace;font-size:.7rem;
  letter-spacing:.14em;color:#5d7a99;margin-bottom:6px;}

.idle-feed{background:#0d1117;border:1px solid #1e2d3d;border-radius:4px;
  height:420px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:12px;}
</style>
""", unsafe_allow_html=True)

# ── Sound ─────────────────────────────────────────────────────────────────────
SOUND_METHOD = None
try:
    import winsound; SOUND_METHOD = "winsound"
except ImportError: pass
if not SOUND_METHOD:
    try:
        import subprocess
        subprocess.run(["beep","--version"],capture_output=True,check=True)
        SOUND_METHOD = "beep_cmd"
    except: pass
if not SOUND_METHOD:
    SOUND_METHOD = "bell"

def play_beep():
    """Loud siren-style alert — 7 rising+falling cycles, ~7 seconds total."""
    def _b():
        logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

        if SOUND_METHOD == "winsound":
            # winsound.Beep(frequency_hz, duration_ms)
            # 400Hz = deep rumble | 2000Hz = sharp piercing
            # 7 cycles × ~900ms = ~7 seconds total
            for _ in range(7):
                # Rising — 400Hz up to 2000Hz
                for freq, dur in [
                    (400, 150),
                    (600, 150),
                    (800, 150),
                    (1000,150),
                    (1200,150),
                    (1500,150),
                    (1800,150),
                    (2000,300),   # peak — sharpest, most attention-grabbing
                ]:
                    winsound.Beep(freq, dur)
                time.sleep(0.03)
                # Falling — 2000Hz back down to 400Hz
                for freq, dur in [
                    (2000,150),
                    (1800,150),
                    (1500,150),
                    (1200,150),
                    (1000,150),
                    (800, 150),
                    (600, 150),
                    (400, 300),   # deep rumble finish
                ]:
                    winsound.Beep(freq, dur)
                time.sleep(0.05)

        elif SOUND_METHOD == "beep_cmd":
            import subprocess
            pattern = []
            for _ in range(7):
                for f,d in [
                    (400,150),(600,150),(800,150),(1000,150),
                    (1200,150),(1500,150),(1800,150),(2000,300),
                    (2000,150),(1800,150),(1500,150),(1200,150),
                    (1000,150),(800,150),(600,150),(400,300),
                ]:
                    pattern += ["-f", str(f), "-l", str(d), "-n"]
            if pattern and pattern[-1] == "-n":
                pattern = pattern[:-1]
            subprocess.run(["beep"] + pattern, capture_output=True)

        else:
            # Terminal bell fallback — 15 bells
            for _ in range(15):
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(0.10)

    threading.Thread(target=_b, daemon=True).start()

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(name):
    from ultralytics import YOLO
    return YOLO(name)

MODEL_OPTIONS = {
    "YOLOv8n  ⚡"      : "yolov8n.pt",
    "YOLOv8s  ⚖"     : "yolov8s.pt",
    "YOLOv8m  🎯": "yolov8m.pt",
    "YOLOv9c  🚀"  : "yolov9c.pt",
    "YOLOv9e  🏆"  : "yolov9e.pt",
}

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_RED   = (0,   0,   255)
C_GREEN = (0,   200,  50)
C_WHITE = (255, 255, 255)
C_BLACK = (0,   0,     0)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_corner_marks(img, pt1, pt2, color, length=22, thickness=4):
    x1,y1 = pt1; x2,y2 = pt2
    for p1,p2 in [
        ((x1,y1),(x1+length,y1)), ((x1,y1),(x1,y1+length)),
        ((x2,y1),(x2-length,y1)), ((x2,y1),(x2,y1+length)),
        ((x1,y2),(x1+length,y2)), ((x1,y2),(x1,y2-length)),
        ((x2,y2),(x2-length,y2)), ((x2,y2),(x2,y2-length)),
    ]:
        cv2.line(img, p1, p2, color, thickness)


def put_label(img, text, pt, bg_color, text_color=C_WHITE):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),bl = cv2.getTextSize(text, font, 0.6, 1)
    x,y = pt
    x = max(0, min(x, img.shape[1]-tw-8))
    y = max(th+10, y)
    cv2.rectangle(img, (x-4,y-th-6), (x+tw+4,y+bl), bg_color, -1)
    cv2.putText(img, text, (x,y), font, 0.6, text_color, 1, cv2.LINE_AA)


def draw_alert_banner(img, names, active):
    """Flashing red banner at top of frame."""
    if not active or not names: return
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (w,56), (0,0,160), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    txt = "  ⚠  ALERT — " + "  ·  ".join(n.upper() for n in names) + "  DETECTED"
    cv2.putText(img, txt, (14,36),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, C_WHITE, 2, cv2.LINE_AA)


def draw_hud(img, fps, targets, alert_count):
    """Bottom-left HUD overlay."""
    h, w = img.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Watching: {', '.join(t.upper() for t in targets)}",
        f"Alerts: {alert_count}",
        "Press STOP to end",
    ]
    y0 = h - len(lines)*22 - 10
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160,160,160), 1, cv2.LINE_AA)


def process_frame(frame, model, targets, conf, imgsz,
                  flash_on, show_all, min_ratio, use_tta, use_multiscale):
    """
    Run YOLO on frame, draw red boxes on targets, green on others.
    Returns annotated RGB frame + detected target labels + confs.
    """
    h,w = frame.shape[:2]
    found_labels, found_confs = [], []

    # ── Primary inference ─────────────────────────────────────────────────────
    res = model(frame, conf=conf, verbose=False,
                imgsz=imgsz, iou=0.4, augment=use_tta)[0]

    target_boxes = []   # (x1,y1,x2,y2,label,score)

    for box in res.boxes:
        label = model.names[int(box.cls[0])].lower()
        score = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        bw,bh = x2-x1, y2-y1
        if bw < w*min_ratio or bh < h*min_ratio: continue
        if bw > w*0.92     or bh > h*0.92:       continue

        if label in targets:
            target_boxes.append((x1,y1,x2,y2,label,score))
            found_labels.append(label)
            found_confs.append(score)
        elif show_all:
            cv2.rectangle(frame, (x1,y1), (x2,y2), C_GREEN, 2)
            put_label(frame, f"{label}  {score:.0%}", (x1,y1-8), (20,100,20))

    # ── Multi-scale centre crop pass (only if primary missed targets) ─────────
    if use_multiscale and not found_labels:
        pad_x, pad_y = int(w*0.12), int(h*0.12)
        crop = frame[pad_y:h-pad_y, pad_x:w-pad_x]
        if crop.size > 0:
            try:
                res2 = model(crop, conf=max(0.10,conf-0.05), verbose=False,
                             imgsz=imgsz, iou=0.4)[0]
                ch,cw = crop.shape[:2]
                for box in res2.boxes:
                    label = model.names[int(box.cls[0])].lower()
                    score = float(box.conf[0])
                    cx1,cy1,cx2,cy2 = map(int, box.xyxy[0])
                    bw,bh = cx2-cx1, cy2-cy1
                    if bw < cw*min_ratio or bh < ch*min_ratio: continue
                    if label in targets:
                        # Map crop coords back to original frame
                        ox1,oy1 = cx1+pad_x, cy1+pad_y
                        ox2,oy2 = cx2+pad_x, cy2+pad_y
                        target_boxes.append((ox1,oy1,ox2,oy2,label,score))
                        found_labels.append(label)
                        found_confs.append(score)
            except: pass

    # ── Draw target boxes (red) ───────────────────────────────────────────────
    for (x1,y1,x2,y2,label,score) in target_boxes:
        # Thick red rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), C_RED, 3)
        # Corner targeting marks
        draw_corner_marks(frame, (x1,y1), (x2,y2), C_RED)
        # Semi-transparent red fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,180), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        # Label
        put_label(frame, f"⚠ {label.upper()}  {score:.0%}", (x1,y1-8), (0,0,180))

    # ── Alert banner ──────────────────────────────────────────────────────────
    draw_alert_banner(frame, list(set(found_labels)), flash_on and bool(found_labels))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, found_labels, found_confs


# ── Session state ─────────────────────────────────────────────────────────────
for k,v in dict(running=False, alert_log=deque(maxlen=100),
                total_alerts=0, total_frames=0, fps=0.0).items():
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 4px'>
      <span style='font-family:Share Tech Mono,monospace;font-size:.65rem;
                   letter-spacing:.2em;color:#5d7a99;'>THREAT MONITOR</span><br>
      <span style='font-size:1.5rem;font-weight:700;letter-spacing:.05em;
                   color:#c8d6e5;text-transform:uppercase;'>Alert System</span>
    </div><hr style='margin:6px 0 16px'>
    """, unsafe_allow_html=True)

    # Model selector
    st.markdown("##### 🤖 YOLO Model")
    model_label = st.radio("model", list(MODEL_OPTIONS.keys()),
                           index=1, label_visibility="collapsed")
    model_file  = MODEL_OPTIONS[model_label]
    st.caption(f"File: `{model_file}`")

    st.markdown("---")

    # Target objects
    st.markdown("##### 🎯 Target Objects")
    q1,q2,q3 = st.columns(3)
    preset_sk = q1.button("✂+🔪", use_container_width=True, help="Scissors & Knife")
    preset_s  = q2.button("✂ Only", use_container_width=True)
    preset_k  = q3.button("🔪 Only", use_container_width=True)

    if "target_preset" not in st.session_state:
        st.session_state.target_preset = ["scissors","knife"]
    if preset_sk: st.session_state.target_preset = ["scissors","knife"]
    if preset_s:  st.session_state.target_preset = ["scissors"]
    if preset_k:  st.session_state.target_preset = ["knife"]

    target_objects = st.multiselect("Objects", COCO_CLASSES,
                                    default=st.session_state.target_preset,
                                    label_visibility="collapsed")
    custom = st.text_input("Custom label", placeholder="e.g. gun")
    if custom.strip():
        target_objects = list(set(target_objects + [custom.strip().lower()]))

    st.markdown("---")

    # Detection settings
    st.markdown("##### ⚙️ Detection Settings")
    camera_idx  = 0
    confidence  = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05, format="%.2f")
    infer_size  = st.select_slider("Inference resolution", [320,416,480,640], value=640)
    beep_cool   = st.slider("Beep cooldown (s)", 0.5, 10.0, 2.0, 0.5, format="%.1fs")
    min_ratio   = st.slider("Min object size", 3, 30, 5, 1, format="%d%%") / 100.0
    show_all    = st.checkbox("Show all objects", value=True,
                               help="Also draw green boxes on non-target objects")

    st.markdown("---")
    st.markdown("##### 🔬 Accuracy Boosters")
    use_tta        = st.checkbox("Test-time augmentation (TTA)", value=True)
    use_multiscale = st.checkbox("Multi-scale centre crop pass", value=True)

    st.markdown("---")
    st.markdown("##### 🔊 Sound")
    sound_on = st.checkbox("Enable alert beep", value=True)
    st.caption(f"Method: `{SOUND_METHOD}`")

    st.markdown("---")
    c1,c2 = st.columns(2)
    start_btn = c1.button("▶  START", use_container_width=True)
    stop_btn  = c2.button("■  STOP",  use_container_width=True)

    if start_btn and target_objects:
        st.session_state.running      = True
        st.session_state.total_alerts = 0
        st.session_state.total_frames = 0
        st.session_state.fps          = 0.0
        st.session_state.alert_log.clear()
    if start_btn and not target_objects:
        st.warning("Select at least one target.")
    if stop_btn:
        st.session_state.running = False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
hcol, bcol = st.columns([5,1])
with hcol:
    st.markdown("""
    <h1 style='margin:0;font-size:1.9rem;letter-spacing:.08em;color:#c8d6e5;font-weight:700;'>
         REAL-TIME CRIME DETECTION & ALERT
    </h1>
    <p style='margin:2px 0 0;font-family:Share Tech Mono,monospace;
              font-size:.7rem;color:#5d7a99;letter-spacing:.12em;'>
        YOLOV8 / YOLOV9 · LIVE WEBCAM · RED BOX ALERT · REAL-TIME DASHBOARD
    </p>""", unsafe_allow_html=True)
with bcol:
    pill = "pill-live" if st.session_state.running else "pill-idle"
    dot  = "●" if st.session_state.running else "○"
    txt  = "LIVE" if st.session_state.running else "IDLE"
    st.markdown(f'<span class="pill {pill}">{dot} {txt}</span>', unsafe_allow_html=True)

st.markdown("<hr style='margin:10px 0 14px'>", unsafe_allow_html=True)

# Metrics
m1,m2,m3,m4,m5 = st.columns(5)
fps_ph    = m1.empty()
frames_ph = m2.empty()
alerts_ph = m3.empty()
model_ph  = m4.empty()
conf_ph   = m5.empty()

def refresh_metrics():
    fps_ph.metric("FPS",             f"{st.session_state.fps:.1f}")
    frames_ph.metric("Frames",       st.session_state.total_frames)
    alerts_ph.metric("Total Alerts", st.session_state.total_alerts)
    model_ph.metric("Model",         model_file.replace(".pt",""))
    conf_ph.metric("Confidence",     f"{confidence:.0%}")

refresh_metrics()
st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)

# ── Two-column: webcam feed | alert log ───────────────────────────────────────
feed_col, log_col = st.columns([3,1], gap="medium")

with feed_col:
    st.markdown("<p class='feed-label'>CAMERA FEED</p>", unsafe_allow_html=True)
    frame_ph = st.empty()

with log_col:
    lh1,lh2 = st.columns([3,1])
    lh1.markdown("<p class='feed-label'>ALERT LOG</p>", unsafe_allow_html=True)
    if lh2.button("🗑 Clear", use_container_width=True):
        st.session_state.alert_log.clear()
        st.session_state.total_alerts = 0
        st.rerun()

    log_ph = st.empty()

    st.markdown("---")
    st.markdown("<p class='feed-label'>WATCHING</p>", unsafe_allow_html=True)
    for t in (target_objects or ["—"]):
        color = "#e74c3c" if t in ["scissors","knife"] else "#e8a87c"
        st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:.8rem;"
                    f"color:{color};padding:2px 0;'>⬡ {t.upper()}</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""""", unsafe_allow_html=True)


def render_log():
    if not st.session_state.alert_log:
        log_ph.markdown(
            "<div class='empty-log'>NO ALERTS YET</div>", unsafe_allow_html=True)
        return
    rows = ""
    for ev in list(st.session_state.alert_log):
        label_str = " · ".join(l.upper() for l in ev["labels"])
        avg_conf  = sum(ev["confs"])/len(ev["confs"]) if ev["confs"] else 0
        rows += f"""
        <div class='alert-card'>
          <div class='alabel'>⚠ {label_str}</div>
          <div class='ats'>🕐 {ev['timestamp']}</div>
          <div class='aconf'>Conf: {avg_conf:.0%}</div>
        </div>"""
    log_ph.markdown(rows, unsafe_allow_html=True)


# ── Idle placeholder ──────────────────────────────────────────────────────────
if not st.session_state.running:
    frame_ph.markdown("""
    <div class='idle-feed'>
      <span style='font-size:3rem;opacity:.25;'>📷</span>
      <span style='font-family:Share Tech Mono,monospace;font-size:.8rem;
                   letter-spacing:.14em;color:#3a4a5a;'>PRESS START TO BEGIN</span>
    </div>""", unsafe_allow_html=True)
    render_log()


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    with st.spinner(f"Loading {model_file}…"):
        model = load_model(model_file)

    cap = cv2.VideoCapture(int(camera_idx))
    if not cap.isOpened():
        st.error(f"Cannot open camera {camera_idx}.")
        st.session_state.running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Pre-warm model so first frame isn't slow
    dummy = np.zeros((480,640,3), dtype=np.uint8)
    try: model(dummy, verbose=False, imgsz=infer_size)
    except: pass

    targets_lower  = [t.lower() for t in target_objects]
    last_beep      = 0
    flash_until    = 0
    fps_timer      = time.time()
    frame_buf      = 0
    SKIP           = 2      # run YOLO every 2nd frame → smoother display

    last_detected  = []
    last_confs     = []

    try:
        while st.session_state.running:
            # Always grab fresh frame — drain OS buffer
            cap.grab(); cap.grab()
            ret, frame = cap.retrieve()
            if not ret: time.sleep(0.03); continue

            frame_buf += 1
            st.session_state.total_frames += 1

            # FPS counter
            if frame_buf % 15 == 0:
                st.session_state.fps = 15 / (time.time() - fps_timer + 1e-9)
                fps_timer = time.time()

            now      = time.time()
            flash_on = now < flash_until

            # Run YOLO every SKIP frames; re-use last result on skipped frames
            if frame_buf % SKIP == 0:
                try:
                    frame_rgb, detected, confs = process_frame(
                        frame, model, targets_lower, confidence,
                        infer_size, flash_on, show_all, min_ratio,
                        use_tta, use_multiscale
                    )
                    last_detected = detected
                    last_confs    = confs
                except Exception as e:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_detected = []; last_confs = []
            else:
                # Skipped frame — draw banner only, no YOLO
                draw_alert_banner(frame, last_detected, flash_on and bool(last_detected))
                draw_hud(frame, st.session_state.fps, targets_lower,
                         st.session_state.total_alerts)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected, confs = last_detected, last_confs

            # Draw HUD on YOLO frames too
            if frame_buf % SKIP == 0:
                draw_hud(frame, st.session_state.fps, targets_lower,
                         st.session_state.total_alerts)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Alert trigger
            if detected:
                flash_until = now + 1.5
                if now - last_beep > beep_cool:
                    if sound_on: play_beep()
                    last_beep = now
                    st.session_state.total_alerts += 1
                    avg_conf = sum(confs)/len(confs) if confs else 0
                    st.session_state.alert_log.appendleft({
                        "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        "labels":    list(set(detected)),
                        "confs":     confs,
                    })

            # Display frame
            frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)

            # Refresh metrics + log every 20 frames
            if frame_buf % 20 == 0:
                refresh_metrics()
                render_log()

    finally:
        cap.release()
        st.session_state.running = False
        refresh_metrics()
        render_log()