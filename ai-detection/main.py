"""
Object Detection with YOLOv8 - Scissors Alert System
=====================================================
Detects scissors (and other objects) via webcam.
When scissors are detected:
  - Draws a RED bounding box around them
  - Shows an alert message on screen
  - Plays a beep sound alert

Requirements:
    pip install ultralytics opencv-python numpy

Optional (for better beep sound):
    pip install playsound  (Windows/Mac)
    sudo apt-get install python3-gst-1.0  (Linux alternative)

Usage:
    python detect_scissors.py
    python detect_scissors.py --target "knife"        # detect knife instead
    python detect_scissors.py --target "scissors" "knife" "gun"  # multiple targets
    python detect_scissors.py --camera 1              # use second camera
    python detect_scissors.py --confidence 0.4        # lower confidence threshold
"""

import cv2
import numpy as np
import argparse
import time
import sys
import threading

# ── Try importing sound libraries ────────────────────────────────────────────
SOUND_METHOD = None
try:
    import winsound  # Windows built-in
    SOUND_METHOD = "winsound"
except ImportError:
    pass

if SOUND_METHOD is None:
    try:
        import subprocess
        subprocess.run(["beep", "--version"], capture_output=True, check=True)
        SOUND_METHOD = "beep_cmd"
    except Exception:
        pass

if SOUND_METHOD is None:
    try:
        import os
        SOUND_METHOD = "print_bell"   # fallback: terminal bell
    except Exception:
        pass

# ── YOLO import ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: 'ultralytics' package not found.")
    print("Install it with:  pip install ultralytics")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Sound helpers
# ─────────────────────────────────────────────────────────────────────────────

def play_alert_sound():
    """Play a beep/alert sound (non-blocking)."""
    def _play():
        if SOUND_METHOD == "winsound":
            for _ in range(3):
                winsound.Beep(1000, 200)
                time.sleep(0.1)
        elif SOUND_METHOD == "beep_cmd":
            import subprocess
            subprocess.run(["beep", "-f", "1000", "-l", "200",
                            "-n", "-f", "1000", "-l", "200"], capture_output=True)
        else:
            # Terminal bell — works in most terminals
            sys.stdout.write("\a\a\a")
            sys.stdout.flush()

    threading.Thread(target=_play, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colors  (BGR)
COLOR_ALERT   = (0,   0,   255)   # Red  – target object
COLOR_NORMAL  = (0,   200,  50)   # Green – other objects
COLOR_TEXT_BG = (0,   0,   200)
COLOR_WHITE   = (255, 255, 255)
COLOR_YELLOW  = (0,   220, 255)


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    # Lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180,  0,  90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270,  0,  90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90,  0,  90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0,  0,  90, color, thickness)


def draw_corner_marks(img, pt1, pt2, color, length=20, thickness=3):
    """Draw corner tick marks (like targeting reticle) on bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def put_label(img, text, pt, bg_color, text_color=COLOR_WHITE, font_scale=0.6, thickness=1):
    """Draw a filled rectangle label with text."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pt
    # Keep label inside frame
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + 10, y)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_alert_banner(img, message, alert_flash):
    """Draw a flashing red alert banner at the top of the frame."""
    h, w = img.shape[:2]
    if alert_flash:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, f"⚠  ALERT: {message}",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, COLOR_WHITE, 2, cv2.LINE_AA)
    return img


def draw_hud(img, fps, target_labels, detected_count):
    """Draw HUD info overlay in the bottom-left corner."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"FPS: {fps:.1f}",
        f"Watching: {', '.join(target_labels)}",
        f"Detected: {detected_count}",
        "Press Q to quit",
    ]
    y0 = h - (len(lines) * 22) - 10
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(img, line, (10, y), font, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(target_labels, camera_index=0, confidence=0.45):
    target_labels_lower = [t.lower().strip() for t in target_labels]
    print(f"\n{'='*55}")
    print(f"  YOLOv8 Object Detector — Alert System")
    print(f"{'='*55}")
    print(f"  Watching for : {', '.join(target_labels)}")
    print(f"  Camera index : {camera_index}")
    print(f"  Confidence   : {confidence}")
    print(f"  Press Q in the video window to quit")
    print(f"{'='*55}\n")

    # Load YOLOv8n (nano — fastest; swap to yolov8s/m for accuracy)
    print("Loading YOLOv8 model…")
    model = YOLO("yolov8n.pt")   # auto-downloads on first run
    print("Model loaded ✓\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    last_beep_time  = 0
    beep_cooldown   = 2.0      # seconds between beeps
    fps_timer       = time.time()
    frame_count     = 0
    fps             = 0.0
    alert_flash     = False
    flash_timer     = 0

    print("Camera open — starting detection…\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to grab frame. Retrying…")
            time.sleep(0.05)
            continue

        # ── Run inference ────────────────────────────────────────────────────
        results = model(frame, conf=confidence, verbose=False)[0]

        detected_targets = 0
        alert_names = []

        for box in results.boxes:
            cls_id     = int(box.cls[0])
            label      = model.names[cls_id].lower()
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_target = label in target_labels_lower

            if is_target:
                detected_targets += 1
                alert_names.append(label)
                # Thick red bounding box + corner marks
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ALERT, 3)
                draw_corner_marks(frame, (x1, y1), (x2, y2), COLOR_ALERT, length=22, thickness=4)
                # Pulsing red fill (semi-transparent)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                # Label
                put_label(frame, f"⚠ {label.upper()}  {conf_score:.0%}",
                          (x1, y1 - 8), COLOR_TEXT_BG)
            else:
                # Normal green box for non-target objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NORMAL, 2)
                put_label(frame, f"{label}  {conf_score:.0%}",
                          (x1, y1 - 8), (30, 130, 30))

        # ── Alert logic ──────────────────────────────────────────────────────
        if detected_targets > 0:
            now = time.time()
            if now - last_beep_time > beep_cooldown:
                play_alert_sound()
                last_beep_time = now
                print(f"[ALERT]  Detected: {', '.join(alert_names)}  "
                      f"(conf ≥ {confidence:.0%})")
            flash_timer = time.time()
            alert_flash = True
        else:
            # Keep flash for 1s after detection disappears
            if time.time() - flash_timer > 1.0:
                alert_flash = False

        # ── Overlays ─────────────────────────────────────────────────────────
        if alert_flash:
            draw_alert_banner(frame,
                              f"{', '.join(set(alert_names))} DETECTED!",
                              alert_flash)

        # FPS counter
        frame_count += 1
        if frame_count % 15 == 0:
            fps = 15 / (time.time() - fps_timer)
            fps_timer = time.time()

        draw_hud(frame, fps, target_labels, detected_targets)

        cv2.imshow("YOLOv8 Object Detector — Alert System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 real-time object detector with red-box alert."
    )
    parser.add_argument(
        "--target", nargs="+", default=["scissors"],
        help="Object class name(s) to trigger alert on. "
             "Must match COCO labels (e.g. scissors knife person). "
             "Default: scissors"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.45,
        help="Detection confidence threshold 0-1 (default: 0.45)"
    )
    args = parser.parse_args()
    run_detection(args.target, args.camera, args.confidence)