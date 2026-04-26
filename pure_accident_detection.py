import cv2
import os
import requests
import base64
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
# Replace with the path to your test video
VIDEO_PATH   = 'test_2.mp4'   
OUTPUT_VIDEO = 'accident_output.mp4'

# ── Roboflow API parameters ───────────────────────────────────
API_KEY = os.getenv("ACCIDENT_API_KEY")
MODEL_ID = os.getenv("ACCIDENT_MODEL_ID")
VERSION = os.getenv("ACCIDENT_VERSION", "1")

ROBOFLOW_API_URL = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}"

# ── Single parameter: confidence threshold ────────────────────
# How confident the model must be to mark a frame as ACCIDENT
# 0.30 = more detections (may have false positives)
# 0.55 = fewer detections (may miss some)
CONF = 0.30

# ─────────────────────────────────────────────────────────────
print(f'Model API URL: {ROBOFLOW_API_URL}')
print(f'Confidence threshold: {CONF}')

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: Video file {VIDEO_PATH} not found.")
    exit(1)

cap          = cv2.VideoCapture(VIDEO_PATH)
fps          = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 20.0

W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (W, H)
)

frame_idx         = 0
accident_frames   = 0
no_accident_frames= 0

print(f'\n🎬 Processing {total_frames} frames one by one ...\n')
print(f"⚠️ Note: Making an API call for every frame may take some time depending on your connection speed.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ── Run Roboflow on this single frame ─────────────────────────
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        continue
    img_str = base64.b64encode(buffer).decode('utf-8')

    # Construct the API request
    # Roboflow API expects confidence as an integer between 0 and 100
    conf_int = int(CONF * 100)
    params = f"?api_key={API_KEY}&confidence={conf_int}&overlap=50"
    url = ROBOFLOW_API_URL + params
    
    try:
        response = requests.post(
            url,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            results = response.json()
            predictions = results.get("predictions", [])
        else:
            print(f"Error {response.status_code}: {response.text}")
            predictions = []
    except Exception as e:
        print(f"Request failed: {e}")
        predictions = []

    # ── Check each detected box ───────────────────────────────
    is_accident   = False
    max_conf      = 0.0
    accident_boxes= []

    for pred in predictions:
        cls_name = pred.get("class", "").lower()
        conf_val = pred.get("confidence", 0.0)
        
        # If confidence is provided as a fraction instead of whole number by the API
        if conf_val > 1.0:
            conf_val = conf_val / 100.0
            
        # If your roboflow class names are different, you can add them here
        if any(kw in cls_name for kw in ['accident', 'crash', 'collision']):
            is_accident = True
            max_conf    = max(max_conf, conf_val)
            
            # Roboflow returns x, y (center) and width, height
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            accident_boxes.append((x1, y1, x2, y2, conf_val, cls_name))

    # ── Draw result on frame ──────────────────────────────────
    annotated = frame.copy()

    # Draw boxes
    for (x1, y1, x2, y2, conf_val, cls_name) in accident_boxes:
        # Red box for accident
        cv2.rectangle(annotated, (x1,y1),(x2,y2), (0,0,220), 3)
        label = f'{cls_name} {conf_val:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+6, y1), (0,0,220), -1)
        cv2.putText(annotated, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # Top banner — RED if accident, GREEN if clear
    if is_accident:
        accident_frames += 1
        banner_col  = (0, 0, 180)
        banner_text = f'ACCIDENT  conf={max_conf:.2f}'
    else:
        no_accident_frames += 1
        banner_col  = (0, 140, 0)
        banner_text = 'NO ACCIDENT'

    # Semi-transparent banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0,0), (W, 55), banner_col, -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
    cv2.putText(annotated, banner_text,
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255,255,255), 2, cv2.LINE_AA)

    # Frame counter bottom-right
    cv2.putText(annotated,
                f'Frame {frame_idx}/{total_frames}  |  conf>={CONF}',
                (W-320, H-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200,200,200), 1)

    out.write(annotated)

    if frame_idx % 10 == 0:
        pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
        print(f'  [{frame_idx:>5}/{total_frames}] {pct:.1f}%  '
              f'accident_frames={accident_frames}  '
              f'clear_frames={no_accident_frames}')

cap.release()
out.release()

# ── Summary ───────────────────────────────────────────────────
pct_accident = (accident_frames / total_frames * 100) if total_frames > 0 else 0
print(f'\n{"="*55}')
print('  ✅  FRAME-BY-FRAME INFERENCE COMPLETE')
print(f'{"="*55}')
print(f'  Total frames       : {total_frames}')
print(f'  Accident frames    : {accident_frames}  ({pct_accident:.1f}%)')
print(f'  Clear frames       : {no_accident_frames}  ({100-pct_accident:.1f}%)')
print(f'  Conf threshold     : {CONF}')
print(f'{"="*55}')

print("""
  📌 WHAT THIS TELLS YOU:
  ─────────────────────────────────────────────────────
  If accident_frames is very high (> 30% of video):
    → Model is too sensitive. Raise CONF to 0.55 or 0.60

  If accident_frames is very low even during real crash:
    → Model is not recognizing it. Lower CONF to 0.30

  If early frames are also marked ACCIDENT:
    → This is the fundamental image-model limitation.
       The model sees a "dangerous looking" frame.
       Only a video-based model (LSTM/3D-CNN) can fix this.
  ─────────────────────────────────────────────────────
""")
