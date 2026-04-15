import asyncio
import cv2
import time
import base64
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from inference_engine import InferenceEngine, run_ocr_tesseract
from math_utils import ByteTrackWrapper, SpeedCalculator, AccidentVerificationEngine
from traffic_manager import TrafficSignalController
import uvicorn
import os

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic
    for engine in engines:
        await engine.cleanup()


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="Smart City AI Traffic Manager", lifespan=lifespan)
# Template configuration - serve from templates/ directory
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Traffic Controller (4 Lanes)
NUM_LANES = 4

class CalibrationRequest(BaseModel):
    lane_idx: int
    p1: float
    p2: float
    real_m: float = 3.0

traffic_controller = TrafficSignalController(num_lanes=NUM_LANES)

# Configurable line coordinates per lane to match video perspective
# Format: (line_a_y, line_b_y)
LANE_LINES = [
    (250, 400), # Lane 1 - Adjusted for new video
    (250, 350), # Lane 2
    (300, 400), # Lane 3
    (300, 400)  # Lane 4
]

# Instantiate engines and trackers per lane
engines = [InferenceEngine(skip_frames=3) for _ in range(NUM_LANES)]
trackers = [ByteTrackWrapper(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=20) for _ in range(NUM_LANES)]

# Speed Trap logic:
# sequence_pixels = pixel distance representing 9m (3m dash + 6m gap)
speed_calcs = [
    SpeedCalculator(line_a_y=LANE_LINES[i][0], line_b_y=LANE_LINES[i][1], sequence_pixels=abs(LANE_LINES[i][0] - LANE_LINES[i][1]), lane_id=i) 
    for i in range(NUM_LANES)
]
accident_verifiers = [AccidentVerificationEngine(verification_buffer_sec=5.0) for _ in range(NUM_LANES)]

# Global UI Setting State
ui_state = {
    "police_override": False,
    "global_speed_limit": 60.0,
    "debug_lane": 0,  # Default debug for lane 1 (index 0)
    "display_mode": "both",  # "speed", "bbox", or "both"
    "mode": "general"  # "general" or "testing"
}

# Pre-open all video captures at module level for instant lane switching
VIDEO_SOURCES = ["speedtest.mp4", "test_2.mp4", "test_3.mp4", "test_4.mov"]
global_caps = []
for _src in VIDEO_SOURCES:
    _cap = cv2.VideoCapture(_src)
    if not _cap.isOpened():
        _cap = cv2.VideoCapture(0)
    global_caps.append(_cap)

def draw_lane_info(frame, lane_idx, predictions):
    lane = traffic_controller.lanes[lane_idx]
    line_a, line_b = LANE_LINES[lane_idx]
    
    # Draw Traffic Light Indicator in Top Left
    light_color = (0, 0, 255) # RED
    if lane.light_state == "GREEN":
        light_color = (0, 255, 0)
    elif lane.light_state == "YELLOW":
        light_color = (0, 255, 255)
        
    cv2.circle(frame, (50, 50), 30, light_color, -1)
    cv2.circle(frame, (50, 50), 30, (255, 255, 255), 2)
    
    cv2.putText(frame, f"LANE {lane_idx+1} | PCU: {lane.pcu_density:.1f}", (90, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    # Show speed limit on frame
    speed_limit = ui_state.get("global_speed_limit", 60.0)
    cv2.putText(frame, f"SPEED LIMIT: {speed_limit:.0f} km/h", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if lane.ambulance_detected:
         cv2.putText(frame, "AMBULANCE", (90, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    if lane.accident_detected:
         cv2.putText(frame, "ACCIDENT!", (90, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
         
    # Speed Trap Lines (Dynamic per lane)
    cv2.line(frame, (0, line_a), (frame.shape[1], line_a), (0, 255, 0), 2)
    cv2.line(frame, (0, line_b), (frame.shape[1], line_b), (0, 0, 255), 2)
    cv2.putText(frame, "LINE A", (5, line_a - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "LINE B", (5, line_b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    display_mode = ui_state.get("display_mode", "both")
    show_bbox = display_mode in ("bbox", "both")
    show_speed = display_mode in ("speed", "both")

    # Draw current predictions with Class Labels
    if show_bbox:
        for pred in predictions:
            startX, startY, endX, endY = pred["bbox"]
            label = f"{pred['class']} ({pred['confidence']:.2f})"
            
            # Color based on class
            cls_lower = pred['class'].lower()
            is_speeding = pred.get("is_speeding", False)
            
            if is_speeding:
                 color = (0, 0, 255) # BRIGHT RED for overspeed
            elif 'ambulance' in cls_lower:
                color = (0, 255, 0) # Green
            elif 'police' in cls_lower or 'person' in cls_lower:
                color = (255, 255, 0) # Cyan/Yellowish
            elif 'accident' in cls_lower or 'crash' in cls_lower:
                color = (0, 0, 255) # Red
            else:
                color = (255, 0, 0) # Blue (Generic)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 4 if is_speeding else 2)
            
            # Draw Label Box
            label_text = f"OVERSPEED: {label}" if is_speeding else label
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (startX, startY - 20), (startX + w, startY), color, -1)
            cv2.putText(frame, label_text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw speed on tracked vehicles with contrasting background
    if show_speed:
        for objectID, speed in lane.speeds.items():
            if objectID in lane.bboxes:
                bbox = lane.bboxes[objectID]
                cx = (bbox[0] + bbox[2]) // 2
                cy = bbox[3] + 18  # Below bbox
                speed_text = f"{speed:.0f} km/h"
                speed_color = (0, 0, 255) if speed > speed_limit else (0, 255, 0)
                (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Draw black background rectangle for contrast
                cv2.rectangle(frame, (cx - 35, cy - th - 5), (cx - 35 + tw + 10, cy + 5), (0, 0, 0), -1)
                cv2.rectangle(frame, (cx - 35, cy - th - 5), (cx - 35 + tw + 10, cy + 5), speed_color, 2)
                cv2.putText(frame, speed_text, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)

    return frame

async def process_lane(frame, lane_idx, video_time_sec=None):
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
        
    frame = cv2.resize(frame, (640, 480))
    engine = engines[lane_idx]
    lane = traffic_controller.lanes[lane_idx]
    
    # 1. Inference Engine Detection (Async)
    run_all = (engine.frame_count % 15 == 0)
    out = await engine.process_frame(frame, run_all_checks=run_all)
    
    predictions = out["traffic"]
    if run_all:
        lane.ambulance_detected = out["ambulance"]
        lane.accident_detected = out["accident"]
        if out["police"]:
            traffic_controller.police_detected = True

    lane.update_pcu(predictions)
    
    rects = [p["bbox"] for p in predictions]
    
    # 2. Tracking System
    lane.objects, lane.bboxes = trackers[lane_idx].update(rects)
    lane.speeds = speed_calcs[lane_idx].update(lane.objects, video_time_sec=video_time_sec)
    
    # Debug: Log vehicle centroid Y positions vs trap lines every 60 frames for ALL lanes
    if engine.frame_count % 60 == 0:
        line_a, line_b = LANE_LINES[lane_idx]
        sc = speed_calcs[lane_idx]
        centroids_y = {oid: c[1] for oid, c in lane.objects.items()}
        print(f"[L{lane_idx}] Detections={len(predictions)} Tracked={len(lane.objects)} Lines=({line_a},{line_b}) Crossings={len(sc.object_crossing)} Speeds={lane.speeds} Y={centroids_y}")
    
    # Accident verification model fallback check
    if accident_verifiers[lane_idx].check_accident(lane.objects, lane.bboxes, lane.speeds):
        lane.accident_detected = True
        
    # Overspeeding log checks
    current_time = time.time()
    speed_limit = ui_state.get("global_speed_limit", 60.0)
    
    for objectID, speed in lane.speeds.items():
        if speed > speed_limit:
            # Highlight speeding vehicles in RED
            for p in predictions:
                if p["bbox"] == lane.bboxes.get(objectID):
                    p["is_speeding"] = True
            
            if objectID not in lane.ocr_cache:
                if current_time - lane.last_ocr_time >= 2.0:
                    lane.last_ocr_time = current_time
                    bbox = lane.bboxes[objectID]
                    plate_text = await run_ocr_tesseract(frame, bbox)
                    lane.ocr_cache[objectID] = plate_text
                    
                    # Get actual class from predictions
                    v_class = "Vehicle"
                    for p in predictions:
                        if p["bbox"] == bbox:
                            v_class = p["class"]
                            break
                    
                    lane.add_infraction(objectID, plate_text, speed, v_class)
                    
    return draw_lane_info(frame, lane_idx, predictions)

async def generate_video_feed():
    last_frames = [None] * NUM_LANES
        
    while True:
        traffic_controller.police_detected = False
        frames_to_process = []
        
        for i in range(NUM_LANES):
            lane = traffic_controller.lanes[i]
            
            if lane.light_state == "RED" and last_frames[i] is not None:
                current_frame = last_frames[i]
            else:
                ret, current_frame = global_caps[i].read()
                if not ret:
                    global_caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, current_frame = global_caps[i].read()
                last_frames[i] = current_frame
            
            # Get video timestamp in seconds for accurate speed calculation
            video_ts = global_caps[i].get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames_to_process.append((current_frame, video_ts))
            
        processed_frames = await asyncio.gather(*(process_lane(frames_to_process[i][0], i, video_time_sec=frames_to_process[i][1]) for i in range(NUM_LANES)))
        
        traffic_controller.set_police_override_setting(ui_state["police_override"])
        traffic_controller.update_signals()
        
        top_row = cv2.hconcat([processed_frames[0], processed_frames[1]])
        bottom_row = cv2.hconcat([processed_frames[2], processed_frames[3]])
        grid = cv2.vconcat([top_row, bottom_row])
        grid = cv2.resize(grid, (960, 540)) 
        
        success, buffer = cv2.imencode('.jpg', grid, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if success:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        await asyncio.sleep(0.01)

async def generate_single_lane_feed(lane_idx):
    last_frame = None
    
    while True:
        traffic_controller.police_detected = False
        lane = traffic_controller.lanes[lane_idx]
        
        if lane.light_state == "RED" and last_frame is not None:
            current_frame = last_frame
        else:
            ret, current_frame = global_caps[lane_idx].read()
            if not ret:
                global_caps[lane_idx].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, current_frame = global_caps[lane_idx].read()
            last_frame = current_frame
        
        # Get video timestamp in seconds for accurate speed calculation
        video_ts = global_caps[lane_idx].get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        processed = await process_lane(current_frame, lane_idx, video_time_sec=video_ts)
        
        traffic_controller.set_police_override_setting(ui_state["police_override"])
        traffic_controller.update_signals()
        
        processed = cv2.resize(processed, (960, 540))
        success, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/{lane_idx}")
async def single_lane_video_feed(lane_idx: int):
    if 0 <= lane_idx < NUM_LANES:
        return StreamingResponse(generate_single_lane_feed(lane_idx), media_type="multipart/x-mixed-replace; boundary=frame")
    return JSONResponse(status_code=400, content={"message": "Invalid lane index"})

@app.get("/api/capture_frame/{lane_idx}")
async def capture_frame(lane_idx: int):
    """Capture the first frame of a lane's video for calibration."""
    if lane_idx < 0 or lane_idx >= NUM_LANES:
        return JSONResponse(status_code=400, content={"message": "Invalid lane index"})
    
    src = VIDEO_SOURCES[lane_idx] if lane_idx < len(VIDEO_SOURCES) else VIDEO_SOURCES[0]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"message": "Cannot open video source"})
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return JSONResponse(status_code=500, content={"message": "Cannot read frame"})
    
    frame = cv2.resize(frame, (960, 540))
    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return JSONResponse(status_code=500, content={"message": "Cannot encode frame"})
    
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return {"image": img_b64, "width": 960, "height": 540}

@app.post("/api/toggle_police")
async def toggle_police(request: Request):
    body = await request.json()
    ui_state["police_override"] = body.get("enabled", False)
    return {"status": "success", "police_override": ui_state["police_override"]}

@app.post("/api/set_speed_limit")
async def set_speed_limit(request: Request):
    body = await request.json()
    new_limit = body.get("limit", 60.0)
    ui_state["global_speed_limit"] = float(new_limit)
    return {"status": "success", "global_speed_limit": ui_state["global_speed_limit"]}

@app.post("/api/set_display_mode")
async def set_display_mode(request: Request):
    body = await request.json()
    mode = body.get("mode", "both")
    if mode in ("speed", "bbox", "both"):
        ui_state["display_mode"] = mode
    return {"status": "success", "display_mode": ui_state["display_mode"]}

@app.post("/api/set_mode")
async def set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode", "general")
    if mode in ("general", "testing"):
        ui_state["mode"] = mode
    return {"status": "success", "mode": ui_state["mode"]}

@app.post("/api/calibrate_lane")
async def calibrate_lane(req: CalibrationRequest):
    if 0 <= req.lane_idx < NUM_LANES:
        # Scale Y coordinates from calibration canvas (960x540) to processing resolution (640x480)
        scale_y = 480.0 / 540.0
        p1_scaled = int(req.p1 * scale_y)
        p2_scaled = int(req.p2 * scale_y)
        
        line_a = min(p1_scaled, p2_scaled)
        line_b = max(p1_scaled, p2_scaled)
        
        # Update the trap line positions
        LANE_LINES[req.lane_idx] = (line_a, line_b)
        
        # Update speed calculator with new line positions AND new PPM
        sc = speed_calcs[req.lane_idx]
        sc.line_a_y = line_a
        sc.line_b_y = line_b
        sc.set_calibration(p1_scaled, p2_scaled, req.real_m)
        
        # Reset tracking state so vehicles are re-evaluated with new lines
        sc.object_crossing = {}
        sc.object_speeds = {}
        
        print(f"CALIBRATION Lane {req.lane_idx}: Lines moved to Y={line_a}, Y={line_b} | PPM={sc.pixels_per_meter:.2f}")
        return {"status": "success", "lane": req.lane_idx, "line_a": line_a, "line_b": line_b}
    return JSONResponse(status_code=400, content={"message": "Invalid lane index"})

@app.get("/api/state")
async def get_state():
    data = {
        "police_override": ui_state["police_override"],
        "global_speed_limit": ui_state["global_speed_limit"],
        "display_mode": ui_state["display_mode"],
        "mode": ui_state["mode"],
        "police_detected": traffic_controller.police_detected,
        "active_lane_idx": traffic_controller.active_lane_idx,
        "phase": traffic_controller.state,
        "lanes": []
    }
    for lane in traffic_controller.lanes:
        data["lanes"].append({
            "id": lane.lane_id,
            "pcu": lane.pcu_density,
            "light": lane.light_state,
            "infractions": lane.infractions,
            "ambulance_detected": lane.ambulance_detected,
            "accident_detected": lane.accident_detected
        })
    return data


# ===== TESTING MODE =====
# Dedicated instances for testing (separate from traffic management)
test_state_data = {
    "video_path": None,
    "speed_limit": 60.0,
    "display_mode": "both",  # "speed", "bbox", "both"
    "line_a": 200,
    "line_b": 350,
    "infractions": [],
    "running": False,
}
test_engine = InferenceEngine(skip_frames=3)
test_tracker = ByteTrackWrapper(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=20)
test_speed_calc = SpeedCalculator(line_a_y=200, line_b_y=350, sequence_pixels=150, lane_id=99)
test_cap = None  # Will hold the VideoCapture for the test video
test_objects = {}
test_bboxes = {}
test_speeds = {}

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse(request=request, name="test.html")

from fastapi import UploadFile, File

@app.post("/api/test/upload")
async def upload_test_video(file: UploadFile = File(...)):
    global test_cap
    # Save uploaded video
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, "test_upload.mp4")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    test_state_data["video_path"] = file_path
    test_state_data["infractions"] = []
    test_state_data["running"] = True
    
    # Reset test instances
    global test_tracker, test_speed_calc, test_objects, test_bboxes, test_speeds
    test_tracker = ByteTrackWrapper(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=20)
    test_speed_calc = SpeedCalculator(line_a_y=test_state_data["line_a"], line_b_y=test_state_data["line_b"], 
                                       sequence_pixels=abs(test_state_data["line_a"] - test_state_data["line_b"]), lane_id=99)
    test_objects = {}
    test_bboxes = {}
    test_speeds = {}
    test_engine.frame_count = 0
    test_engine.last_predictions = []
    
    # Open new capture
    if test_cap is not None:
        test_cap.release()
    test_cap = cv2.VideoCapture(file_path)
    
    return {"status": "success", "filename": file.filename}

async def generate_test_feed():
    global test_cap, test_objects, test_bboxes, test_speeds
    
    if test_cap is None or not test_cap.isOpened():
        return
    
    while test_state_data["running"]:
        ret, frame = test_cap.read()
        if not ret:
            test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = test_cap.read()
            if not ret:
                break
        
        video_ts = test_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame = cv2.resize(frame, (640, 480))
        
        # 1. Detection
        out = await test_engine.process_frame(frame, run_all_checks=False)
        predictions = out["traffic"]
        
        rects = [p["bbox"] for p in predictions]
        
        # 2. Tracking
        test_objects, test_bboxes = test_tracker.update(rects)
        
        # 3. Speed calculation
        test_speeds = test_speed_calc.update(test_objects, video_time_sec=video_ts)
        
        # 4. Check overspeeding
        speed_limit = test_state_data["speed_limit"]
        for objectID, speed in test_speeds.items():
            if speed > speed_limit:
                for p in predictions:
                    if p["bbox"] == test_bboxes.get(objectID):
                        p["is_speeding"] = True
                
                # Log infraction
                exists = any(inf['id'] == objectID for inf in test_state_data["infractions"])
                if not exists:
                    test_state_data["infractions"].insert(0, {
                        "id": objectID,
                        "speed": f"{speed:.1f}",
                        "type": "Vehicle",
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    test_state_data["infractions"] = test_state_data["infractions"][:30]
        
        # 5. Draw overlays
        display_mode = test_state_data["display_mode"]
        line_a = test_state_data["line_a"]
        line_b = test_state_data["line_b"]
        
        # Draw trap lines
        cv2.line(frame, (0, line_a), (frame.shape[1], line_a), (0, 255, 0), 2)
        cv2.line(frame, (0, line_b), (frame.shape[1], line_b), (0, 0, 255), 2)
        cv2.putText(frame, "LINE A", (5, line_a - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "LINE B", (5, line_b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Speed limit badge
        cv2.putText(frame, f"LIMIT: {speed_limit:.0f} km/h", (480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "TESTING MODE", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        show_bbox = display_mode in ("bbox", "both")
        show_speed = display_mode in ("speed", "both")
        
        if show_bbox:
            for pred in predictions:
                startX, startY, endX, endY = pred["bbox"]
                is_speeding = pred.get("is_speeding", False)
                color = (0, 0, 255) if is_speeding else (255, 0, 0)
                label = f"{pred['class']} ({pred['confidence']:.2f})"
                if is_speeding:
                    label = f"OVERSPEED: {label}"
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 4 if is_speeding else 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (startX, startY - 20), (startX + w, startY), color, -1)
                cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if show_speed:
            for objectID, speed in test_speeds.items():
                if objectID in test_bboxes:
                    bbox = test_bboxes[objectID]
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = bbox[3] + 18
                    speed_text = f"{speed:.0f} km/h"
                    speed_color = (0, 0, 255) if speed > speed_limit else (0, 255, 0)
                    (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (cx - 35, cy - th - 5), (cx - 35 + tw + 10, cy + 5), (0, 0, 0), -1)
                    cv2.rectangle(frame, (cx - 35, cy - th - 5), (cx - 35 + tw + 10, cy + 5), speed_color, 2)
                    cv2.putText(frame, speed_text, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        
        # Encode and yield
        frame = cv2.resize(frame, (960, 540))
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0.01)

@app.get("/test_feed")
async def test_feed():
    return StreamingResponse(generate_test_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/test/capture_frame")
async def test_capture_frame():
    if test_cap is None or not test_cap.isOpened():
        return JSONResponse(status_code=400, content={"message": "No test video loaded"})
    
    current_pos = test_cap.get(cv2.CAP_PROP_POS_FRAMES)
    test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = test_cap.read()
    test_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    if not ret:
        return JSONResponse(status_code=500, content={"message": "Cannot read frame"})
    
    frame = cv2.resize(frame, (960, 540))
    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return {"image": img_b64, "width": 960, "height": 540}

@app.post("/api/test/calibrate")
async def test_calibrate(request: Request):
    global test_speed_calc
    body = await request.json()
    p1 = body.get("p1", 200)
    p2 = body.get("p2", 350)
    real_m = body.get("real_m", 3.0)
    
    scale_y = 480.0 / 540.0
    p1_scaled = int(p1 * scale_y)
    p2_scaled = int(p2 * scale_y)
    
    line_a = min(p1_scaled, p2_scaled)
    line_b = max(p1_scaled, p2_scaled)
    
    test_state_data["line_a"] = line_a
    test_state_data["line_b"] = line_b
    
    test_speed_calc.line_a_y = line_a
    test_speed_calc.line_b_y = line_b
    test_speed_calc.set_calibration(p1_scaled, p2_scaled, real_m)
    test_speed_calc.object_crossing = {}
    test_speed_calc.object_speeds = {}
    
    return {"status": "success", "line_a": line_a, "line_b": line_b}

@app.post("/api/test/settings")
async def test_settings(request: Request):
    body = await request.json()
    if "speed_limit" in body:
        test_state_data["speed_limit"] = float(body["speed_limit"])
    if "display_mode" in body:
        if body["display_mode"] in ("speed", "bbox", "both"):
            test_state_data["display_mode"] = body["display_mode"]
    return {"status": "success", **test_state_data}

@app.get("/api/test/state")
async def test_get_state():
    return {
        "speed_limit": test_state_data["speed_limit"],
        "display_mode": test_state_data["display_mode"],
        "line_a": test_state_data["line_a"],
        "line_b": test_state_data["line_b"],
        "running": test_state_data["running"],
        "has_video": test_state_data["video_path"] is not None,
        "infractions": test_state_data["infractions"],
        "tracked_count": len(test_objects),
        "speed_count": len(test_speeds),
    }


# --- Static File Mounting (Defined AFTER routes to avoid overlap) ---

# Mount general static folder (for CSS, evidence images etc.)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SMART CITY TRAFFIC DASHBOARD IS STARTING")
    print("👉 Dashboard: http://localhost:8000")
    print("🧪 Testing:   http://localhost:8000/test")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

