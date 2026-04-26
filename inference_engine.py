import cv2
import httpx
import base64
import os
import asyncio
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the Haar cascade for license plates (comes built-in with OpenCV)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def _find_plate_region(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blur = cv2.bilateralFilter(equalized, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1200 or h < 15:
            continue
        aspect = w / float(h + 1e-6)
        if 2.0 <= aspect <= 7.0:
            candidates.append((area, x, y, w, h))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1:]


def _run_tesseract_sync(frame, bbox):
    startX, startY, endX, endY = bbox
    # Add small padding for context
    vehicle_crop = frame[max(0, startY-10):min(frame.shape[0], endY+10), max(0, startX-10):min(frame.shape[1], endX+10)]
    if vehicle_crop.size == 0:
        return "N/A"
    
    vehicle_gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(vehicle_gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 15))
    
    if len(plates) > 0:
        px, py, pw, ph = plates[0]
        plate_crop = vehicle_crop[py:py+ph, px:px+pw]
    else:
        contour_plate = _find_plate_region(vehicle_gray)
        if contour_plate is not None:
            x, y, w, h = contour_plate
            plate_crop = vehicle_crop[y:y+h, x:x+w]
        else:
            h, w = vehicle_crop.shape[:2]
            plate_crop = vehicle_crop[int(h*0.55):, :]

    if plate_crop.size == 0:
        return "UNREADABLE"

    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.equalizeHist(plate_gray)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    _, thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    text = pytesseract.image_to_string(resized, config=custom_config).strip()
    clean_text = "".join(ch for ch in text if ch.isalnum() or ch == '-')
    if len(clean_text) >= 3 and any(ch.isdigit() for ch in clean_text):
        return clean_text.upper()
    return "UNREADABLE"

async def run_ocr_tesseract(frame, bbox):
    return await asyncio.to_thread(_run_tesseract_sync, frame, bbox)

class InferenceEngine:
    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_predictions = []
        self.semaphore = asyncio.Semaphore(2)
        
        # Load variables inside init to ensure load_dotenv has run
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_id = os.getenv("WORKSPACE_ID")
        self.model_id = os.getenv("MODEL_ID")
        self.version = os.getenv("VERSION", "1")
        
        self.amb_model = os.getenv("AMBULANCE_MODEL_ID")
        self.amb_ver = os.getenv("AMBULANCE_VERSION", "1")
        self.pol_model = os.getenv("POLICE_MODEL_ID")
        self.pol_ver = os.getenv("POLICE_VERSION", "1")
        self.acc_model = os.getenv("ACCIDENT_MODEL_ID")
        self.acc_version = os.getenv("ACCIDENT_VERSION", "1")
        self.acc_api_key = os.getenv("ACCIDENT_API_KEY", self.api_key)
        self.acc_workspace = os.getenv("ACCIDENT_WORKSPACE_ID", self.workspace_id)
        
        # Diagnostic tracking
        self.last_status = {} # model_name -> status_code
        self.error_messages = {}
        self.disabled_models = set()

        self.last_amb_preds = []
        self.last_pol_preds = []
        self.last_acc_preds = []

        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        self.client = httpx.AsyncClient(timeout=10.0, limits=limits)
        
        # Construction of URLs with tuned parameters for maximum detection
        params = f"&confidence=20&overlap=50"
        self.api_url = f"https://detect.roboflow.com/{self.model_id}/{self.version}?api_key={self.api_key}{params}"
        self.ambulance_api_url = f"https://detect.roboflow.com/{self.amb_model}/{self.amb_ver}?api_key={self.api_key}{params}"
        self.police_api_url = f"https://detect.roboflow.com/{self.pol_model}/{self.pol_ver}?api_key={self.api_key}{params}"
        
        self.accident_api_url = f"https://detect.roboflow.com/{self.acc_model}/{self.acc_version}?api_key={self.acc_api_key}{params}"

    async def _post_inference(self, url, img_str):
        if not url or "None" in url or "//" in url.split("roboflow.com")[-1]: return None
        model_tag = url.split('/')[3]
        if model_tag in self.disabled_models:
            return None
        
        async with self.semaphore:
            try:
                response = await self.client.post(
                    url,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 200:
                    msg = f"API Error: {response.status_code} - {response.text}"
                    if response.status_code == 403:
                        self.disabled_models.add(model_tag)
                        msg = f"❌ AUTH ERROR: Model '{model_tag}' is Forbidden. Check Roboflow Key/Plan."
                    print(msg)
                    return {"predictions": [], "error": response.status_code}
            except Exception as e:
                print(f"Inference Connection Exception: {e}")
        return None

    async def process_frame(self, frame, run_all_checks=False):
        self.frame_count += 1
        
        # Skip inference on intermediate frames — return cached predictions
        if self.frame_count % self.skip_frames != 0:
            return {"traffic": self.last_predictions, "ambulance": False, "police": False, "accident": False}
        
        success, buffer = cv2.imencode('.jpg', frame)
        if not success: 
            return {"traffic": self.last_predictions, "ambulance": False, "police": False, "accident": False}
            
        img_str = base64.b64encode(buffer).decode('utf-8')
        tasks = {"traffic": self._post_inference(self.api_url, img_str)}
        
        if run_all_checks:
            if self.amb_model:
                tasks["ambulance"] = self._post_inference(self.ambulance_api_url, img_str)
            if self.pol_model:
                tasks["police"] = self._post_inference(self.police_api_url, img_str)
            if self.acc_model:
                tasks["accident"] = self._post_inference(self.accident_api_url, img_str)

        results = await asyncio.gather(*tasks.values())
        raw_responses = dict(zip(tasks.keys(), results))
        
        # Consolidate all detections into one list
        traffic_data = raw_responses.get("traffic")
        if traffic_data:
            current_traffic = self.parse_traffic_response(traffic_data)
            # Persistence: If current is empty, keep last for a short while
            # or just merge them if you want maximum stability
            if len(current_traffic) > 0:
                self.last_predictions = current_traffic
            elif self.frame_count % 5 == 0: # Cleanup every 5 frames if still nothing
                self.last_predictions = []
        
        # Persistence for specialized detections
        if run_all_checks:
            self.last_amb_preds = self.parse_traffic_response(raw_responses.get("ambulance")) if raw_responses.get("ambulance") else []
            self.last_pol_preds = self.parse_traffic_response(raw_responses.get("police")) if raw_responses.get("police") else []
            self.last_acc_preds = self.parse_traffic_response(raw_responses.get("accident")) if raw_responses.get("accident") else []
        
        # Merge all current valid predictions with basic duplicate removal
        all_detections = self.last_predictions.copy()
        
        specialized_preds = self.last_amb_preds + self.last_pol_preds + self.last_acc_preds
        for s_pred in specialized_preds:
            # Check if this box overlaps significantly with any existing box
            is_duplicate = False
            s_bbox = s_pred["bbox"]
            for t_pred in all_detections:
                t_bbox = t_pred["bbox"]
                # Simple Intersection over Union (IoU) or just center proximity
                # Here we use center proximity for speed
                scX, scY = (s_bbox[0]+s_bbox[2])/2, (s_bbox[1]+s_bbox[3])/2
                tcX, tcY = (t_bbox[0]+t_bbox[2])/2, (t_bbox[1]+t_bbox[3])/2
                dist = ((scX-tcX)**2 + (scY-tcY)**2)**0.5
                if dist < 30: # If centers are closer than 30 pixels, consider same object
                    is_duplicate = True
                    # If specialized model is "more specific", update the class
                    if s_pred["class"].lower() in ["ambulance", "police", "accident"]:
                        t_pred["class"] = s_pred["class"]
                    break
            if not is_duplicate:
                all_detections.append(s_pred)
        
        out = {
            "traffic": all_detections, 
            "ambulance": self._check_class_presence(raw_responses.get("ambulance"), "ambulance") if run_all_checks else (len(self.last_amb_preds) > 0),
            "police": self._check_class_presence(raw_responses.get("police"), "police", "person") if run_all_checks else (len(self.last_pol_preds) > 0),
            "accident": self._check_class_presence(raw_responses.get("accident"), "accident") if run_all_checks else (len(self.last_acc_preds) > 0)
        }
            
        return out

    def parse_traffic_response(self, data):
        predictions = []
        for pred in data.get("predictions", []):
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            predictions.append({
                "bbox": [int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)],
                "class": pred.get("class", "Vehicle"),
                "confidence": pred.get("confidence", 0.0)
            })
        return predictions

    def _check_class_presence(self, data, *target_classes):
        if not data: return False
        for pred in data.get("predictions", []):
            # Lowered confidence threshold to 0.25 to prevent rejecting true positives,
            # especially for the accident model which might output lower confidence bounds
            if any(tc in pred.get("class", "").lower() for tc in target_classes) and pred.get("confidence", 0) > 0.25:
                return True
        return False

    async def cleanup(self):
        await self.client.aclose()
