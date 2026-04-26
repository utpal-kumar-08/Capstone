import httpx
import os
from dotenv import load_dotenv
import cv2
import base64
import numpy as np

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
acc_api_key = os.getenv("ACCIDENT_API_KEY")

dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
success, buffer = cv2.imencode('.jpg', dummy_img)
img_str = base64.b64encode(buffer).decode('utf-8')

urls = [
    f"https://detect.roboflow.com/jy516/capstone-cwifn/3?api_key={api_key}",
    f"https://detect.roboflow.com/capstone-cwifn/3?api_key={api_key}",
    f"https://detect.roboflow.com/jth00/traffic-accident-detection-whv7l/1?api_key={acc_api_key}",
    f"https://detect.roboflow.com/traffic-accident-detection-whv7l/1?api_key={acc_api_key}"
]

for url in urls:
    print(f"\nTesting POST {url.split('?')[0]} with its respective key")
    resp = httpx.post(url, headers={'Content-Type': 'application/x-www-form-urlencoded'}, data=img_str)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
