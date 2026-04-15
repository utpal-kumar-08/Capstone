import asyncio
import httpx
import base64
import os
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
VERSION = os.getenv("VERSION")

print(f"Testing with MODEL_ID: {MODEL_ID}, VERSION: {VERSION}")

async def test_inference():
    # Just a blank image for testing API connectivity
    import numpy as np
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    success, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    url = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"
    print(f"URL: {url}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_inference())
