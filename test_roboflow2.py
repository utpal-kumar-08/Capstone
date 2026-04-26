import httpx
import base64
import asyncio
import cv2
import numpy as np

async def test_api():
    api_key = "FfAa3OyIQ324aDKFWnPu"
    
    url = f"https://detect.roboflow.com/traffic-accident-detection-whv7l/1?api_key={api_key}"
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, data=img_str, headers={"Content-Type": "application/x-www-form-urlencoded"})
            print(f"URL: {url}")
            print(f"Status: {res.status_code}")
            print(f"Response: {res.text}\n")
        except Exception as e:
            print(e)
                
asyncio.run(test_api())
