import httpx
import base64
import asyncio

async def test_api():
    api_key = "zN3HCltTtKg0EgEe7P2O"
    
    # Try different URLs
    urls = [
        f"https://detect.roboflow.com/indiantraffic/indiantraffic/1?api_key={api_key}",
        f"https://detect.roboflow.com/indiantraffic/1?api_key={api_key}"
    ]
    
    # Create dummy image
    import cv2
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    async with httpx.AsyncClient() as client:
        for url in urls:
            try:
                res = await client.post(url, data=img_str, headers={"Content-Type": "application/x-www-form-urlencoded"})
                print(f"URL: {url}")
                print(f"Status: {res.status_code}")
                print(f"Response: {res.text}\n")
            except Exception as e:
                print(e)
                
asyncio.run(test_api())
