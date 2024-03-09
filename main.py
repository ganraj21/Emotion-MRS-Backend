import json
import io
import base64

from typing import Optional
from fastapi import FastAPI, WebSocket

import cv2
import numpy as np
from PIL import Image
from fer import FER

app = FastAPI()
detector = FER()

def decode_base64_image(encoded_data: str) -> Optional[np.ndarray]:
    try:
        image_byt64 = encoded_data.split(',')[1]
        # decode and convert into image
        image = np.frombuffer(base64.b64decode(image_byt64), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        payload = await websocket.receive_text()
        payload = json.loads(payload)

        # Decode image
        image = decode_base64_image(payload['data']['image'])
        
        if image is not None:
            # Detect Emotion via Tensorflow model
            prediction = detector.detect_emotions(image)
            response = {
                "predictions": prediction[0]['emotions'],
                "emotion": max(prediction[0]['emotions'], key=prediction[0]['emotions'].get)
            }
            await websocket.send_json(response)
            
            # Add logging for debugging
            print("Emotion Detection Result:", response)

    except Exception as e:
        print(f"Error processing payload: {e}")

    finally:
        websocket.close()
