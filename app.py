import cv2
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Loading models...")

yolo_model = YOLO('yolov8n.pt')
cnn_model = load_model(r"C:\PhD Thesis\CNN_GANS_LSTM\TomatoClass_Split\results\tomato_cnn_model.h5")
fnn_model = load_model(r"C:\PhD Thesis\CNN_GANS_LSTM\RegressionModel\FNN_Regression_Model.h5")

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_tomato(image):
    processed_image = preprocess_image(image)
    predictions = cnn_model.predict(processed_image, verbose=0)
    confidence = float(np.max(predictions))
    class_idx = np.argmax(predictions)
    
    classes = ["Unripe", "Ripe", "Overripe", "Diseased"]
    prediction = classes[class_idx] if class_idx < len(classes) else "Unknown"
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

def regress_tomato_properties(image):
    processed_image = preprocess_image(image)
    predictions = fnn_model.predict(processed_image, verbose=0)[0]
    
    return {
        "weight_g": float(predictions[0]) * 200 + 50,
        "size_cm": float(predictions[1]) * 10 + 3,
        "pressure_kpa": float(predictions[2]) * 100 + 50,
        "force_n": float(predictions[3]) * 20 + 5,
        "torque_n_m": float(predictions[4]) * 0.5 + 0.1,
        "grip_force_n": float(predictions[5]) * 15 + 2,
        "time_s": float(predictions[6]) * 10 + 1
    }

def detect_objects(image):
    results = yolo_model(image, verbose=False)
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])
        class_name = results[0].names[int(box.cls[0])]
        
        h, w = image.shape[:2]
        bbox = [
            float((x1 + x2) / 2 / w),
            float((y1 + y2) / 2 / h),
            float((x2 - x1) / w),
            float((y2 - y1) / h)
        ]
        
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })
    
    return detections

def get_best_tomato_detection(detections):
    for detection in detections:
        if detection["confidence"] > 0.5:
            return detection
    return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detections = detect_objects(image)
        best_detection = get_best_tomato_detection(detections)
        classification = classify_tomato(image)
        regression = regress_tomato_properties(image)
        
        has_tomato = best_detection is not None
        
        enhanced_classification = {
            **classification,
            "bbox": best_detection["bbox"] if best_detection else None,
            "detection_confidence": best_detection["confidence"] if best_detection else 0.0,
            "detection_label": best_detection["class"] if best_detection else "No Tomato",
            "hasTomato": has_tomato,
            "validation": "valid" if has_tomato else "invalid"
        }

        _, processed_jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_image = base64.b64encode(processed_jpeg).decode()
        
        return {
            "classification": enhanced_classification,
            "regression": regression,
            "detection_info": {
                "total_objects": len(detections),
                "all_detections": detections,
                "best_detection": best_detection
            },
            "image": processed_image,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)