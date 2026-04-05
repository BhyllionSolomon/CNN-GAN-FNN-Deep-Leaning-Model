import os
import io
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# =========================================================
# FASTAPI CONFIGURATION
# =========================================================
app = FastAPI()

# Allow frontend requests (React runs on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DEFINE FNN REGRESSION MODEL
# =========================================================
class FNNRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FNNRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, 7)  # 7 regression outputs

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# =========================================================
# LOAD MODEL
# =========================================================
base_path = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\Combined_Features"
model_path = os.path.join(base_path, "fnn_tomato_regression.pth")

input_size = 256  # Adjust based on your feature vector length
hidden_size = 512

model = FNNRegression(input_size, hidden_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

print(f"✅ FNN regression model loaded from: {model_path}")

# =========================================================
# DUMMY FEATURE EXTRACTOR (Replace with your CNN feature extractor)
# =========================================================
def extract_features_from_image(image: Image.Image):
    """
    Replace this with your real CNN feature extractor.
    For now, it returns dummy features for testing.
    """
    np.random.seed(42)
    return np.random.rand(256).astype(np.float32)  # Example feature vector of size 256

# =========================================================
# DETECTION ENDPOINT
# =========================================================
@app.post("/detect")
async def detect_tomato(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Extract CNN+GAN+other features (replace with your real extractor)
        features = extract_features_from_image(image)
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predict 7 regression outputs
        with torch.no_grad():
            preds = model(x_tensor).squeeze(0).numpy()

        # Feature names and values
        feature_names = ["weight", "torque", "pressure", "hardness", "moisture", "elasticity", "gripForce"]
        feature_values = {name: round(float(val), 4) for name, val in zip(feature_names, preds)}

        # Example ripeness decision (you can replace with your CNN)
        ripeness_status = "ripe" if feature_values["gripForce"] > 0.5 else "unripe"
        confidence = np.clip(np.mean(preds) / 10, 0, 1)  # dummy confidence

        # Return result JSON
        return {
            "status": ripeness_status,
            "features": feature_values,
            "confidence": float(confidence)
        }

    except Exception as e:
        print("❌ Error during detection:", e)
        return {"status": "none", "features": {}, "confidence": 0.0}

# =========================================================
# ROOT ENDPOINT
# =========================================================
@app.get("/")
def root():
    return {"message": "Tomato FNN Regression Backend Active ✅"}
