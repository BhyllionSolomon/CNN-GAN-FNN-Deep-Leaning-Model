import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class TomatoCNN(nn.Module):
    def __init__(self):
        super(TomatoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform pipeline
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# Model globals
model = None
classes = ["Unripe", "Ripe"]


def load_model():
    """Load the trained CNN model once"""
    global model
    if model is None:
        model_path = "C:/Users/Solomon/Documents/CNN_GANS_LSTM/TomatoClass_Split/tomato_cnn.pth"
        model = TomatoCNN().to(device)
        if os.path.exists(model_path):
            print(f"✅ Loading trained TomatoCNN model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            raise FileNotFoundError(f"❌ Model file not found at {model_path}")


def detect_and_classify(frame):
    """Detect only tomatoes and classify them"""
    load_model()  # ensure model is ready

    detections = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ✅ Stricter HSV filters for tomato colors
    lower_red1, upper_red1 = np.array([0, 120, 120]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 120]), np.array([180, 255, 255])
    lower_green, upper_green = np.array([35, 80, 80]), np.array([85, 255, 255])

    # Masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_green)

    # Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:  # ✅ ignore small noise
                continue
            x, y, w, h = cv2.boundingRect(c)
            tomato_crop = frame[y:y+h, x:x+w]

            if tomato_crop.size > 0:
                img_rgb = cv2.cvtColor(tomato_crop, cv2.COLOR_BGR2RGB)
                pil_img = transform(img_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(pil_img)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred_class = torch.max(probs, 1)
                label = f"{classes[pred_class.item()]} ({conf.item():.2f})"
                detections.append(label)

                # ✅ Draw bounding box immediately
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        detections.append("No tomato detected")

    return frame, detections


# cd C:\GripForce\Tomato_GripForceEstimator\backend
# & .venv\Scripts\Activate.ps1    
#  Get-ExecutionPolicy 
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\.venv\Scripts\Activate.ps1
# uvicorn main:app --reload --host 127.0.0.1 --port 8000