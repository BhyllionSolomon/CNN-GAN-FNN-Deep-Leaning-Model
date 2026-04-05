import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# -------------------------------
# 1. Load CNN Model (ResNet18 Directly)
# -------------------------------
def load_cnn_model(cnn_path, num_classes=2):  # 2 classes: Ripe & Rotten
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    model.eval()
    return model

# -------------------------------
# 2. Define FNN Model (Grip Force)
# -------------------------------
class FNNModel(nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()
        self.fc1 = nn.Linear(1330, 512)  # matches saved model
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)     # grip force components
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------
# 3. Tomato Camera Class
# -------------------------------
class TomatoCamera:
    def __init__(self):
        self.cnn_path = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\TomatoClass_Split\tomato_cnn.pth"
        self.fnn_path = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\Combined_Features\fnn_tomato_model.pth"

        self.cnn_model = load_cnn_model(self.cnn_path, num_classes=2)
        self.fnn_model = FNNModel()
        self.fnn_model.load_state_dict(torch.load(self.fnn_path, map_location="cpu"))
        self.fnn_model.eval()

        self.classes = ["Ripe", "Rotten"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # -------------------------------
    # Simple tomato presence check
    # -------------------------------
    def tomato_present(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        red_pixels = cv2.countNonZero(mask)
        return red_pixels > 500  # threshold: adjust if needed

    # -------------------------------
    # Feature extraction & prediction
    # -------------------------------
    def extract_features(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            cnn_output = self.cnn_model(img_t)
            _, predicted_class = torch.max(cnn_output, 1)
            predicted_class_name = self.classes[predicted_class.item()]

            feature_extractor = nn.Sequential(*list(self.cnn_model.children())[:-1])
            cnn_features = feature_extractor(img_t)
            cnn_features = torch.flatten(cnn_features, 1)

            # Placeholder for extra features (replace with real ones later)
            extra_features = torch.zeros((1, 818))
            combined_features = torch.cat([cnn_features, extra_features], dim=1)

            fnn_output = self.fnn_model(combined_features)
            grip_force = torch.softmax(fnn_output, dim=1)
            predicted_force = grip_force[0].numpy()

        return predicted_class_name, predicted_force

    # -------------------------------
    # Run camera
    # -------------------------------
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only predict if tomato is present
            if self.tomato_present(frame):
                predicted_class, predicted_force = self.extract_features(frame)
            else:
                predicted_class, predicted_force = "No Tomato", [0.0, 0.0]

            # Display results
            cv2.putText(frame, f"Tomato: {predicted_class}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.putText(frame, f"Grip Force: [{predicted_force[0]:.2f}, {predicted_force[1]:.2f}]",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

            cv2.imshow("Tomato Classification & Grip Force", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# -------------------------------
# 4. Run App
# -------------------------------
if __name__ == "__main__":
    app = TomatoCamera()
    app.run_camera()
