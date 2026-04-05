import cv2

class CameraState:
    def __init__(self):
        self.cap = None
        self.is_running = False

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
        return self.is_running

    def stop_camera(self):
        if self.is_running and self.cap:
            self.cap.release()
            self.is_running = False
        return not self.is_running

    def get_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

camera_state = CameraState()
