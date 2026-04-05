import cv2
from detector_pipeline import detect_and_classify

cap = cv2.VideoCapture(0)  # open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tomato detection + classification
    frame, detections = detect_and_classify(frame)

    # Show video with bounding boxes
    cv2.imshow("Tomato Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
