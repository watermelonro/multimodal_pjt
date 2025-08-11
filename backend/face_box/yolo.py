from ultralytics import YOLO
# loading YOLO Model

def load_model():
    model = YOLO("models/best.pt")
    return model