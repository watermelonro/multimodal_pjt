from ultralytics import YOLO
# loading YOLO Model

def load_model():
    model = YOLO("model/yolo_best.pt")
    return model