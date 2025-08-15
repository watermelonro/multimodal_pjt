from ultralytics import YOLO
import os
# loading YOLO Model

def load_model():
    module_dir = os.path.dirname(__file__)
    model_path = os.path.join(module_dir, '../../models/best.pt')
    model = YOLO(model_path)
    return model