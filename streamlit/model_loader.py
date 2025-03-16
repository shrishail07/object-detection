from ultralytics import YOLO

def load_model():
    """Load the YOLOE model."""
    model = YOLO("yoloe-v8l-seg.pt").cuda()  # Use GPU if available
    return model
