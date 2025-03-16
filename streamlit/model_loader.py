from ultralytics import YOLOE

def load_model():
    """Load the YOLOE model."""
    model = YOLOE("yoloe-v8l-seg.pt").cuda()  # Use GPU if available
    return model
