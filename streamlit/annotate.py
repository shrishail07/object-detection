import supervision as sv

def annotate_image(image, detections):
    """Annotate an image with bounding boxes and labels."""
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    return annotated_image
