from ultralytics import YOLO

# Load a model
model = YOLO('model\yolov8n-pose.pt')  # build a new model from YAML
result = model(source=0, show=True)