from ultralytics import YOLO

# Load a model
model = YOLO('model\yolov8m.pt')  # build a new model from YAML
result = model.track(source=0, show=True)
# result = model(source=0, show=True)