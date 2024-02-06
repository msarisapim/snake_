import torch
from ultralytics import YOLO

def train_hand(data_path, Project, experiment):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model='yolov8n.pt')  # load a pretrained model
    model.to(device)

    # Train the model
    results = model.train(data=data_path, epochs=20, imgsz=640, project = Project, name = experiment)

if __name__ == '__main__':
    # save path
    Project = "runs/detect_hand"
    experiment = "handtrack"

    # data path
    data= 'data.yaml'

    # train the model
    train_hand(data, Project, experiment)