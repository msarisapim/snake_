import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
def train_hand():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model='yolov8n.pt')  # load a pretrained model
    model.to(device)

    # Train the model
    results = model.train(data='data.yaml', epochs=20, imgsz=640, project = Project, name = experiment)

def draw_boxes(image_path, boxes):
    # Read the image in BGR format
    image = cv2.imread(image_path)

    # Convert BGR to RGB for displaying in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Iterate over the bounding boxes and draw them
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4].astype(int)
        if len(box) > 4:
            conf = box[4]
            label = f'Conf: {conf:.2f}'
        else:
            label = 'Object'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Convert BGR to RGB for displaying in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.title("Image with Bounding Boxes")
    plt.axis('off')  # Turn off axis numbers and labels
    plt.show()

    # # Save the image instead of displaying it
    # output_path = image_path.replace('.jpg', '_bbox_results.jpg')  # Modify as needed
    # cv2.imwrite(output_path, image)

if __name__ == '__main__':
    # save path
    Project = "runs/detect_hand"
    experiment = "train"

    # train the model
    # train_hand()

    # Load a model
    model = YOLO('runs/detect/train23/weights/best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(['dataset/test/images/00000019_png.rf.a3371106b5d24dfc2216dc3168b7df39.jpg',
                     'dataset/test/images/00000127_png.rf.42c693f7f2e3fbf6d43a81ca585697f1.jpg',
                     'dataset/me/pim_001.jpg',
                     'dataset/me/pim_004.jpg'])  # return a list of Results objects

    # Process results list
    for result in results:
        image_path = result.path  # Get the path of the original image
        print(image_path)


        boxes = result.boxes  # Get the Boxes object
        # print(boxes)
        draw_boxes(image_path, boxes)



