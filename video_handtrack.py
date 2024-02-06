from ultralytics import YOLO
import cv2

def draw_boxes(image, boxes):
    # Convert BGR to RGB for displaying in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Iterate over the bounding boxes and draw them
    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box[:4].astype(int)
        if len(box) > 4:
            conf = box[4]
            label = f'Conf: {conf:.2f}'
        else:
            conf = boxes[i].conf.cpu().numpy()
            label = f'hand: {conf[0]:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/detect_hand/vid001_to_vid008/weights/best.pt')  # pretrained YOLOv8n model

    # Run inference on the source (video)
    source = 'dataset/pim/vid010.mp4'
    model.predict(source, show=True)#, save=True, project = 'runs/results', name = 'video')


    # # Run batched inference on a list of images
    # results = model(['path/to/image.jpg'])  # return a list of Results objects
    #
    # # Process results list
    # for result in results:
    #     image_path = result.path  # Get the path of the original image
    #     print(image_path)
    #
    #     boxes = result.boxes  # Get the Boxes object
    #     # print(boxes)
    #     # Read the image in BGR format
    #     image = cv2.imread(image_path)
    #     image = draw_boxes(image, boxes)
    #
    #     # Display the image using matplotlib
    #     # Convert BGR to RGB for displaying in matplotlib
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     plt.imshow(image_rgb)
    #     plt.title("Image with Bounding Boxes")
    #     plt.axis('off')  # Turn off axis numbers and labels
    #     plt.show()
    #
    #     # Save the image instead of displaying it
    #     output_path = image_path.replace('.jpg', '_bbox_results.jpg')  # Modify as needed
    #     cv2.imwrite(output_path, image)


