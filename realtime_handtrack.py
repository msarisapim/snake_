import cv2
from ultralytics import YOLO

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (0, 255, 255)

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
        cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    return image

def realtime_hand():
    # Load a model
    model = YOLO('runs/detect_hand/handtrack/weights/best.pt')  # pretrained YOLOv8n model

    # Initialize the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # process
        results = model.predict(frame)
        conf = results[0].boxes.conf.cpu().numpy()

        # hand position
        bbox_array = results[0].boxes.xywh.cpu().numpy()
        if len(results[0].boxes.xywh.cpu().numpy()) > 0:
            # If detections are present, extract the top-left corner of the first bounding box
            follow_point = bbox_array[0][:2]
            print(f"point: {follow_point}")

            # Draw a yellow point (circle)
            cv2.circle(frame, (int(follow_point[0]), int(follow_point[1])), 5, YELLOW, -1)  # Yellow color (0, 255, 255)

        else:
            print("No detections")


        # Draw boxes when confidence larger than theshold
        if len(conf) > 0 and conf[0] > 0.3:
            frame_with_boxes = draw_boxes(frame, results[0].boxes)
            # Display the frame
            cv2.imshow('Webcam', frame_with_boxes)
        else:
            # Display the frame
            cv2.imshow('Webcam', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("This is Real-time hand tracking")
    realtime_hand()
