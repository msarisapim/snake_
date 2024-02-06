import cv2
from ultralytics import YOLO
import pygame
import math
import random
import threading
import queue

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

def handtrack():

    model = YOLO('runs/detect_hand/handtrack/weights/best.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            yield None  # Yield None if frame not read correctly

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        results = model.predict(frame)
        conf = results[0].boxes.conf.cpu().numpy()

        bbox_array = results[0].boxes.xywh.cpu().numpy() #center
        center = None
        if len(bbox_array) > 0:
            # If detections are present, extract the center of the first bounding box
            center = bbox_array[0][:2]
            print(f"center: {center}")
            # Draw a yellow point (circle) at the center
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, YELLOW, -1)  # Yellow color (0, 255, 255)

        else:
            print("No detections")

        if len(conf) > 0 and conf[0] > 0.3:
            frame_with_boxes = draw_boxes(frame, results[0].boxes)
            cv2.imshow('Webcam', frame_with_boxes)
        else:
            cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        yield center

    cap.release()
    cv2.destroyAllWindows()

class Snake:
    def __init__(self):
        self.position = [WIDTH // 2, HEIGHT // 2]
        self.body = [[WIDTH // 2, HEIGHT // 2], [WIDTH // 2 - 10, HEIGHT // 2], [WIDTH // 2 - 20, HEIGHT // 2]]
        self.size = 10
        self.grow = False
        self.direction = (0, 1)  # Initial direction, e.g., moving upwards

    def follow_hand(self, hand_pos):
        x_diff = hand_pos[0] - self.position[0]
        y_diff = hand_pos[1] - self.position[1]
        angle = math.atan2(y_diff, x_diff)

        speed = 4  # Lower number for slower speed

        self.position[0] += math.cos(angle) * speed
        self.position[1] += math.sin(angle) * speed

        self.body.insert(0, list(self.position))

        if not self.grow:
            self.body.pop()
        self.grow = False

    def update(self):
        self.position[0] += self.direction[0] * self.size
        self.position[1] += self.direction[1] * self.size

        self.body.insert(0, list(self.position))

        if not self.grow:
            self.body.pop()
        self.grow = False

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], self.size, self.size))

    def eat(self, food_pos):
        distance = math.sqrt((self.position[0] - food_pos[0])**2 + (self.position[1] - food_pos[1])**2)
        if distance < 10:  # A threshold distance for eating the food
            self.grow = True
            return True
        return False

    def check_collision(self):
        # Check collision with boundaries
        if self.position[0] < 0 or self.position[0] > WIDTH - self.size or \
                self.position[1] < 0 or self.position[1] > HEIGHT - self.size:
            return True

        # # Check collision with itself
        # for segment in self.body[1:]:
        #     if segment == self.position:
        #         return True

        # Collision with the cursor is ignored
        return False

#food
def get_random_food_position():
    # Define a margin to keep food away from the edges
    margin = 40

    # Generate random positions, ensuring they are not too close to the frame or corner
    x = random.randint(margin, WIDTH - margin)
    y = random.randint(margin, HEIGHT - margin)

    # Make sure the positions are multiples of 10 for grid alignment
    x -= x % 10
    y -= y % 10

    return [x, y]
def draw_text(surface, text, size, x, y, color):
    font = pygame.font.SysFont(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)

def get_camera_resolution():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read from webcam")
    height, width = frame.shape[:2]
    cap.release()
    return width, height

def main_snake_hand(WIDTH, HEIGHT):
    hand_positions = queue.Queue()

    # Initialize Pygame and create the screen object
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Set window size to screen resolution
    pygame.display.set_caption("Snake Game")

    def hand_tracking_thread():
        for hand_pos in handtrack():
            hand_positions.put(hand_pos)

    threading.Thread(target=hand_tracking_thread, daemon=True).start()

    clock = pygame.time.Clock()
    snake = Snake()
    food_pos = get_random_food_position()  # Pass screen dimensions
    font_color = BLACK

    # Define the position and dimensions of the black outline
    outline_margin = 35
    outline_x = outline_margin
    outline_y = outline_margin
    outline_width = WIDTH - 2 * outline_margin
    outline_height = HEIGHT - 2 * outline_margin

    running = True  # Initialize the running variable

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not hand_positions.empty():
            hand_pos = hand_positions.get()
            if hand_pos is not None:
                snake.follow_hand(hand_pos)
                print(f"hand pos:{hand_pos}")

        if snake.eat(food_pos):
            food_pos = get_random_food_position()  # Pass screen dimensions

        screen.fill(WHITE)
        snake.draw(screen)
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

        # Draw the black outline
        pygame.draw.rect(screen, BLACK, pygame.Rect(outline_x, outline_y, outline_width, outline_height), 2)

        # Display the snake's length on the screen
        snake_length = len(snake.body)
        draw_text(screen, f'Score: {snake_length - 3}', 36, WIDTH // 2, 10, font_color)

        # Check for game over conditions
        if (
            snake.check_collision() or
            snake.position[0] <= outline_x or
            snake.position[0] >= outline_x + outline_width or
            snake.position[1] <= outline_y or
            snake.position[1] >= outline_y + outline_height
        ):
            draw_text(screen, 'Game Over', 48, WIDTH // 2, HEIGHT // 2, font_color)
            pygame.display.flip()
            pygame.time.delay(2000)
            running = False  # Set running to False to exit the game loop

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    # test_camera()  # Test camera before starting the game
    WIDTH, HEIGHT = get_camera_resolution()
    main_snake_hand(WIDTH, HEIGHT)

