import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Snake Class
class Snake:
    def __init__(self):
        # Place the snake more centrally to avoid immediate boundary collision
        self.position = [WIDTH // 2, HEIGHT // 2]
        self.body = [[WIDTH // 2, HEIGHT // 2], [WIDTH // 2 - 10, HEIGHT // 2], [WIDTH // 2 - 20, HEIGHT // 2]]
        self.size = 10
        self.grow = False

    def follow_mouse(self, mouse_pos):
        x_diff = mouse_pos[0] - self.position[0]
        y_diff = mouse_pos[1] - self.position[1]
        angle = math.atan2(y_diff, x_diff)

        # Reduce the speed of the snake
        speed = 3  # Lower number for slower speed
        self.position[0] += math.cos(angle) * speed
        self.position[1] += math.sin(angle) * speed

        self.body.insert(0, list(self.position))

        if not self.grow:
            self.body.pop()
        self.grow = False

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], self.size, self.size))

    def eat(self, food_pos):
        distance = math.sqrt((self.position[0] - food_pos[0])**2 + (self.position[1] - food_pos[1])**2)
        if distance < 15:  # A threshold distance for eating the food
            self.grow = True
            return True
        return False

    def check_collision(self):
        # Check collision with boundaries
        if self.position[0] < 0 or self.position[0] > WIDTH - self.size or \
           self.position[1] < 0 or self.position[1] > HEIGHT - self.size:
            return True

        # Check collision with itself
        for segment in self.body[1:]:
            if segment == self.position:
                return True

        # Collision with the cursor is ignored
        return False

# Food
def get_random_food_position():
    return [random.randrange(1, WIDTH // 10) * 10, random.randrange(1, HEIGHT // 10) * 10]

def draw_text(surface, text, size, x, y, color):
    font = pygame.font.SysFont(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)


def main():
    clock = pygame.time.Clock()
    snake = Snake()
    food_pos = get_random_food_position()
    running = True
    font_color = (0, 0, 0)  # Color for the text

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        mouse_pos = pygame.mouse.get_pos()
        snake.follow_mouse(mouse_pos)

        if snake.eat(food_pos):
            food_pos = get_random_food_position()

        screen.fill(WHITE)
        snake.draw(screen)
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

        # Display score
        score = len(snake.body) - 3  # Initial length is 3
        draw_text(screen, f'Score: {score}', 36, WIDTH // 2, 10, font_color)

        if snake.check_collision():
            draw_text(screen, 'Game Over', 48, WIDTH // 2, HEIGHT // 2, font_color)
            pygame.display.flip()  # Update the display with the game over message
            pygame.time.delay(2000)  # Delay to show the message
            running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()