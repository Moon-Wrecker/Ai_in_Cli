import pygame
import random

# Initialize Pygame
pygame.init()

# Set window dimensions
window_width = 600
window_height = 400
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Simple Snake Game")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

# Snake initial position and size
snake_x = window_width / 2
snake_y = window_height / 2
snake_size = 10
snake_list = []
snake_length = 1

# Food initial position
food_x = round(random.randrange(0, window_width - snake_size) / 10.0) * 10.0
food_y = round(random.randrange(0, window_height - snake_size) / 10.0) * 10.0

# Game loop
game_over = False
clock = pygame.time.Clock()
snake_speed = 15

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

    # Movement
    # (Add movement logic here - this is a placeholder)

    # Draw everything
    window.fill(black)
    pygame.draw.rect(window, red, [food_x, food_y, snake_size, snake_size])
    # (Add snake drawing logic here - this is a placeholder)

    pygame.display.update()
    clock.tick(snake_speed)

pygame.quit()
quit()