import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Hand-Controlled Snake Game")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Snake Variables
snake_pos = [[WIDTH // 2, HEIGHT // 2]]
snake_dir = "RIGHT"
speed = 7
food_pos = [random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE, random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE]
game_running = True
paused = False
blocks_eaten = 0

# OpenCV Camera
cap = cv2.VideoCapture(0)

# Buffers for direction change confirmation
direction_buffer = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
buffer_threshold = 15

# Global variable to store last detected finger position
last_finger_pos = (0, 0)

def calculate_direction(x, y):
    global snake_dir, last_finger_pos
    direction = None
    dx = x - last_finger_pos[0]
    dy = y - last_finger_pos[1]

    if abs(dx) > 20 or abs(dy) > 20:  # Threshold to detect significant movement
        if abs(dx) > abs(dy):  # Horizontal movement
            if dx > 0 and snake_dir != "LEFT":  # Right
                direction = "RIGHT"
            elif dx < 0 and snake_dir != "RIGHT":  # Left
                direction = "LEFT"
        else:  # Vertical movement
            if dy > 0 and snake_dir != "UP":  # Down
                direction = "DOWN"
            elif dy < 0 and snake_dir != "DOWN":  # Up
                direction = "UP"
    
    last_finger_pos = (x, y)
    return direction

def detect_hand_direction(frame):
    global snake_dir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * 200), int(index_finger_tip.y * 150)
            
            # Calculate direction based on finger movement
            direction = calculate_direction(x, y)
            if direction:
                snake_dir = direction

            # Highlight detected palm
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame

def draw_direction_indicator():
    font = pygame.font.SysFont("Arial", 24)
    direction_text = font.render(f"Direction: {snake_dir}", True, WHITE)
    screen.blit(direction_text, (10, 10))

def draw_block_count():
    font = pygame.font.SysFont("Arial", 24)
    block_text = font.render(f"Blocks Eaten: {blocks_eaten}", True, WHITE)
    screen.blit(block_text, (WIDTH - 200, HEIGHT - 30))

while game_running:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = detect_hand_direction(frame)
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                game_running = False
            if event.key == pygame.K_p:
                paused = not paused
    
    if not paused:
        screen.fill(BLACK)
        
        # Move the snake
        head_x, head_y = snake_pos[0]
        if snake_dir == "UP":
            head_y -= GRID_SIZE
        elif snake_dir == "DOWN":
            head_y += GRID_SIZE
        elif snake_dir == "LEFT":
            head_x -= GRID_SIZE
        elif snake_dir == "RIGHT":
            head_x += GRID_SIZE
        
        # Wrap around logic
        if head_x < 0:
            head_x = WIDTH - GRID_SIZE
        elif head_x >= WIDTH:
            head_x = 0
        if head_y < 0:
            head_y = HEIGHT - GRID_SIZE
        elif head_y >= HEIGHT:
            head_y = 0
        
        new_head = [head_x, head_y]
        if new_head in snake_pos:
            game_running = False  # End game if snake bites itself
        
        snake_pos.insert(0, new_head)
        if new_head == food_pos:
            food_pos = [random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE, random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE]
            blocks_eaten += 1
        else:
            snake_pos.pop()
        
        # Draw snake and food (on the game screen only, not on the camera window)
        for block in snake_pos:
            pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], GRID_SIZE, GRID_SIZE))
        
        # Convert OpenCV frame to Pygame and display it as an overlay
        frame = cv2.resize(frame, (200, 150))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (WIDTH - 210, 10))  # Position small cam window
        
        # Draw direction indicator
        draw_direction_indicator()
        
        # Draw block count
        draw_block_count()

    # Pause message
    if paused:
        font = pygame.font.SysFont("Arial", 48)
        pause_text = font.render("Game Paused", True, WHITE)
        screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 50))
    
    pygame.display.update()
    clock.tick(speed)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
