import pygame
import numpy as np
import random

# Initialize flags
CAN_SLIDE_INTO_GAP = True
SEE_NEAREST_GAP = True

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont("Arial", 24)

# Screen size
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning - Jump and Slide")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Ground properties
ground_height = 50
gap_width = 100
min_gap_distance = 300
max_gap_distance = 500

# Agent properties
agent_width, agent_height = 50, 50
agent_x, agent_y = 50, HEIGHT - ground_height - agent_height
velocity_y = 0
gravity = 0.5
jump_force = -10
is_jumping = False
is_sliding = False
slide_duration = 20
slide_counter = 0

# Game parameters
fps = 60
clock = pygame.time.Clock()

# Q-learning parameters
state_size = (10, 10, 10)  # discretize the state space
action_size = 3  # jump, slide, do nothing
q_table = np.zeros(state_size + (action_size,))

# Hyperparameters
alpha = 0.1  # learning rate
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration-exploitation tradeoff
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# Randomly generate a new gap ahead
def generate_gap(current_position):
    gap_start = random.randint(current_position + min_gap_distance, current_position + max_gap_distance)
    return gap_start

# Define reward structure
def get_reward(agent_x, agent_y, gap_start, gap_width):
    # Reward for clearing the gap
    if agent_x > gap_start + gap_width:
        return 100  # Success
    # Penalty for falling into the gap
    elif agent_x >= gap_start and agent_x <= gap_start + gap_width and agent_y >= HEIGHT - ground_height - agent_height:
        return -100  # Fell in the gap
    return -1  # Neutral step

# Discretize the agent's position into states
def get_state(agent_x, agent_y):
    state_x = min(int(agent_x / (WIDTH / state_size[0])), state_size[0] - 1)
    state_y = min(int(agent_y / (HEIGHT / state_size[1])), state_size[1] - 1)
    state_gap = min(int((500 + gap_start - agent_x) / 100), state_size[2] - 1)
    if SEE_NEAREST_GAP:
        return state_x, state_y, state_gap
    return state_x, state_y
    
stop_criteria = False

# Main training loop
for episode in range(episodes):
    agent_x, agent_y = 50, HEIGHT - ground_height - agent_height
    velocity_y = 0
    is_jumping = False
    is_sliding = False
    slide_counter = 0
    gap_start = generate_gap(agent_x)
    episode_distance = 0
    
    if stop_criteria:
        break
    
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_criteria = True
                break
        # Select action (exploration vs exploitation)
        state = get_state(agent_x, agent_y)
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2])  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Perform action
        if action == 1 and not is_jumping and not is_sliding:  # Jump
            is_jumping = True
            velocity_y = jump_force
        elif action == 2 and not is_jumping and not is_sliding:  # Slide
            is_sliding = True
            slide_counter = slide_duration
            agent_height = 48  # Shrink the agent's height for sliding
        elif action == 0:  # Do nothing
            pass

        # Apply physics (jumping/falling)
        if is_jumping:
            velocity_y += gravity
            agent_y += velocity_y
            if agent_y >= HEIGHT - ground_height - agent_height:
                agent_y = HEIGHT - ground_height - agent_height
                is_jumping = False
        
        # Handle sliding action
        if is_sliding:
            slide_counter -= 1
            if slide_counter <= 0:
                is_sliding = False
                agent_height = 50  # Restore agent's height after sliding
            if CAN_SLIDE_INTO_GAP and agent_x >= gap_start and agent_x <= gap_start + gap_width and agent_y >= HEIGHT - ground_height - agent_height: # Slided into the gap
                agent_y += 5
        
        # Update agent's position
        episode_distance += 5  # Calulate total distance moved
        if(agent_x + 5 > WIDTH):
            gap_start = generate_gap(agent_x)
        agent_x = (agent_x + 5) % WIDTH  # Move forward

        # Check if episode is done
        reward = get_reward(agent_x, agent_y, gap_start, gap_width)
        if reward == 100 or reward == -100:
            done = True
        
        # Update Q-table
        next_state = get_state(agent_x, agent_y)
        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        # Render the game
        screen.fill(WHITE)
        episode_text = font.render(f"Episode: {episode}", True, BLACK)
        distance_text = font.render(f"Distance moved: {episode_distance}", True, BLACK)
        screen.blit(episode_text,(50, 50))
        screen.blit(distance_text,(50, 75))        
        pygame.draw.rect(screen, BLACK, (0, HEIGHT - ground_height, WIDTH, ground_height))
        pygame.draw.rect(screen, WHITE, (gap_start, HEIGHT - ground_height, gap_width, ground_height))
        pygame.draw.rect(screen, RED, (agent_x, agent_y, agent_width, agent_height))
        pygame.display.update()

        clock.tick(fps)
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

pygame.quit()
