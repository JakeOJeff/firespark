import cv2
import mediapipe as mp
import numpy as np
import math
import random

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open primary webcam (index 0)
cap = cv2.VideoCapture(0)

# List to store particles
particles = []

# Helper function to calculate the Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Helper function to calculate the angle between two points
def calculate_angle(x1, y1, x2, y2, x3, y3):
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return abs(angle)

# Particle class to handle individual particle behavior
class Particle:
    def __init__(self, x, y, x_vel, y_vel):
        self.x = x
        self.y = y
        self.size = random.randint(10, 20)
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.color = (random.randint(0, 5), random.randint(122, 142), random.randint(250, 255))  # BGR Format

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel
        self.size -= 0.5  # Decrease size over time

    def draw(self, frame):
        if self.size > 0:
            cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), self.color, -1)

# Main loop for capturing frames and processing hands
while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror-like interaction
    frame = cv2.flip(frame, 1)

    # Convert frame from BGR to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Process hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get fingertip landmarks (thumb and index finger)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            # Get other fingertips (middle, ring, pinky)
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            mcp = hand_landmarks.landmark[5]  # MCP joint for reference

            # Convert normalized landmark coordinates to pixel coordinates
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate the distance between the thumb and index finger
            shortestDist = 0
            fingerlists = [8, 12, 16, 20]
            for i in fingerlists:
                currentLandmark = hand_landmarks.landmark[i]
                distance_thumb_index = calculate_distance(thumb_x, thumb_y, int(currentLandmark.x * w),  int(currentLandmark.y * h))
                if shortestDist == 0:
                    shortestDist = distance_thumb_index
                elif distance_thumb_index < shortestDist:
                    shortestDist = distance_thumb_index
            # Threshold to determine if the hand is clasped (adjust as necessary)
            if shortestDist > 100:  # Distance threshold
                # Emit particles from each fingertip
                fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
                fingertip_coords = [(int(fingertip.x * w), int(fingertip.y * h)) for fingertip in fingertips]

                for tip_x, tip_y in fingertip_coords:
                    # Emit particles from each fingertip
                    for _ in range(10):
                        x_vel = random.uniform(-2, 2) + (tip_x - mcp.x * w) * 0.05
                        y_vel = random.uniform(-2, 2) + (tip_y - mcp.y * h) * 0.05
                        particles.append(Particle(tip_x, tip_y, x_vel, y_vel))

            # Draw hand landmarks on the frame for visualization
           # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Move and draw particles on the frame
    new_particles = []
    for particle in particles:
        particle.move()
        particle.draw(frame)
        # Keep only particles that still have a size greater than 0
        if particle.size > 0:
            new_particles.append(particle)
    particles = new_particles

    # Display the frame with particles
    cv2.imshow("Spell Casting", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
