import os
import random
import cv2
import numpy as np
import pygame  # For sound alerts
from ultralytics import YOLO
from tracker import Tracker

# Initialize pygame for sound
pygame.init()
alert_sound = pygame.mixer.Sound(r"data\beep.mp3")  # Replace with your alert sound file
# If you don't have a sound file, you can create one or download a free one

# Paths to the video and output
video_path = os.path.join(r"data\ankara-2.mp4")  # Update with your video path
video_out_path = os.path.join('.', 'data', 'car-track.mp4')

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the video")
    exit()

# Resize frame to improve performance
frame = cv2.resize(frame, (640, 480))

# Video writer to save the output
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),
                          (640, 480))

# Load your custom YOLO model
model = YOLO(r"runs\detect\train2\weights\best.pt")  # Update with your model path

tracker = Tracker()

# Define colors for tracking
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Define your custom class names
class_names = ['car', 'number plate', 'blur number plate', 'two wheeler', 'auto', 'bus', 'truck']  # Update with your class names

# Detection threshold
detection_threshold = 0.7

# Parameters for proximity detection
# Adjust these values based on your specific scenario
proximity_threshold = 0.2  # When object takes up this fraction of frame height, consider it "close"
alert_cooldown = 30  # Frames between alerts to avoid constant beeping
alert_counter = 0

# Variable to track if alert is active
alert_active = False

# Dictionary to store class IDs for each detection
detection_classes = {}

while ret:
    # Resize frame to improve performance
    frame = cv2.resize(frame, (640, 480))
    
    # Run YOLO model on the current frame
    results = model(frame)
    
    # Process detections
    for result in results:
        detections = []
        detection_classes = {}  # Reset for each frame
        
        for i, r in enumerate(result.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = r  # Extract bounding box and class_id
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            
            if score > detection_threshold:
                # Only include the original 5 elements expected by the tracker
                detections.append([x1, y1, x2, y2, score])
                # Store class ID separately with position as key
                key = f"{x1}_{y1}_{x2}_{y2}"
                detection_classes[key] = class_id

        # Update the tracker with the current frame's detections
        tracker.update(frame, detections)

        # Check for close vehicles and draw bounding boxes
        alert_active = False  # Reset for current frame
        
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            
            # Generate a key to look up the class ID
            key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
            
            # Find the closest matching key in detection_classes
            best_key = None
            min_distance = float('inf')
            
            for det_key in detection_classes.keys():
                det_x1, det_y1, det_x2, det_y2 = map(int, det_key.split('_'))
                
                # Calculate distance between detection and track
                center_dist = ((det_x1 + det_x2)/2 - (x1 + x2)/2)**2 + ((det_y1 + det_y2)/2 - (y1 + y2)/2)**2
                
                if center_dist < min_distance:
                    min_distance = center_dist
                    best_key = det_key
            
            # Get the class ID if a matching detection was found
            class_id = detection_classes.get(best_key, 0) if best_key and min_distance < 2500 else 0
            
            # Get information about the detected object
            obj_height = y2 - y1
            frame_height = frame.shape[0]
            
            # Determine if this is a vehicle class
            vehicle_classes = [0, 3, 4, 5, 6]  # car, two wheeler, auto, bus, truck
            is_vehicle = class_id in vehicle_classes
            
            # Check if object is close (takes up significant portion of the screen)
            is_close = (obj_height / frame_height) > proximity_threshold and is_vehicle
            
            # Determine color based on proximity
            box_color = (0, 0, 255) if is_close else colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # Draw class name and track ID
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
            cv2.putText(frame, f'{class_name}', (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Check if any object is close enough to trigger an alert
            if is_close:
                alert_active = True
    
    # Handle alerts
    if alert_active:
        # Display warning symbol
        warning_text = "!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 6)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        # Draw warning symbol with background for visibility
        cv2.putText(frame, warning_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
        
        # Play sound alert (with cooldown to avoid constant beeping)
        if alert_counter <= 0:
            alert_sound.play()
            alert_counter = alert_cooldown
    
    # Decrease alert counter
    if alert_counter > 0:
        alert_counter -= 1

    # Display the frame with detections
    cv2.imshow('Tracking', frame)

    # Write the frame with detections to the output video
    cap_out.write(frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release resources
pygame.quit()
cap.release()
cap_out.release()
cv2.destroyAllWindows()