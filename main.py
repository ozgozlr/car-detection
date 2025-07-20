from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLOv8n model
model = YOLO(r"runs\detect\train2\weights\best.pt")
results = model(source = "WhatsApp Video 2025-03-08 at 17.47.16.mp4", show=True , conf=0.8, save=True, show_labels = True)




