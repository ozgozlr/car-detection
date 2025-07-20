from ultralytics import YOLO
import torch

def train_yolo():
    # Check if CUDA (GPU) is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO("yolov8s.yaml")

    # Train the model, specifying the device
    results = model.train(data="config.yaml", epochs=100, device=device.type)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_yolo()
