# car-detection
## Overview
This project implements a car detection and alert system designed specifically for e-bikes. Using state-of-the-art computer vision and deep learning techniques, the system detects approaching vehicles from a rear-mounted camera, tracks them persistently, and triggers audio-visual alerts when a vehicle comes dangerously close to the rider.

---

## Features
- **Real-time object detection** using YOLOv8 to identify vehicles behind the e-bike.
- **Deep SORT-based tracking** to assign consistent IDs to moving cars and avoid duplicate alerts.
- **Proximity alert system** with sound and visual warnings activated when vehicles enter a critical distance.

- Custom dataset annotated for urban traffic scenarios to improve detection accuracy.

---

## Technologies Used
- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for video processing and image manipulation
- [Deep SORT](https://github.com/nwojke/deep_sort) for object tracking
- Python 3.8+
- NumPy, SciPy, and other standard scientific libraries


To run the code:

```bash
git clone https://github.com/ozgozlr/car-detection.git
git clone https://github.com/nwojke/deep_sort.git
pip install -r requirements.txt
python tracker-car.py
