# Fire & Smoke Detection Alarm System

A real-time Fire and Smoke Detection Alarm System built using YOLOv8, OpenCV, and Pygame.
This project detects fire and smoke from webcam or video input and automatically triggers an alarm.

Features
Real-time fire and smoke detection
Webcam live detection support
Video file detection support
Automatic alarm system
Bounding box detection
Confidence score display
Fast YOLOv8-based detection
Technologies Used
Python
YOLOv8
OpenCV
Pygame
Ultralytics
Project Structure
Fire-Smoke-Detection-Alarm-System/
│
├── fire_camera.py
├── fire_video.py
├── smoke.pt
├── police.mp3
├── fire3.mp4
├── requirements.txt
└── README.md
Installation
1. Clone Repository
git clone https://github.com/Ghouse707/Fire-Smoke-Detection-Alarm-System.git
2. Open Project Folder
cd Fire-Smoke-Detection-Alarm-System
3. Install Required Libraries
pip install -r requirements.txt
Requirements

Create a requirements.txt file and add:

ultralytics
opencv-python
pygame
torch
torchvision
numpy
Run Webcam Detection
python fire_camera.py
Run Video Detection
python fire_video.py
Controls
Press q to quit the application
How It Works
OpenCV captures webcam/video frames.
YOLOv8 model detects fire and smoke.
Bounding boxes are drawn around detected objects.
Alarm sound starts automatically when fire/smoke is detected.
Alarm stops when detection disappears.
Output
Real-time fire detection
Smoke detection
Alarm activation
Detection confidence score
Live bounding boxes
Future Improvements
Email alert system
SMS notifications
CCTV integration
Snapshot saving
Cloud deployment
Mobile app alerts

# Model File

This project requires a fire/smoke detection YOLO model file.

Download any YOLO fire/smoke detection model (`.pt` file) and place it inside the project folder.

Example:

```bash
smoke.pt
```

Then update the model path in the code if needed:

```python
MODEL_PATH = "smoke.pt"
```

You can use:
- Custom trained YOLO fire detection models
- YOLOv8 fire/smoke models available online

Make sure the model supports fire and smoke detection classes.

---

# Author

Ghouse Pasha

---

