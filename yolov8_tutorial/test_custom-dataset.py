from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train2/weights/best.pt')  # load a custom model

# Predict with the model
results = model('test_image.jpg')  # predict on an image