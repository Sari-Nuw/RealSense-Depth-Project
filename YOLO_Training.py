from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml") 

# Use the model
if __name__ == '__main__':
    # Train the model
    # More than 2 workers causes issues (may work with different device) 
    # Epochs -> How well trained the model is
    results = model.train(data=r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python Code 3.10\config.yml", workers = 2, epochs=300)