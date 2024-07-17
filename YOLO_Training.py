from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")
#model = YOLO(r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\runs\detect\train\weights\last.pt")

#results = model.train(data="allmushrooms_config.yml", epochs = 5) # train the model

# Use the model
if __name__ == '__main__':
    # Train the model
    # More than 2 workers causes issues (may work with different device) 
    # Epochs -> How well trained the model is
    # results = model.train(data=r"C:\Users\Molly\Desktop\Mushrooms A\obj_train_data\config.yml", workers = 2, epochs=25, patience = 0)
    results = model.train(data=r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\allmushrooms_config.yml", workers = 2, epochs=5, imgsz = 640)