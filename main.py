from ultralytics import YOLO
import os

train_images_path = "C:/Users/latel/PycharmProjects/DogRecognition_le_vrai/Dataset/Images/train"
val_images_path = "C:/Users/latel/PycharmProjects/DogRecognition_le_vrai/Dataset/Images/val"

print(f"Train images path exists: {os.path.exists(train_images_path)}")
print(f"Val images path exists: {os.path.exists(val_images_path)}")

model = YOLO('runs/Detect/train/weights/best.pt')

results = model.predict(source='Dataset/golden_x_labrador.png', save=True)

