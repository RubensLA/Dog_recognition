# Dog_recognition

📋 Project Results and Overview

This project focuses on the accurate recognition of dogs within images, encompassing both dog detection and breed identification. Using advanced computer vision and machine learning techniques.

 Key Objectives
- Develop a system capable of detecting dogs in images with high accuracy.
- Classify dog breeds to account for the vast diversity among canine species.

Key Results
- Achieved good performance in detecting dogs in diverse environments.
- Implemented a breed classification system with high precision.

Insights and Significance
- The project underscores the importance of tailored models for specific tasks like breed recognition, as generic models may lack the necessary granularity.
- Applications of this technology range from pet identification and shelter management to security systems.

File Structure
Below is the organized file structure for the project:

![image](https://github.com/user-attachments/assets/a3d8150e-1188-4fd4-9dab-c0b614279947)

 # To set up and run the project, follow the steps below:

Clone the Repository
Clone the project folder to your local machine.

Install Dependencies
Ensure you have Python 3.8+ installed. Set up a virtual environment :

python -m venv venv
venv\Scripts\activate          # On Windows

Then, install the required dependencies:
pip install ultralytics opencv-python

Prepare the Dataset
Ensure the dataset structure matches the Dataset/ folder as shown above.
Update data.yaml to point to the correct paths for your training and validation datasets.

Train the Model

To train the YOLO model, edit main.py and run:

replace the line : model = YOLO('runs/Detect/train/weights/best.pt') by model = YOLO("yolov8n.pt") to load the original untrained model

replace the last line wich is model.predict(source='Dataset/golden_x_labrador.png', save=True) by model.train(data='data.yaml', epochs=50, batch=16, imgsz=640)

Run Predictions
run the following code.

python main.py
The script saves results (e.g., bounding boxes and classifications) in the runs/ folder.

# Code Explanation
The core logic of the project is in main.py:

![image](https://github.com/user-attachments/assets/857bcda2-2edc-4046-8791-b32bfe6a18ef)

# Training Metrics

![image](https://github.com/user-attachments/assets/88adc559-661c-424f-936d-1234c3647146)

Explanation:
Box Loss: Measures the error in bounding box localization.
Class Loss: Measures the classification error for detected objects.
DFL Loss: Refers to Distribution Focal Loss, used for bounding box regression.

# Validation Metrics

![image](https://github.com/user-attachments/assets/1e8d9460-73cf-4d5b-a1ff-454969c31df1)

Explanation:

Precision (P): Indicates how many of the detected objects are correctly identified as dogs.
Recall (R): Measures the system's ability to detect all dog instances in the dataset.
mAP@50: The mean Average Precision at IoU threshold 0.50, a standard metric for object detection tasks.
mAP@50-95: The mean Average Precision averaged across multiple IoU thresholds (0.50 to 0.95), reflecting model performance across stricter thresholds.


# Key References

YOLOv8 (Ultralytics)

The project uses the YOLOv8 architecture for object detection and classification.
Official repository and documentation: https://github.com/ultralytics/ultralytics

# Known Issues and Limitations
Dataset Size

- Performance may vary depending on the diversity and size of the training dataset. More diverse samples can improve accuracy.

Lighting and Background Variations

- The model may underperform in images with poor lighting or cluttered backgrounds.

Breed Classification

- Some breeds are not classified with this project due to the performance of my computer. Only 20 are classified and more could be.

# Potential Improvements
Real-Time Inference

- Optimize the system for real-time applications by fine-tuning the model for faster inference.

Multi-Animal Detection

- Extend the model to detect and classify multiple animals within a single image.

Improved Breed Identification

- Add a larger, more diverse dataset to refine breed classification accuracy and to classify more breeds.






