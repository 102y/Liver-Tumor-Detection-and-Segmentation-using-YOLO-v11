# Liver-Tumor-Detection-and-Segmentation-using-YOLO-v11
This project aims to use the YOLO v11 model to detect liver tumors in medical images and identify their locations using Segmentation techniques. The model is trained on the Liver Cuy dataset from Roboflow, which contains medical images of livers with labeled tumor locations. The project applies YOLO v11-Instance Segmentation to detect and segment liver tumors in X-ray or MRI images.

Project Execution Steps:
1. Importing the YOLO Library
   from ultralytics import YOLO
The ultralytics library is imported, which provides advanced tools for the latest version of the YOLO model. This version supports Segmentation in addition to detection and classification tasks.
2. Loading the Pre-trained Model
python
model = YOLO("C://Users//NITRO//Desktop//AI Proj//TUMOR_LIVER Computer Vision Project//Liver Cuy.v1i.yolov11//yolo11n-seg.pt")
The pre-trained YOLO v11 model is loaded from the file yolo11n-seg.pt. This model has been trained on the Liver Cuy dataset from Roboflow, which contains liver images with annotated tumor locations.
3. Training the Model on the Dataset
python
train_results = model.train(
    data="C://Users//NITRO//Desktop//AI Proj//TUMOR_LIVER Computer Vision Project//Liver Cuy.v1i.yolov11//data.yaml",
    epochs=10,  # Number of training epochs
    imgsz=640,  # Image size used during training
    device="cpu",  # Device for training ('cpu' or 'cuda' for GPU)
)
The model is trained using the Liver Cuy dataset, which includes liver images and tumor locations. The training parameters such as epochs (number of cycles), image size (imgsz), and the training device (CPU or GPU) are specified. This phase allows the model to learn from the dataset and improve its ability to detect liver tumors.
4. Loading the Trained Model After Training
python
model = YOLO('runs//segment//train//weights//best.pt')
After training, the best-performing model is loaded from runs//segment//train//weights//best.pt. This model contains the optimal weights achieved during training.
5. Testing the Model on a New Image
python
results = model("C://Users//NITRO//Desktop//AI Proj//TUMOR_LIVER Computer Vision Project//Liver Cuy.v1i.yolov11//test//images//Psht_9_JPG.rf.b667033d8df3efe9f0b8da1713942fb3.jpg", save=True)
results[0].show()
After loading the trained model, it is tested on a new liver image that was not part of the training set. The results are saved, and the detected tumor regions are displayed using results[0].show(). The model identifies and segments the tumor areas in the image.
6. Testing the Model on a Batch of Images
python
results = model("test_images", save=True)
This step tests the model on a batch of images stored in the test_images folder. The model processes multiple images and identifies tumors in each, saving and displaying the results after analysis.

Data and Project Structure:
Liver Cuy Dataset: The dataset used for this project is Liver Cuy from Roboflow, which contains liver images with labeled tumor locations. The dataset is pre-configured for YOLO v11 and supports segmentation to accurately detect tumor regions in liver images.

Project Files and Folders:

yolo11n-seg.pt: The pre-trained model for liver tumor detection.
data.yaml: Contains dataset configurations, such as paths to images and labels.
test_images: Folder containing images for testing the model.
runs/segment/train/weights/best.pt: The trained model with the best weights after the training phase.

How to Run the Project:
Setting Up the Environment: Ensure that the required libraries such as ultralytics, torch, and opencv are installed in your Python environment. It's recommended to use a virtual Python environment to manage dependencies.

Preparing the Data: Download the Liver Cuy dataset from Roboflow and configure it for training. The data.yaml file should contain the correct paths to the images and their corresponding annotations.

Training the Model: Train the model using the data specified in data.yaml, setting appropriate training parameters like epochs and image size.

Testing and Evaluation: After training, test the model on a new set of images (those not used during training). Use results[0].show() to visualize the detected tumor regions in each image.

Notes:
GPU Usage: If you have a GPU available, you can replace device="cpu" with device="cuda" to significantly speed up the training process.
Model Fine-tuning: You can adjust training parameters like epochs and image size to optimize the model’s performance. Additionally, the model can be fine-tuned further using techniques such as self-improvement or modifying network layers.
Customization: The project can be customized to detect tumors or anomalies in other types of medical images by retraining the model on different datasets.

Conclusion:
The "Liver Tumor Detection and Segmentation using YOLO v11" project is a powerful tool for analyzing liver medical images. By leveraging YOLO v11-Instance Segmentation, this project enables accurate detection and segmentation of liver tumors, improving the efficiency and accuracy of medical diagnoses.


Dataset link 
https://universe.roboflow.com/brain-tumor-yhoga/liver-cuy
