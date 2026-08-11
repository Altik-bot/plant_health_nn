Plant Health Neural Network

Deep learning system for plant disease detection from leaf images. Built with PyTorch and deployed with a FastAPI backend. The system classifies multiple disease types and provides visual explanations using Grad-CAM.

Overview
This project focuses on image-based plant disease classification using transfer learning. It follows a structured pipeline from data processing to model evaluation and deployment.

The model takes a leaf image as input and returns:

Predicted disease class
Confidence score
Grad-CAM heatmap showing important regions

Features

Multi-class image classification
Transfer learning with ResNet18 and MobileNet
Training pipeline with augmentation
Evaluation with accuracy, precision, recall, F1 score, confusion matrix
Latency measurement for inference
FastAPI backend for real-time predictions
Grad-CAM visualization for explainability

Tech Stack

Python
PyTorch, torchvision
FastAPI
NumPy, Matplotlib
pytorch-grad-cam
PIL

Project Structure
data/
models/
training/
app/

Data module handles loading and preprocessing
Model module defines CNN backbone and classifier
Training module handles optimization and logging
Evaluation module computes metrics and confusion matrix
Inference module processes new images and returns predictions

Model Details

Backbone: ResNet18, MobileNet
Input size: 224 x 224
Loss: CrossEntropyLoss
Optimizer: Adam
Scheduler: StepLR
Batch size: 32

Training includes augmentation such as rotation, flipping, and brightness changes to improve generalization.

Results

Accuracy reached about 99 percent on selected classes
Strong confusion matrix separation across 9 classes
Fast convergence within 10 epochs
Stable training performance

Model Comparison

ResNet18
Faster inference, about 0.75 to 0.85 seconds
Stable accuracy
MobileNet
Slightly higher peak accuracy
Slower inference, about 1.0 to 1.2 seconds

Accuracy per Epoch
Shows training accuracy progression for both models. Both converge quickly. MobileNet reaches slightly higher peak. ResNet18 remains stable.

Latency Distribution
Shows inference time distribution. ResNet18 has lower and tighter latency. MobileNet is slower but consistent.

API
Endpoints:

GET /health
Returns service status
POST /predict
Input: image file
Output:
predicted class
confidence score
Grad-CAM visualization

Example flow:

Upload image
Model processes input
API returns prediction and explanation

Explainability
Grad-CAM highlights the regions of the leaf used for prediction. This helps verify that the model focuses on disease patterns rather than background noise.

Setup

Clone repository
git clone <your-repo-url>
Create virtual environment
python3 -m venv venv
source venv/bin/activate
Install dependencies
pip install torch torchvision fastapi uvicorn pillow matplotlib pytorch-grad-cam
Train model
python train_eval.py
Run API
uvicorn app:app --host 0.0.0.0 --port 8000
Test endpoint
Send POST request with image to /predict

Future Work

Add more plant species and diseases
Optimize for edge devices
Deploy public API
Add frontend interface
Real-time camera inference

Status
Core system complete.
Model training, evaluation, and Grad-CAM implemented.
API functional.
Model comparison and latency analysis completed.
