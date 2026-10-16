# Real-Time Driver Drowsiness Detection System

## Project Overview

This project aims to detect driver drowsiness using Machine Learning, Deep Learning, and Computer Vision. The system monitors the driver's eyes through a webcam and activates an alarm when prolonged eye closure is detected.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Machine Learning and Deep Learning

## Main Features

- Real-time webcam monitoring
- Facial landmark detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection based on prolonged eye closure
- Audio alarm when drowsiness is detected
- CNN-based image classification experiments

## Dataset

We use the Driver Drowsiness Dataset (DDD), containing images in two categories:

- Drowsy
- Non Drowsy

Dataset source: [https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

## Project Structure

- `src/` — Source code for preprocessing, CNN training, evaluation, and webcam detection
- `models/` — Supporting model files
- `results/` — Dataset split files and visualization results
- `dataset_inspection.py` — Dataset inspection script

## CNN Model

A Convolutional Neural Network (CNN) was trained for drowsiness image classification. The trained model file, `baseline_cnn.keras`, is not included in this repository because it exceeds the GitHub web upload limit. The training code is provided in the source files.

## Real-Time Detection

The webcam prototype uses MediaPipe facial landmarks to calculate the Eye Aspect Ratio (EAR). If the eyes remain closed for approximately two seconds, the alarm activates. It stops after the eyes remain open for approximately three seconds.

## Team

- Razeen Baig
- Sabeel saeed
- touseef

## Project Status

The project includes dataset preprocessing, CNN training and evaluation code, and a real-time webcam-based drowsiness detection prototype. Further testing on independent data is needed to assess generalization.

## Disclaimer

This project is an academic prototype and should not be relied upon as the sole method of preventing driver drowsiness or road accidents.
