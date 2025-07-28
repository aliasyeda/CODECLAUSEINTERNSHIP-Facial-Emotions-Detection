# CODECLAUSEINTERNSHIP-Facial-Emotions-Detection

Facial Emotion Detection using CNN – JAFFE Dataset
Internship Project ( CodeClause)

## 🔍 Project Overview

This project focuses on detecting facial emotions using a Convolutional Neural Network (CNN) model trained on the JAFFE (Japanese Female Facial Expression) dataset. The model is capable of identifying 7 basic human emotions from grayscale facial images with high accuracy.

The aim is to provide a computer vision-based solution that can automatically recognize emotions such as happy, sad, angry, surprise, neutral, fear, and disgust from static images.

## 🧠 Problem Statement

In real-world human-computer interaction, recognizing emotions plays a crucial role in personal assistants, psychological analysis, safety monitoring, etc. This project solves the problem of manually interpreting facial expressions by enabling machines to automatically detect and classify them using deep learning.

## 🎯 Objectives

Preprocess and organize facial emotion data from the JAFFE dataset

Build and train a CNN-based image classification model

Test the trained model with real-time or custom facial images

Predict and display the emotion associated with each facial image

## 🗃️ Dataset Used

JAFFE (Japanese Female Facial Expression) Dataset

Contains 213 grayscale images of 7 facial expressions from 10 Japanese female subjects.

Image dimensions: 256x256 (converted to 48x48 for model training)

Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## Dataset Download Link:

🔗 https://zenodo.org/record/3451524/files/jaffe.zip
This is the publicly available version hosted on Zenodo. Download and extract it locally to train/test the model.

## ⚙️ Tools & Technologies Used

Jupyter Notebook

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

## 🛠️ Project Workflow

Collected and prepared the JAFFE dataset (including renaming and organizing into emotion-labeled folders).

Preprocessed the data using image resizing, grayscale conversion, and normalization.

Split the dataset into training and testing sets.

Built a CNN model using Keras with multiple convolutional, pooling, and dense layers.

Trained the model on the JAFFE dataset and evaluated its performance.

Saved the trained model for future predictions.

Loaded new unseen facial images and tested them using the saved model.

Successfully predicted the correct emotions such as happy, angry, etc., and displayed them using Matplotlib.

## 📌 Features

Accurate emotion classification from facial images

Works on .tiff, .jpg, or .png grayscale images

Simple, clean interface for prediction

High flexibility to test custom images


## ✅ Final Output
The model successfully predicts and displays the emotion associated with facial images on sample.png :
✅ Predicted Emotion: Angry

## 📚 Learning Outcome

Hands-on experience with CNN model building for image classification

Gained understanding of facial emotion recognition techniques

Learned how to preprocess grayscale images for deep learning

Learned how to integrate OpenCV for image handling and prediction

Understood model saving/loading for practical deployment

## 👨‍💻 Author

Developed by
**Syeda Alia Samia**  
GitHub:[Syeda Alia Samia](https://github.com/your-github-username)
