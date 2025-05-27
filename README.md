# Age and Gender Prediction Model

This repository features a deep learning model for age and gender prediction from facial images, built with TensorFlow/Keras CNNs. Optimized for Google Colab, it offers real-time webcam and file upload prediction capabilities.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Features](#features)
-   [Technologies](#technologies)
-   [Setup (Google Colab)](#setup-google-colab)
-   [Dataset](#dataset)
-   [Model Architecture](#model-architecture)
-   [Usage](#usage)
-   [Future Enhancements](#future-enhancements)

## Project Overview

Develops and trains a deep learning model to predict age and gender from facial images using a multi-output CNN architecture for simultaneous regression (age) and classification (gender).

## Features

-   **Age Prediction:** Estimates age from face images.
-   **Gender Prediction:** Classifies gender (Male/Female) from face images.
-   **Real-time & File Upload:** Supports predictions from webcam (Colab only) or uploaded image files.

## Technologies

-   Python
-   TensorFlow / Keras (Deep Learning)
-   OpenCV (`cv2`) (Image Processing)
-   NumPy, Pandas, Matplotlib, Scikit-learn
-   Google Colab specific libraries for integration (`google.colab.*`)

## Setup (Google Colab)

1.  **Open in Colab:** Upload `age_&_gender_prediction.py` to Google Colab.
2.  **Mount Google Drive:** The script auto-mounts your Drive. Authorize when prompted.
3.  **Dataset:**
    * Download [UTKFace Dataset](https://www.kaggle.com/datasets/jangedubey/utkface-new) (`archive.zip`).
    * Upload `archive.zip` to your Google Drive (e.g., `MyDrive/datasets/`).
    * **Update `zip_path` in script:**
        ```python
        zip_path = "/content/drive/MyDrive/datasets/archive.zip" # CHANGE THIS PATH
        ```
    * The script handles extraction.

  **OR you can download the model from trained model folder and run the remaing code for Real-time or File Upload.**

*(For local setup, manually manage dataset, install dependencies, and replace Colab-specific webcam/file I/O with standard methods.)*

## Dataset

Uses the **UTKFace Dataset**: over 20,000 annotated facial images (age 0-116, gender, ethnicity), crucial for training a robust model.

## Model Architecture

A CNN using Keras Functional API with a multi-output design:

-   **Input:** Preprocessed facial images.
-   **Conv Base:** `Conv2D` and `MaxPooling2D` layers for feature extraction.
-   **Shared Dense Layers:** Process extracted features.
-   **Age Branch:** `Dense` + `relu` for age regression.
-   **Gender Branch:** `Dense` + `sigmoid` for gender classification.
-   **Training:** Adam optimizer; MAE for age, Binary Crossentropy for gender.

## Usage

After setup and script execution (includes training):

1.  **Webcam Predict (Colab Only):** `capture_and_predict()` function will access your webcam.
2.  **File Upload Predict (Colab Only):** `upload_and_predict()` function will prompt for an image file upload.

Results display predicted age/gender and the image with predictions.

## Future Enhancements

-   **Transfer Learning:** Integrate pre-trained CNNs (ResNet, VGG) for improved accuracy.
-   **Ethnicity Prediction:** Extend the model for multi-task learning, including ethnicity.
-   **Robustness:** Enhance performance across varying lighting, poses, and occlusions.
-   **Deployment:** Explore deploying as a web app (Flask/Django) or mobile app (TF Lite).
-   **Advanced Evaluation:** Implement more detailed metrics and visualizations.

## Acknowledgements

Thanks to the UTKFace Dataset creators, TensorFlow/Keras, OpenCV, and Google Colab for their invaluable contributions.
