# GTSRB-ML
This repository contains a complete pipeline for a 43 - class German Traffic Sign Recognition (GTSRB) deep learning project. The solution uses Transfer Learning with a MobileNetv2 backbone, agressive data augmentation, and fine-tuning to achieve maximum performance. The final model is wrapped in a Python class for deployment and demonstrated via an interactive Gradio web application.

## Key-Features
* **Goal:** To accurately calssify images of German Traffic Signs into one of 43 distinct classes.
* **Architecture:** Transfer Learning using a pre-trained MobileNetV2 CNN.
* **Performance:** Achieved a final Validation Accuracy of 99.30% and a Test Set Accuracy of 93.52%.
* **Code Quality:** Implemented a clean, modular piepline using python classes for better maintainability and reproducibility.
* **Deployment Ready:** The model is saved in the Keras format (.keras) and demonstrated using a simple Gradio interface for immediate inference.

## Tech Stack
* **Language:** Python 3.0+
* **Deep Learning:** TesnorFlow/Keras
* **Data Manipulation:** Pandas, Numpy
* **Visualization:** Matplotlib, Seaborn
* **Image Processing:** OpenCV(cv2)
* **Deployment:** Gradio

## Project Structure
The code is organized into modualr classes to separate concerns:
1. Config: A dataclass managing all global constants to ensure2.  reproducibility.
2. GTSRBDataManager: Handles downlaoding, extracting, organizing direcotry structures, and generating TensorFlow Datasets.
3. EDAExplorer: Utilities for Exploratory Data Analysis, including class distribution plots and brightness analysis.
4. TrafficSignModel: Defines the MobileNetV2 architecture, training loop, and fine-tuning logic.
5. TrafficSignPredictor: An inference wrapper to make predicitons on new, unseen images.

## Setup and Installation
This project is designed to run primarily on Google Colab with Google Drive mounted for data storage, but it can be adapted to local machines.

**1. Enviroment Setup**
It is recommended to use a virtual enviroment if running locally.
```bash
# Install necessary dependencies
pip install tensorflow pandas numpy matplotlib seaborn opencv-python scikit-learn gradio
```
**2. Data Preparation**
The project relies on GTSRB dataset. A helper script _download_data.py_ has been provided to automatically download and set up the data for you.
Do update the config class in the notebook to match your new paths.

## Results
* **Confusion Matrix:** The model shows strong diagonal density, indicating correct calssifications across most categories.
* **Metrics:**
  * **Valiadation Accuracy:** ~99%
 * **Test Set Accuracy:** 93.52%
