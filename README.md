# HW1: Computer Vision

This repository contains the code and resources for Homework 1 of the Computer Vision course. The project involves visualizing predictions from a trained model on video data, drawing bounding boxes around detected objects, and annotating them with class labels and confidence scores.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Directories](#files-and-directories)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates how to process video data using a deep learning model for object detection. It reads a video file, runs the model to detect objects in each frame, and visualizes the results by drawing bounding boxes and labels around detected objects.

## Requirements

- Python 3.6 or higher
- OpenCV
- PyTorch
- A compatible deep learning model for object detection

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/tomerbaruch99/HW1_computer_vision.git
    cd HW1_computer_vision
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your video file**:
   Ensure your video file is in a format compatible with OpenCV, such as MP4.

2. **Prepare your model**:
   Load or download a pre-trained model compatible with the object detection task.

3. **Run the visualization script**:
    ```sh
    python vizualization.py
    ```

   Modify the script to specify the path to your video file and output file, as well as the model to be used for predictions.

### Example

```python
from your_model_module import load_model

# Load your model
model = load_model()

# Define the classid to classname mapping
classid_classname = {0: "Class1", 1: "Class2", ...}

# Run the visualization
visualize_predictions('path/to/your/video.mp4', 'path/to/output/video.mp4', model, classid_classname)
