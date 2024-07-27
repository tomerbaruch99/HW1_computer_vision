# YOLO Model Training and Pseudo-Label Generation

This project focuses on training YOLO models using labeled data and pseudo-labeled data generated from videos. The repository includes scripts for creating dataset configurations, training models, generating pseudo-labels, and visualizing predictions.

## Project Structure

- `train.py`: Trains a YOLO model using the original labeled dataset.
- `generate_pseudo_labels.py`: Generates pseudo-labels from videos using a pretrained YOLO model and combines them with the original dataset.
- `final_model.py`: Trains a YOLO model using the combined dataset (original and pseudo-labeled data).
- `video.py`: Generates predictions on an out-of-distribution (OOD) video using a pretrained YOLO model and visualizes the predictions.
- `predict.py`: Generates predictions on a single image using a pretrained YOLO model and visualizes the predictions.
- `requirements.txt`: Lists the Python package dependencies for the project.
- `dataset.yaml`: Configuration file for the original dataset.
- `combined_dataset.yaml`: Configuration file for the combined dataset.

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/tomerbaruch99/HW1_computer_vision
    cd your-repo
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Create YAML Configuration Files

Generate the YAML configuration files for the original and combined datasets:
```sh
python create_yamls.py
```

### Train YOLO Model on Original Dataset

Train a YOLO model using the original labeled dataset:
```sh
python train.py
```

### Generate Pseudo-Labels

Generate pseudo-labels from videos using a pretrained YOLO model and combine them with the original dataset:
```sh
python generate_pseudo_labels.py
```

### Train YOLO Model on Combined Dataset

Train a YOLO model using the combined dataset (original and pseudo-labeled data):
```sh
python final_model.py
```

### Visualize Predictions on OOD Video

Generate and visualize predictions on an out-of-distribution (OOD) video using a pretrained YOLO model:
```sh
python video.py
```

### Predict on a Single Image

Generate and visualize predictions on a single image using a pretrained YOLO model:
```sh
python predict.py
```

## Dataset Configuration Files

- `dataset.yaml`: Configuration file for the original labeled dataset.
- `combined_dataset.yaml`: Configuration file for the combined dataset (original and pseudo-labeled data).

## Requirements

- opencv-python
- numpy
- torch
- torchvision
- ultralytics==8.2.62

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov8) for the YOLO model implementation.

---