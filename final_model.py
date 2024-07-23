from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('./runs/detect/train3/weights/best.pt') # ood_predictions1

    model.train(data='combined_dataset.yaml', epochs=25, batch=16, imgsz=640, device=0)
    