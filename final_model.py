from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO('./runs/detect/train3/weights/best.pt') # ood_predictions1,2.3
    # model = YOLO('./runs/detect/train6/weights/best.pt') # ood_predictions4
    # model = YOLO('./runs/detect/train9/weights/best.pt') # ood_predictions4
    model = YOLO('./runs/detect/train11/weights/best.pt') # ood_predictions6

    model.train(data='combined_dataset.yaml', epochs=25, batch=32, imgsz=640, device=0)    