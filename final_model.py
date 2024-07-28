from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('./initial_train_best.pt')
    model.train(data='combined_dataset.yaml', epochs=25, batch=8, imgsz=640, device=0, augment=True)    