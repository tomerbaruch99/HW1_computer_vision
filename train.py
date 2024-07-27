from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8n.pt')

    model.train(data='./dataset.yaml', epochs=50, batch=16, imgsz=640, device=0)


