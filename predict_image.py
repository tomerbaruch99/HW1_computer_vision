import cv2
from ultralytics import YOLO

def predict_image(image_path):
    image = cv2.imread(image_path)
    results = model(image_path, stream=True, device=0)
    # for result in results:
    for box in results.boxes:
        x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        cv2.putText(image, f'{class_id} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite('output.jpg', image)

if __name__ == '__main__':

    model = YOLO('./runs/detect/train5/weights/best.pt') # ood_predictions1 # train3 --> train5 both id

    image_path = '/datashare/HW1/labeled_image_data/images/train/20_2_24_1_1.jpg'
    predict_image(image_path)