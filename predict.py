import cv2
from ultralytics import YOLO

def predict_image(image_path, model, classid_classname, output_path):
    image = cv2.imread(image_path)
    results = model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 153, 255), 5) # (204, 255, 204) (127, 0, 255)
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            cv2.putText(image, f'{classid_classname[class_id]}  {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 153, 255), 4) # (178, 102, 255)
        cv2.imwrite(output_path, image)



if __name__ == '__main__':

    # best
    # model = YOLO('./runs/detect/train5/weights/best.pt') # ood_predictions1 # train3 --> train5 both id

    # worst
    # model = YOLO('./runs/detect/train8/weights/best.pt') # ood_predictions4 # train7 --> train8 both id
    
    # WOW on the picture looking good
    model = YOLO('./runs/detect/train10/weights/best.pt') # ood_predictions4 # train9 --> train10 both id
    classid_classname = {0: 'Empty', 1: 'Tweezers', 2: 'Needle_driver'}

    image_path = '/datashare/HW1/labeled_image_data/images/train/0018fa1f-output_0063.png'
    predict_image(image_path, model, classid_classname, 'output.jpg')