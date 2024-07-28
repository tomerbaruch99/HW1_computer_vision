import cv2
from ultralytics import YOLO
import argparse
import os
import shutil 

def predict_image(image_path, model, classid_classname, output_path, save_dir='./image_predictions_dir'):
    image = cv2.imread(image_path)
    results = model(image)

    # Remove the existing directories if they exist
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    for result in results:
        labels = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
            x_center, y_center, width, height = [box.xywhn[0][i].item() for i in range(4)]
            label = int(box.cls.item())
            labels.append([label, x_center, y_center, width, height])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 153, 255), 5)
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            cv2.putText(image, f'{classid_classname[class_id]}  {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 153, 255), 4) # (178, 102, 255)
        if labels:
            image_path = os.path.join(save_dir, "image.jpg")
            label_path = os.path.join(save_dir, "label.txt")
            cv2.imwrite(image_path, image)
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(" ".join(map(str, label)) + '\n')
        cv2.imwrite(output_path, image)



if __name__ == '__main__':

    # example_image_path = '/datashare/HW1/labeled_image_data/images/train/20_2_24_1_1.jpg'

    parser = argparse.ArgumentParser(description="Predict YOLO on an Image")
    parser.add_argument('--image', type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    if not args.image:
        print("Please provide the path to the image file")
        exit()

    model = YOLO('./best.pt')
    classid_classname = {0: 'Empty', 1: 'Tweezers', 2: 'Needle_driver'}

    predict_image(args.image, model, classid_classname, './image_prediction.jpg')
