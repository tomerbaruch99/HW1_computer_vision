import os
import cv2
import shutil
from ultralytics import YOLO

def load_labeled_data(image_dir, label_dir):
    images = []
    labels = []
    for file in os.listdir(image_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(image_dir, file)
            images.append(img_path)
            label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}")
                continue
            labels.append(label_path)
    return images, labels


def generate_pseudo_labels(model, id_videos, conf_threshold=0.8, save_dir='./pseudo_data'):
    frame_count = 0
    pseudo_image_dir = os.path.join(save_dir, "images")
    pseudo_label_dir = os.path.join(save_dir, "labels")

    # Remove the existing directories if they exist
    if os.path.exists(pseudo_image_dir):
        shutil.rmtree(pseudo_image_dir)
    if os.path.exists(pseudo_label_dir):
        shutil.rmtree(pseudo_label_dir)

    
    os.makedirs(pseudo_image_dir, exist_ok=True)
    os.makedirs(pseudo_label_dir, exist_ok=True)

    for video_path in id_videos:
        results = model(video_path, stream=True, device=0)
        # results = model.track(video_path, stream=True, device=0)

        
        for result in results:
            labels = []
            frame = result.orig_img

            for box in result.boxes:
                # print(box.conf.item())
                if box.conf >= conf_threshold:
                # if float(box.conf.item()) >= conf_threshold:
                    x_center, y_center, width, height = [box.xywhn[0][i].item() for i in range(4)]
                    label = int(box.cls.item())
                    labels.append([label, x_center, y_center, width, height])
            
            if labels:
                frame_path = os.path.join(pseudo_image_dir, f"pseudo_{frame_count:06d}.jpg")
                label_path = os.path.join(pseudo_label_dir, f"pseudo_{frame_count:06d}.txt")
                cv2.imwrite(frame_path, frame)
                with open(label_path, 'w') as f:
                    for label in labels:
                        f.write(" ".join(map(str, label)) + '\n')
                frame_count += 1

def combine_datasets(labeled_data, save_dir='./pseudo_data'):

    pseudo_image_dir = os.path.join(save_dir, "images")
    pseudo_label_dir = os.path.join(save_dir, "labels")

    for image_path, label_path in labeled_data:
        shutil.copy(image_path, pseudo_image_dir)
        shutil.copy(label_path, pseudo_label_dir)
        

if __name__ == '__main__':

    # train2 - robert terminal command - 100 epochs
    # train3 - model.train(data='./dataset.yaml', epochs=50, batch=16, imgsz=640, device=0) # amp = False,
    # train6 - model.train(data='./dataset.yaml', epochs=100, batch=32, imgsz=640, device=0)

    # model = YOLO('./runs/detect/train3/weights/best.pt') # ood_predictions1
    # model = YOLO('./runs/detect/train6/weights/best.pt') # ood_predictions4
    # model = YOLO('./runs/detect/train9/weights/best.pt') # ood_predictions5
    model = YOLO('./runs/detect/train11/weights/best.pt') # ood_predictions6
    
    train_image_dir = '/datashare/HW1/labeled_image_data/images/train'
    train_label_dir = '/datashare/HW1/labeled_image_data/labels/train'

    # id_videos =  ['/datashare/HW1/id_video_data/20_2_24_1.mp4']
    id_videos =  ['/datashare/HW1/id_video_data/20_2_24_1.mp4', '/datashare/HW1/id_video_data/4_2_24_B_2.mp4']

    train_images, train_labels = load_labeled_data(train_image_dir, train_label_dir)

    generate_pseudo_labels(model, id_videos) # , conf_threshold=0.5, save_dir)

    combine_datasets(list(zip(train_images, train_labels)))