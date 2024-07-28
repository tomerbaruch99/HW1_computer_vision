import cv2
from ultralytics import YOLO
import os
import shutil
import argparse

def visualize_predictions(video_path, output_path, model, classid_classname, confidence_threshold=0.5, save_dir='./video_predictions'):
    results = model(video_path, stream=True, device=0)
    first_frame = next(results)
    frame = first_frame.orig_img
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    frame_count = 0
    pseudo_frame_dir = os.path.join(save_dir, "frames")
    pseudo_label_dir = os.path.join(save_dir, "labels")

    # Remove the existing directories if they exist
    if os.path.exists(pseudo_frame_dir):
        shutil.rmtree(pseudo_frame_dir)
    if os.path.exists(pseudo_label_dir):
        shutil.rmtree(pseudo_label_dir)
    
    os.makedirs(pseudo_frame_dir, exist_ok=True)
    os.makedirs(pseudo_label_dir, exist_ok=True)

    for result in results:
        labels = []
        frame = result.orig_img
        for box in result.boxes:
            if box.conf >= confidence_threshold:
                x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
                x_center, y_center, width, height = [box.xywhn[0][i].item() for i in range(4)]
                label = int(box.cls.item())
                labels.append([label, x_center, y_center, width, height])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 153, 255), 5) # (127, 0, 255), 5) # (0, 255, 0)
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                cv2.putText(frame, f'{classid_classname[class_id]} , {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 153, 255), 4) # 0.9, (127, 0, 255), 2) # (255, 105, 180) (0, 255, 0)
        
        if labels:
            frame_path = os.path.join(pseudo_frame_dir, f"{frame_count:06d}.jpg")
            label_path = os.path.join(pseudo_label_dir, f"{frame_count:06d}.txt")
            cv2.imwrite(frame_path, frame)
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(" ".join(map(str, label)) + '\n')
            frame_count += 1
        out.write(frame)
    out.release()

if __name__ == '__main__':

    # ood_video1 = '/datashare/HW1/ood_video_data/4_2_24_A_1.mp4'
    # ood_video2 = '/datashare/HW1/ood_video_data/surg_1.mp4'

    parser = argparse.ArgumentParser(description="Visualize YOLO Predictions on a Video")
    parser.add_argument('--video', type=str, required=True, help="Path to the video file")
    args = parser.parse_args()
    if not args.video:
        print("Please provide the path to the video file")
        exit()

    model = YOLO('./best.pt')
    classid_classname = {0: 'Empty', 1: 'Tweezers', 2: 'Needle_driver'}

    visualize_predictions(args.video, './video_predictions.mp4', model, classid_classname)