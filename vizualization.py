import cv2
from ultralytics import YOLO

def visualize_predictions(video_path, output_path, model, classid_classname):
    results = model(video_path, stream=True, device=0)
    first_frame = next(results)
    frame = first_frame.orig_img
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    
    for result in results:
        frame = result.orig_img
        for box in result.boxes:
            x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 127), 2) # (0, 255, 0)
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            cv2.putText(frame, f'{classid_classname[class_id]} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 127), 2) # (255, 105, 180) (0, 255, 0)
        out.write(frame)
    out.release()

if __name__ == '__main__':

    # train 4 is model on train3 and then after training on pseudo labels of the 1st video
    ood_video = '/datashare/HW1/ood_video_data/4_2_24_A_1.mp4'
    model = YOLO('./runs/detect/train4/weights/best.pt') # ood_predictions1 # train3 --> train4 1st id
    model = YOLO('./runs/detect/train5/weights/best.pt') # ood_predictions1 # train3 --> train5 both id

    classid_classname = {0: 'Empty', 1: 'Tweezers', 2: 'Needle_driver'}

    visualize_predictions(ood_video, './ood_predictions.mp4', model, classid_classname)

