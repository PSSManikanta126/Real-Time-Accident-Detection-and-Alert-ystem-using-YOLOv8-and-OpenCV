import cv2
import datetime
from yolov8 import YOLOv8
from threading import Thread
import winsound
import threading

def play_beep():
    winsound.Beep(1000, 500)  # Adjust frequency and duration as needed



def detect(video_file_path,model_path):
    video_capture = cv2.VideoCapture(video_file_path)

    # model_path = r"model/yolo_best.onnx"
    object_detector = YOLOv8(model_path, confidence_threshold=0.5, iou_threshold=0.5)

    send_counter = 0

    while video_capture.isOpened():
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to stop
            break

        try:
            ret, frame = video_capture.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue



        boxes, scores, class_ids = object_detector(frame)
        # combined_image, is_accident = object_detector.draw_detections(frame)
        # Draw detections
        combined_img,is_normal,is_moderate,is_critical,is_minor = object_detector.draw_detections(frame)



        if is_critical:
            cv2.putText(combined_img, "Critical", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        elif is_moderate:
            cv2.putText(combined_img, "Moderate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 3, cv2.LINE_AA)

        elif is_minor:
            cv2.putText(combined_img, "Minor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Resize the image
        width = 1000  # specify the width you want
        height = 700  # specify the height you want
        resized_img = cv2.resize(combined_img, (width, height))

        cv2.imshow("Result", resized_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video_capture.release()
            break

    video_capture.release()

        
while True:
    img_path = input("Enter video path or type break to exit:" ).replace("'","").replace('"','')
    if 'break' in img_path.lower() or 'exit' in img_path.lower(): 
        break
    else:
        detect(img_path,r"model/yolo_best.onnx")
