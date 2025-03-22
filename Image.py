import cv2 
from yolov8 import YOLOv8
 
def detect(img_url,model_path):
    # Initialize yolov8 object detector
    # model_path = r"model/yolo_best.onnx"
    yolov8_detector = YOLOv8(model_path, confidence_threshold=0.2, iou_threshold=0.3)

    # Read image
    img = cv2.imread(img_url)
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)

    # Draw detections
    combined_img,is_normal,is_moderate,is_critical,is_minor = yolov8_detector.draw_detections(img)

    print(is_normal,is_moderate,is_critical,is_minor)

    if is_critical:
        print("Critical Accident Detected")
        cv2.putText(combined_img, "Critical", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    elif is_moderate:
        print("Moderate Accident Detected")
        cv2.putText(combined_img, "Moderate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 3, cv2.LINE_AA)

    elif is_minor:
        print("Minor Accident Detected")
        cv2.putText(combined_img, "Minor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    #elif is_normal:
       #cv2.putText(combined_img, "Normal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 3, cv2.LINE_AA)

 
 
    # cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
        # Resize the image
    width = 1000  # specify the width you want
    height = 500  # specify the height you want
    resized_img = cv2.resize(combined_img, (width, height))

    cv2.imshow("Result", resized_img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return
        
while True:
    img_path = input("Enter image path or type break to exit:" ).replace("'","").replace('"','')
    if 'break' in img_path.lower() or 'exit' in img_path.lower(): 
        break
    else:
        detect(img_path,r"model/yolo_best.onnx")
