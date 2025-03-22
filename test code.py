import cv2
import datetime
from yolov8 import YOLOv8
import threading
import winsound
import requests #used for IOT 
from geopy.geocoders import Nominatim #used for Location Dynmaic or Static

def iot(location, message):
    #link for the location to sent to given number
    url = #link
    querystring = {"authorization":"pPEQomKe3BXBbAbzoLtMplvRe6Sy27wYfDgBYPhdbE0K0QpsZQFIroBo1EpA","message": message,"language":"english","route":"q","numbers":"9492713271"}
    headers = {
        'cache-control': "no-cache"
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    print(response.text)

def play_beep():
    winsound.Beep(1000, 500) # Adjust frequency and duration as needed

def get_location_from_ip():
    # Use IP-based geolocation to obtain the approximate location
    response = requests.get('https://ipinfo.io')
    data = response.json()
    loc = data['loc']
    latitude, longitude = loc.split(',')
    return float(latitude), float(longitude)

def get_location(latitude, longitude):
    geolocator = Nominatim(user_agent="accident_detection")
    location = geolocator.reverse((latitude, longitude))
    return location.address

def detect(model_path):
    # Access webcam using cv2.VideoCapture(0) for default camera
    video_capture = cv2.VideoCapture(1)

    object_detector = YOLOv8(model_path, confidence_threshold=0.5, iou_threshold=0.5)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        # Detect objects in the frame
        boxes, scores, class_ids = object_detector(frame)

        # Draw detections on the frame
        combined_img, is_normal, is_moderate, is_critical, is_minor = object_detector.draw_detections(frame)

        if is_critical:
            cv2.putText(combined_img, "Critical", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            latitude, longitude = get_location_from_ip()
            location = get_location(latitude, longitude)
            print("Location:", location)
            print("Latitude:", latitude)
            print("Longitude:", longitude)
            play_beep()
        elif is_moderate:
            cv2.putText(combined_img, "Moderate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
            latitude, longitude = get_location_from_ip()
            location = get_location(latitude, longitude)
            print("Location:", location)
            play_beep()
        elif is_minor:
            cv2.putText(combined_img, "Minor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            latitude, longitude = get_location_from_ip()
            location = get_location(latitude, longitude)
            print("Location:", location)
            play_beep()
        # Resize the image (optional)
        width = 1000
        height = 700
        resized_img = cv2.resize(combined_img, (width, height))
        cv2.imshow("Result", resized_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Specify the path to your YOLOv8 model weights
model_path = r"model/yolo_best.onnx"

# Call the detect function with the model path
detect(model_path)
