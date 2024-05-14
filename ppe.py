from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0)  # For Webcam
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture(r"../Videos/ppe-2.mp4")  # For Video

model = YOLO("PPE.pt")

classNames = ['Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Person', 'Vest']
myColor = (0, 0, 255)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if currentClass == 'helmet' or currentClass == 'Vest' or currentClass == 'Mask':
                myColor = (0, 255, 0)
            else:
                myColor = (0, 0, 255)

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                               colorB=myColor, colorT=(255, 255, 255))

    cv2.imshow("Image", img)
    cv2.waitKey(2)
