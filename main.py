import torch
import cv2
import pandas
import requests

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0) 
  
while True:
    success, img = cap.read()
    results = model(img)

    # coordinates
    for r in results.pandas().xyxy:
        for obj in r.values:
            x1, y1, x2, y2 = obj[0:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = obj[4]
            # print("Confidence --->",confidence)

            # class name
            cls = str(obj[6])
            # print("Class name -->", cls)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, cls, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

