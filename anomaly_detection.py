import cv2
import cv2.cv2
import time
import imutils
import argparse
import numpy as np
import serial
from os.path import dirname, join

from imutils.video import FPS
from imutils.video import VideoStream

# denetim masası donanım ve ses aygıt ve yazıcılardan port numarası bulunur 
bluetooth = serial.Serial( port = "COM5" , baudrate = 9600, timeout = 1)  


model_path = "models/SSD_MobileNet.caffemodel"
prototxt_path = "models/SSD_MobileNet.prototxt"

min_confidence = 0.4

#Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]


colors = np.random.uniform(0, 255, size=(len(labels), 3))

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNet(prototxt_path, model_path)


print('[Status] Starting Video Stream...')
#Initialize Video Stream
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(0)
fps = FPS().start()
second = 0
#Loop Video Stream
while True:
    keyPoint1 = np.zeros(2, dtype = int)
    keyPoint2 = np.zeros(2, dtype = int)
    for k in range(0,2): 
        ret,image1 = cap1.read()
        ret,image2 = cap2.read()
        height1, width1 = image1.shape[0], image1.shape[1]
        height2, width2 = image2.shape[0], image2.shape[1]
        # #Resize Frame to 400 pixels
        # frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        # (h, w) = frame.shape[:2]
    
        #Converting Frame to Blob
        blob1 = cv2.dnn.blobFromImage(cv2.resize(image1, (300, 300)), 
        	0.007843, (300, 300), 127.5)
        nn.setInput(blob1)
        detections1 = nn.forward()
        
        blob2 = cv2.dnn.blobFromImage(cv2.resize(image2, (300, 300)), 
        	0.007843, (300, 300), 127.5)
    
        #Passing Blob through network to detect and predict
        nn.setInput(blob2)
        detections2 = nn.forward()
    
        ##First Camera
        #Loop over the detections
        for i in np.arange(0, detections1.shape[2]):
    
    	#Extracting the confidence of predictions
            confidence = detections1[0, 0, i, 2]
    
            #Filtering out weak predictions
            if confidence > min_confidence:
                
                #Extracting the index of the labels from the detection
                #Computing the (x,y) - coordinates of the bounding box        
                idx = int(detections1[0, 0, i, 1])
                if labels[idx] == "dog":
                    orb = cv2.cv2.SIFT_create()
                    kp = orb.detect(image1, None)
                    kp, des = orb.compute(image1, kp)
                    print(len(kp))
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "cow":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                # elif labels[idx] == "bird":
                #     print("YOLDA HAYVAN VAR")
                #     bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "cat":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "horse":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "sheep":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "person":
                    print("YOLDA İNSAN VAR")
                    bluetooth.write('P'.encode('utf-8'))
                elif labels[idx] == "car": 
                    orb = cv2.cv2.SIFT_create()
                    kp = orb.detect(image1, None)
                    kp, des = orb.compute(image1, kp)
                    keyPoint1[k] = len(kp)
                # elif labels[idx] == "bottle":
                #     orb = cv2.cv2.SIFT_create()
                #     kp = orb.detect(image1, None)
                #     kp, des = orb.compute(image1, kp)
                #     keyPoint1[k] = len(kp)
                
                
                #Extracting bounding box coordinates
                box = detections1[0, 0, i, 3:7] * np.array([width1, height1, width1, height1])
                (startX, startY, endX, endY) = box.astype("int")
    
                #Drawing the prediction and bounding box
                label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                cv2.rectangle(image1, (startX, startY), (endX, endY), colors[idx], 2)
    
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image1, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                
        ##Second Camera
        for i in np.arange(0, detections2.shape[2]):
    
    	#Extracting the confidence of predictions
            confidence = detections2[0, 0, i, 2]
    
            #Filtering out weak predictions
            if confidence > min_confidence:
                
                #Extracting the index of the labels from the detection
                #Computing the (x,y) - coordinates of the bounding box        
                idx = int(detections2[0, 0, i, 1])
                if labels[idx] == "dog":
                    orb = cv2.cv2.SIFT_create()
                    kp = orb.detect(image2, None)
                    kp, des = orb.compute(image2, kp)
                    print(len(kp))
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "cow":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                # elif labels[idx] == "bird":
                #     print("YOLDA HAYVAN VAR")
                #     bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "cat":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "horse":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "sheep":
                    print("YOLDA HAYVAN VAR")
                    bluetooth.write('H'.encode('utf-8'))
                elif labels[idx] == "person":
                    print("YOLDA İNSAN VAR")
                    bluetooth.write('P'.encode('utf-8'))
                elif labels[idx] == "car": 
                    orb = cv2.cv2.SIFT_create()
                    kp = orb.detect(image2, None)
                    kp, des = orb.compute(image2, kp)
                    keyPoint2[k] = len(kp)
                # elif labels[idx] == "bottle":
                #     orb = cv2.cv2.SIFT_create()
                #     kp = orb.detect(image2, None)
                #     kp, des = orb.compute(image2, kp)
                #     keyPoint2[k] = len(kp)
                    
                
                #Extracting bounding box coordinates
                box = detections2[0, 0, i, 3:7] * np.array([width2, height2, width2, height2])
                (startX, startY, endX, endY) = box.astype("int")
    
                #Drawing the prediction and bounding box
                label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                cv2.rectangle(image2, (startX, startY), (endX, endY), colors[idx], 2)
    
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image2, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                
    if keyPoint1[0] != 0: 
        if abs(keyPoint1[1] - keyPoint1[0])<20:
            print("YOLDA ARABA DURUYOR")
            bluetooth.write('S'.encode('utf-8'))
    
    if keyPoint2[0] != 0:
        if abs(keyPoint2[1] - keyPoint2[0])<20:
            print("YOLDA ARABA DURUYOR")
            bluetooth.write('S'.encode('utf-8'))

    cv2.imshow("Frame1", image1)
    cv2.imshow("Frame2",image2)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
bluetooth.close()