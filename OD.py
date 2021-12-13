import cv2
import numpy as np
import os

whT = 320
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confidences = []

    for output in outputs:
        for detection in output:
            # first 5 values represent x, y, w, h, confidence
            scores = detection[5:]
            # Maximun scroe index
            classId = np.argmax(scores)
            # confidence % of that class
            confidence = scores[classId]

            if confidence> CONFIDENCE_THRESHOLD:
                w,h = int(detection[2] * hT), int(detection[3] * wT)
                x,y = int((detection[0]*wT) - (w/2)), int((detection[1]*wT) - (h/2))

                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidences.append(float(confidence))

    print(len(bbox))
    # only taking 1 box for each object, because there could be several box indicating one object
    indexes = cv2.dnn.NMSBoxes(bbox, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    print(indexes)
    # drawing the boxes
    for i in indexes:
        box = bbox[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



#selection of the webcam
cap = cv2.VideoCapture(0)

#importing the names of all classes
cur_dir = os.getcwd()
classesFile = os.path.join(cur_dir, 'coco_names.txt')
classNames = []

#loading model
modelConfiguration = os.path.join(cur_dir,'yolov3', 'yolov3-320.cfg')
modelWeights = os.path.join(cur_dir,'yolov3', 'yolov3.weights')

#Net work
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

#Setting the network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print('Total classes',len(classNames))

while True:
    # img from webcam
    success, img = cap.read()

    # converting img to blob
    blob = cv2.dnn.blobFromImage(img, 1.0/255, (whT,whT), [0,0,0], 1, crop=False)

    # feeding our image to the network
    net.setInput(blob)

    # getting the output layers (3 layers)
    layerNames = net.getLayerNames()
    outPutLayersIndex = net.getUnconnectedOutLayers()
    print(outPutLayersIndex)
    outputLayerNames = [layerNames[i-1] for i in outPutLayersIndex]

    outputs = net.forward(outputLayerNames)

    '''print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)'''

    findObjects(outputs, img)

    cv2.imshow('Cam', img)
    cv2.waitKey(2)