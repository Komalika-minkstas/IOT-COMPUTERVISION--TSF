#from PIL import Image
import cv2
#import pytesseract
from imutils.object_detection import non_max_suppression
import numpy as np
import matplotlib.pyplot as plt
import time
#pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
image='words_have_power.jpg'
min_confidence=0.3
width=320
height=320
east="frozen_east_text_detection.pb"
image=cv2.imread(image)

#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#text=pytesseract.image_to_string(image)
#print(text)

orig=image.copy()
(H,W)=image.shape[:2]
(newW, newH)=(width,height)
rW=W/float(newW)
rH=H/float(newH)
image=cv2.resize(image,(newW,newH))
(H,W)=image.shape[:2]
layerNames=[
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
print('loading East Text Detector')
net=cv2.dnn.readNet(east)
#construct a blob from image and the perform a forward pass of the model to obtain the 2 output layer sets
blob=cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68,116.78,103.94),swapRB=True,crop=False)
start=time.time()
net.setInput(blob)

(scores,geometry)=net.forward(layerNames)
end=time.time()

#print timing info on text prediction
print('Text detection took {:.6f} seconds'.format(end-start))
# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows,numCols)=scores.shape[2:4]
rects=[]
confidences=[]

#loop over the number of rows
for y in range(0,numRows):

#extract the scores(probabilities),followed by the geometrical data used to derive the potential bounding box coordinates
    scoresData=scores[0,0,y]
    xData0=geometry[0,0,y]
    xData1=geometry[0,1,y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData=geometry[0,4,y]

    for x in range(0,numCols):
        if scoresData[x]<min_confidence:
            continue
        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY)=(x*4.0, y*4.0)
        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle=anglesData[x]
        cos=np.cos(angle)
        sin=np.sin(angle)
        #use the geomtery volume to derive the width and height of the bounding box
        h=xData0[x]+xData2[x]
        w=xData1[x]+xData3[x]

        #compute the starting and ending (x,y)-coordinates for the text prediction bounding box
        endX=int(offsetX+(cos*xData1[x])+(sin*xData2[x]))
        endY=int(offsetY-(sin*xData1[x])+(cos*xData2[x]))
        startX=int(endX-w)
        startY=int(endY-h)
        #add the bounding boxe coordinates and probability scores to our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

boxes=non_max_suppression(np.array(rects),probs=confidences)
for (startX,startY,endX,endY) in boxes:
    startX=int(startX*rW)
    startY=int(startY*rH)
    endX=int(endX*rW)
    endY=int(endY*rH)

    cv2.rectangle(orig, (startX,startY), (endX,endY),(0,255,0),2)

cv2.imshow('Output',orig)
cv2.waitKey(0)