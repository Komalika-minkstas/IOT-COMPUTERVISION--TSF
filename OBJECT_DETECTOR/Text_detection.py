import cv2
import pytesseract
from imutils.object_detection import non_max_suppression
import numpy as np
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
def decode_predictions(scores, geometry):
    (numRows, numCols)=scores.shape[2:4]
    rects=[]
    confidences=[]

    for y in range(0,numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geomtery volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding boxe coordinates and probability scores to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return (rects,confidences)

image="believe3.png"
east="frozen_east_text_detection.pb"
min_confidence=0.5
width=320
height=320
padding=0.3
image=cv2.imread(image)
orig=image.copy()
(origW, origH)=image.shape[:2]
(newW,newH)=(width,height)
rW=origW/float(newW)
rH=origH/float(newH)
image=cv2.resize(image,(newW,newH))
(H,W)=image.shape[:2]
layerNames=[
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
print('loading East Text detector..')
net=cv2.dnn.readNet(east)
blob=cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68,116.78,103.94),swapRB=None,crop=False)
net.setInput(blob)
(scores, geometry)=net.forward(layerNames)
(rects,confidences)=decode_predictions(scores,geometry)
boxes=non_max_suppression(np.array(rects),probs=confidences)
results=[]

for (startX,startY,endX,endY) in boxes:
    #scale the bounding box coordinates based on their ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dx=int((endX-startX)*padding)
    dy=int((endY-startY)*padding)

    # apply padding to each side of the bounding box, respectively
    startX=max(0,startX-dx)
    startY = max(0, startY - dy)
    endX = min(origW, endX + (dx*2))
    endY = min(origH, endY + (dy*2))
    #extract the actual padded ROI
    roi=orig[startY:endY,startX:endX]
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config=("-l eng --oem 1 --psm 7")
    text=pytesseract.image_to_string(roi,config=config)
    results.append(((startX,startY,endX,endY),text))
results=sorted(results,key=lambda r:r[0][1])
for((startX,startY,endX,endY),text) in results:
    print("OCR text")
    print("========")
    print("{}\n".format(text))

    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding the text region of the input image

    text="".join([c if ord(c)<128 else "" for c in text]).strip()
    output=orig.copy()
    cv2.rectangle(output,(startX,startY),(endX,endY),(0,0,255),2)
    cv2.putText(output,text,(startX,startY-18),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,0,255),2)
output=cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
cv2.imshow('Text Detection',output)
cv2.waitKey(0)