from PIL import Image
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
image='text.png'
image=cv2.imread(image)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
text=pytesseract.image_to_string(image)
print(text)
cv2.imshow('Output',image)
cv2.waitKey(0)