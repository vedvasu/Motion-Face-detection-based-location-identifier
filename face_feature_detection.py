
import sys
import os
import cv2
import numpy as np

class Detector:
	
        def detect(self, src):
		raise NotImplementedError("Every Detector must implement the detect method.")

class CascadedDetector(Detector):
	
        def __init__(self, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.1, minNeighbors=8, minSize=(5,5),flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
		if not os.path.exists(cascade_fn):
			raise IOError("No valid cascade found for path=%s." % cascade_fn)
		self.cascade = cv2.CascadeClassifier(cascade_fn)
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.minSize = minSize
	
	def detect(self, src):
		if np.ndim(src) == 3:
			src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		src = cv2.equalizeHist(src)
		rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
		if len(rects) == 0:
			return np.ndarray((0,))
		rects[:,2:] += rects[:,:2]
		return rects



cap = cv2.VideoCapture('sample3.3gp')
detector = CascadedDetector(cascade_fn="D:\e-yantra\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")

while(cap.isOpened()):
        ret, frame = cap.read()
        img = np.array(frame, dtype=np.uint8)
        img = cv2.resize(img,(700,480))

        for i,r in enumerate(detector.detect(img)):
                x0,y0,x1,y1 = r
                #cv2.rectangle(img, (x0,y0),(x1,y1),(0,255,0),1)

              
        cv2.imshow('faces', img) 
        if cv2.waitKey(25) & 0xFF == ord('q'):
                break


