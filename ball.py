import numpy as np
import cv2
import imutils
from collections import deque
import time

greenLower = (10,120,100)
greenUpper = (40,255,255)
pts = deque(maxlen = 10)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

while(True):
	ret, frame = vs.read()
	if(frame is None):
		break
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11,11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations = 2)
	mask = cv2.dilate(mask, None, iterations = 2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)	
	center = None
	if len(cnts) > 0:
		c = max(cnts, key = cv2.contourArea)
		((x,y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if(radius>10):
			cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
			cv2.circle(frame, center, 5, (255,0,255), -1)

	pts.appendleft(center)	

	for i in range(1, len(pts)):
		if(pts[i-1] is None or pts[i] is None):
			continue
		thickness = int(np.sqrt(10/ float(i+1))*2.5)
		cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)

	cv2.imshow("Frame", frame)
	cv2.imshow("Mask", mask)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()


