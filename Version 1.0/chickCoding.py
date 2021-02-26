import cv2
import sys

imagePath = sys.argv[1]
cascPath = "cascade_chick_10stages.xml"

chickCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

wings = chickCascade.detectMultiScale(
	gray,
	scaleFactor = 1.1,
	minNeighbors = 5,
	minSize =(30,30)
	)

#draw rectangle
for(x,y,w,h) in wings:
	cv2.rectangle(image, (x,y,), (x+w, y+h), (0, 255, 0), 2)
	
cv2.imshow("Wings found", image)
cv2.waitKey(0)