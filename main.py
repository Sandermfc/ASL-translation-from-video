import cv2
import numpy as np
import math

"""
Image binarisation works best with adequate lighting and a uniform background (ideally opposite colour from the hand)
"""

# CONSTANTS
MINIMUM_POINT_DISTANCE       = 5
MAXIMUM_FINGER_ANGLE 		 = 30
MINIMUM_FINGER_LENGTH        = 30
DEBUG                        = False

fingerTipDistanceVector      = []
fingerLengths				 = []

translationDictionnary = {"00000": '0',
						  "00010": '1',
						  "00110": '2',
						  "00111": '3',
						  "11110": '4',
						  "11111": '5',
						  "01110": '6',
						  "10110": '7',
						  "11010": '8',
						  "11100": '9'}

# p1 is the center point; result is in degrees
def angle_between_points( p0, p1, p2 ):
	a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
	b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
	c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
	return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi

def pointDist(p1, p2):
	return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def getBinaryHand(hsv):
	lowerBound = np.array([0,10,60])
	upperBound = np.array([30,150,255])
	rangeMaskHSV = cv2.inRange(hsv, lowerBound, upperBound)
	closed = cv2.morphologyEx(rangeMaskHSV, cv2.MORPH_CLOSE, np.ones((3,3)))
	closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((3,3)))
	closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3,3)))
	closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3,3)))
	return closed

def getContour(binImg):
	_, contours, hierarchy = cv2.findContours(binImg, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if contours != []:
		max_contour = max(contours, key=cv2.contourArea)

		epsilon = 0.01*cv2.arcLength(max_contour,True)
		contours = cv2.approxPolyDP(max_contour, epsilon, True)
	return contours

def getCenterMass(contour):
	assert(contour != [])
	moments = cv2.moments(contour)
	if moments['m00']!=0:
		cx = int(moments['m10']/moments['m00']) # cx = M10/M00
		cy = int(moments['m01']/moments['m00']) # cy = M01/M00
	return (cx,cy)

def getFingerTipCoordinates(contour, img = None): # Si DEBUG == True, il faut ajouter l'image
	if contour != []:
		if DEBUG: cv2.drawContours(img,[contour],0,(0,255,0),2)

		# Calculate the convex hull
		hull = cv2.convexHull(contour,returnPoints = False) # hull contains indices in contours of hull points

		defects = cv2.convexityDefects(contour,hull) # les points "finger webbing"
			
		numOfFingers = 0
		fingerTipCoordinates = []
		fingerWebbings = []
		prevFar = (-1,-1)
		if defects == None:
			return []
		for i in xrange(0,defects.shape[0]):
			s,e,f,d = defects[i,0]
			start = tuple(contour[s][0])
			end   = tuple(contour[e][0])
			far   = tuple(contour[f][0])
			if DEBUG: 
				cv2.circle(img, start, 8, [255, 0, 0], -1)
				cv2.circle(img, end,   8, [0, 0, 255], -1)
				cv2.circle(img, far,   8, [0, 255, 0], -1)

			if prevFar != (-1, -1):
				angle = angle_between_points(prevFar, start, far)
				if angle <= MAXIMUM_FINGER_ANGLE:
					fingerLength = (pointDist(prevFar, start) + pointDist(far, start)) / 2
					if fingerLength > MINIMUM_FINGER_LENGTH:
						numOfFingers += 1
						fingerTipCoordinates.append(start)
						if DEBUG: cv2.line(img, far, start, [255,0,255],8, -1)
					else:
						pass
						# fingerDictKey = "0"+fingerDictKey
			prevFar = far

		if DEBUG: print(fingerTipCoordinates)
		for fingertip in fingerTipCoordinates:
			cv2.circle(img, fingertip, 8, [255, 0, 0], -1)
		return fingerTipCoordinates

def getDistanceVector(fingerTipCoordinates):
	distVector = []
	for i in xrange(len(fingerTipCoordinates)-1):
		dist = pointDist(fingerTipCoordinates[i], fingerTipCoordinates[i+1])
		distVector.append(dist)
	return distVector

def approximateKey(distVector, fingerTipDistanceVector):
	def checkCombination(key):
		currVal = 0
		perfectDistVect = []

		leftOne = (key[0] == '1')
		for i in xrange(1,len(key)):
			if key[i] == '0':
				if leftOne:
					currVal += fingerTipDistanceVector[i-1]
			if key[i] == '1':
				if not leftOne:
					leftOne = True
				else:
					perfectDistVect.append(currVal + fingerTipDistanceVector[i-1])
					currVal = 0
		score = sum([ abs(dv - pdv) for dv, pdv in zip(distVector, perfectDistVect)])
		return score

	if len(distVector) >= 4:
		return ("11111",0)
	elif len(distVector) == 3:
		combinationsToCheck = ["11110", "11101", "11011", "10111", "01111"]
	elif len(distVector) == 2:
		combinationsToCheck = ["00111", "01011", "01101", "01110", "10011", "10101", "10110", "11001", "11010", "11100"]
	elif len(distVector) == 1:
		combinationsToCheck = ["11000", "10100", "10010", "10001", "01100", "01010", "01001", "00110", "00101", "00011"]
	else:
		return ("00000", 0) # Should never come here, deal with this case outside this function
	bestScore = checkCombination(combinationsToCheck[0])
	bestKey   = combinationsToCheck[0]
	for i in xrange(1,len(combinationsToCheck)):
		score = checkCombination(combinationsToCheck[i])
		if(score < bestScore):
			bestKey = combinationsToCheck[i]
			bestScore = score
	return (bestKey, score)


def main():
	cap = cv2.VideoCapture(0)

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter('all_digits_mask.avi',fourcc, 20.0, (640,480))

	# Initialisation du vecteur de distance
	doneInitialise = False
	while(cap.isOpened() and not doneInitialise):
		ret,img = cap.read()
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		blur = cv2.GaussianBlur(hsv,(9,9),0)

		mask = getBinaryHand(blur)
		"""
		Contours go from right to left for some reason,
		Generally, the right outer fingertip is the first point.
		"""
		contour = getContour(mask)
		fingerTips = getFingerTipCoordinates(contour)

		if fingerTips != None:
			for fingertip in fingerTips:
				cv2.circle(img, fingertip, 8, [255, 0, 0], -1)

		cv2.imshow("img", img)

		k = cv2.waitKey(1)
		if k == 27:
			cap.release()
			cv2.destroyAllWindows()
			exit()
		elif k == 99:
			fingerTipDistanceVector = getDistanceVector(fingerTips)
			if len(fingerTipDistanceVector) == 4:
				doneInitialise = True
			else:
				print("Incapable de lire tout les 5 doights")
	if DEBUG: print(fingerTipDistanceVector)

	while(cap.isOpened()):
		ret,img = cap.read()
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		blur = cv2.GaussianBlur(hsv,(9,9),0)

		mask = getBinaryHand(blur)
		"""
		Contours go from right to left for some reason,
		Generally, the right outer fingertip is the first point.
		"""
		contour = getContour(mask)
		fingerTips = getFingerTipCoordinates(contour, img) #TODO
		key = ""
		if(fingerTips and len(fingerTips) > 1):
			distVector = getDistanceVector(fingerTips)
			key, score = approximateKey(distVector, fingerTipDistanceVector)
			cv2.putText(img, "score: "+str(score)+"\n(lower is better)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			if key[::-1] in translationDictionnary:
				cv2.putText(img, translationDictionnary[key[::-1]], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			else:
				cv2.putText(img, '?', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
		elif(fingerTips and len(fingerTips) == 1):
			# find out which finger based off of finger length (less reliable, but no other choice if only 1 finger)
			cv2.putText(img, "1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
		else:
			cv2.putText(img, "0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

		cv2.imshow("img", img)
		out.write(img)
		#cv2.imshow('mask', mask)

		k = cv2.waitKey(1)
		if k == 27: # :ESC:
			cap.release()
			out.release() # sauve la video
			cv2.destroyAllWindows()
			exit()
		elif k == 99: # 'c'
			# Reinitialiser
			fingerTipDistanceVector = getDistanceVector(fingerTips)
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			cv2.drawContours(mask,[contour],0,(0,255,0),2)
			cv2.imwrite("mask.jpg", mask)

if __name__ == "__main__":
	main()