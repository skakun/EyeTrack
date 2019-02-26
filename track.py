from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
YELLOW_COLOR = (0, 255, 255)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
def reyeBox(shape, frame):
	maxy=int((shape[42][1]+shape[41][1])/2)
	miny=int((shape[38][1]+shape[39][1])/2)
	minx=shape[37][0]
	maxx=shape[40][0]
	marginx=1*(maxx-minx)
	marginy=1*(maxy-miny)
	maxy+=marginy
	miny-=marginy
	maxx+=marginx
	minx-=marginx
	print("reye\n")
	print("miny: {} \n maxy: {} \n minx: {} \n maxx: {} \n".format(miny,maxy,minx,maxx))
	shiftbox={
			"minx":minx,
			"maxx":maxx,
			"miny":miny,
			"maxy":maxy
		}
	return frame[miny:maxy,minx:maxx],shiftbox

########def leyeBox(shape, frame):
########	miny=int((shape[44][1]+shape[45][1])/2)
########	maxy=int((shape[48][1]+shape[47][1])/2)
########	minx=shape[43][0]
########	maxx=shape[46][0]
########	marginx=(maxx-minx)
########	marginy=(maxy-miny)
########	maxy+=marginy
########	miny-=marginy
########	maxx+=marginx
########	minx-=marginx
########	print("leye\n")
########	print("miny: {} \n maxy: {} \n minx: {} \n maxx: {} \n".format(miny,maxy,minx,maxx))
########	return frame[miny:maxy,minx:maxx]
def transPoint(point,oldScope,newScope,resc=(1,1)):
	return (int(resc[0]*point[0]/oldScope[0]*newScope[0]) ,int(resc[1]*point[1]/oldScope[1]*newScope[1]))

def main():
	capture = cv2.VideoCapture(2)
	begin_t=time.time()
	radius=5
	while True:
#print("Iteration time: {}".format(time.time()-begin_t))
		_, frame = capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects=detector(gray,0)
		if len(rects) > 0:
			rect = rects[0]
		else:
			cv2.imshow("Frame", frame)
			cv2.waitKey(1)
			continue
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		gframe=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gframe = cv2.GaussianBlur(gframe, (radius,radius), 0)
		(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gframe)
		(reye,rshiftbox)=reyeBox(shape,frame)
		rscope=rshiftbox["maxx"]-rshiftbox["minx"],rshiftbox["maxy"]-rshiftbox["miny"]
		print("Reye scope:/n x: {} /n y:{}".format(rscope[0],rscope[1]))
		if rscope[0]<=0 or rscope[1]<=0:
#		cv2.imshow("Frame", frame)
			cv2.waitKey(1)
			continue
		breye=cv2.cvtColor(reye, cv2.COLOR_BGR2GRAY)
		breye=cv2.GaussianBlur(breye, (radius,radius), 0)

		(rminVal, rmaxVal, rminLoc, rmaxLoc) = cv2.minMaxLoc(breye)
		rshiftedLoc=(rminLoc[0]+rshiftbox["minx"],rminLoc[1]+rshiftbox["miny"])
		print("rminloc:\n \t within eye: {}\n within frame \n \t {}".format(rminLoc,rshiftedLoc))
#	cv2.circle(reye, rminLoc,radius, (0, 255, 0), 2)
#	cv2.imshow("reye",reye)
		cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
		cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
		cv2.circle(frame, rshiftedLoc,radius, (0, 255, 0), 2)
		cv2.imshow("Frame", frame)
		sshot=cv2.imread('idylla.jpg',0)
		cursorPos=transPoint(rminLoc,rscope,sshot.shape[:2],(1,1))
	#sshot = pag.screenshot()
		sshot = cv2.cvtColor(np.array(sshot), cv2.COLOR_RGB2BGR)
		cv2.circle(sshot, cursorPos,radius, (0, 0, 255), 2)
		cv2.imshow("Screenshot", sshot)
		cv2.waitKey(1)
main()
