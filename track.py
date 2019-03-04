from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
from scipy.spatial import distance 

radius=5

class eyeSnip():
	def __init__(self,frame,hull,rl):
		self.left_or_right=rl
		if rl=='r':
			self.snip,self.shiftbox,self.eye_aspect_ratio=eyeSnip.reye_box(hull, frame)
		self.scope=self.shiftbox["maxx"]-self.shiftbox["minx"],self.shiftbox["maxy"]-self.shiftbox["miny"]
		self.scope_OK=not (self.scope[0]<=0 or self.scope[1]<=0)
	def calc_darkest_point(self):
		blur_snip=cv2.cvtColor(self.snip,cv2.COLOR_BGR2GRAY)
		blur_snip= cv2.GaussianBlur(blur_snip, (radius,radius), 0)
		(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur_snip)
		return minLoc
	def calc_shifted_darkest_point(self):
			minLoc=self.calc_darkest_point()
			return (minLoc[0]+self.shiftbox["minx"],minLoc[1]+self.shiftbox["miny"])
	@staticmethod
	def reye_box(shape, frame):
		maxy=int((shape[42][1]+shape[41][1])/2)
		miny=int((shape[38][1]+shape[39][1])/2)
		minx=shape[37][0]
		maxx=shape[40][0]
		eye_aspect_ratio=( distance.euclidean(shape[42],shape[38])+distance.euclidean(shape[39],shape[41]))/(2*distance.euclidean(shape[37],shape[40]))
		marginx=int(1*(maxx-minx))
		marginy=int(1*(maxy-miny))
		maxy+=marginy
		miny-=marginy
		maxx+=marginx
		minx-=marginx
		shiftbox={
			"minx":minx,
			"maxx":maxx,
			"miny":miny,
			"maxy":maxy
		}
		return frame[miny:maxy,minx:maxx],shiftbox,eye_aspect_ratio

shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
YELLOW_COLOR = (0, 255, 255)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
def transPoint(point,oldScope,newScope,resc=(1,1)):
	return (int(resc[0]*point[0]/oldScope[0]*newScope[0]) ,int(resc[1]*point[1]/oldScope[1]*newScope[1]))

def main():
	capture = cv2.VideoCapture(0)
	begin_t=time.time()
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
		reye=eyeSnip(frame,shape,'r')
		if not reye.scope_OK:
		#cv2.imshow("Frame",frame)		
			cv2.waitKey(1)
			continue
		print("Right eye:\n Retina pos in frame: {} \n Retina pos in snip: {}\n Ear:{}".format(reye.calc_shifted_darkest_point(),reye.calc_darkest_point(),reye.eye_aspect_ratio ))
		marked_reye=reye.snip
		cv2.circle(marked_reye,reye.calc_darkest_point(),radius, (0, 255, 0), 2)
		cv2.imshow("reye",marked_reye)
		cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
		cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
		cv2.circle(frame, reye.calc_shifted_darkest_point(),radius, (0, 255, 0), 2)
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)
		sshot=cv2.imread('idylla.jpg',0)
		sshot = cv2.cvtColor(np.array(sshot), cv2.COLOR_RGB2BGR)
		cursorPos=transPoint(reye.calc_darkest_point(),reye.scope,sshot.shape[:2],(1,1))
		cv2.circle(sshot, cursorPos,radius, (0, 0, 255), 2)
		cv2.imshow("Screenshot", sshot)
		cv2.waitKey(1)
main()