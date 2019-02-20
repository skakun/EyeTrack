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
def main():
	capture = cv2.VideoCapture(0)
	begin_t=time.time()
	while True:
		print(time.time()-begin_t)
		_, frame = capture.read()
		print("it\n")
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

		cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
		cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)
main()
