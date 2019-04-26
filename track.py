from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import os
import sys
import math
from scipy.spatial import distance
radius = 5
WIDTH, HEIGHT = 640, 480
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
import argparse
class EyeSnipper:
    @staticmethod
    def eye_box_hull(frame, shape, side):
        minx, maxx, miny, maxy, eye_aspect_ratio = 0, 0, 0, 0, 0
        if side == 'r':
            miny = shape[37][1] if shape[37][1] < shape[38][1] else shape[38][1]
            maxy = shape[40][1] if shape[40][1] > shape[41][1] else shape[41][1]
            minx = shape[36][0]
            maxx = shape[39][0]
            eye_aspect_ratio = (distance.euclidean(shape[37], shape[41]) + distance.euclidean(shape[38], shape[40])) / (
                                2 * distance.euclidean(shape[36], shape[39]))
        elif side == 'l':
            miny = shape[43][1] if shape[43][1] < shape[44][1] else shape[44][1]
            maxy = shape[46][1] if shape[46][1] > shape[47][1] else shape[47][1]
            minx = shape[42][0]
            maxx = shape[45][0]
            eye_aspect_ratio = (distance.euclidean(shape[43], shape[47]) + distance.euclidean(shape[44], shape[46])) / (
                                2 * distance.euclidean(shape[42], shape[45]))

        marginx = int(0.1 * (maxx - minx))
        marginy = int(0.1 * (maxy - miny))
    #   minx -= marginx
    #   maxx += marginx
     #  maxy += marginy
     #  miny -= marginy
        shiftbox = {
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy
        }
        return frame[miny:maxy, minx:maxx], shiftbox, eye_aspect_ratio
    @staticmethod
    def get_from_hull(frame,shape,side):
        cropped_frame,shiftbox,ear=EyeSnipper.eye_box_hull(frame,shape,side)
        scope= shiftbox["maxx"] - shiftbox["minx"], shiftbox["maxy"] - shiftbox["miny"]
        snip=EyeSnip(cropped_frame,side,shiftbox,scope)
        snip.check_scope()
        snip.shiftbox_OK=True
        snip.eye_aspect_ratio=1
        snip.old_scope=frame.shape
        return snip
    @staticmethod
    def get_from_haar(frame,cascade):
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes=cascade.detectMultiScale(gray)
        if len(eyes)==0:
            return None,False
        eye=eyes[0]
        minx=eye[0]
        maxx=eye[0]+eye[2]
        miny=eye[1]
        maxy=eye[1]+eye[3]
        marginx = int(0.2 * (maxx - minx))
        marginy = int(0.2 * (maxy - miny))
   #    minx += marginx
   #    maxx -= marginx
        maxy -= marginy
        miny += marginy
        shiftbox = {
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy
        }
        cropped_frame= frame[miny:maxy, minx:maxx]
        scope= shiftbox["maxx"] - shiftbox["minx"], shiftbox["maxy"] - shiftbox["miny"]
        side='r'
        snip=EyeSnip(cropped_frame,side,shiftbox,scope)
        snip.old_scope=frame.shape
        snip.check_scope()
        snip.shiftbox_OK=True
        snip.eye_aspect_ratio=1
        return snip,True
class EyeSnip:
    def __init__(self,snip,side,shiftbox,scope):
        self.snip=snip
        self.side=side
        self.scope=scope
        self.shiftbox=shiftbox
        self.gray_snip= cv2.cvtColor(self.snip,cv2.COLOR_BGR2GRAY)
    def check_scope(self):
        self.scope_OK=not (self.scope[0] <= 0 or self.scope[1] <= 0)
        return self.scope_OK
    def get_countur(self):
        blurred = cv2.GaussianBlur(self.gray_snip, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        image=self.snip.copy()
        for c in cnts:
        # compute the center of the contour
            M = cv2.moments(c)
            if M["m00"]==0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
       #cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return image
    def calc_darkest_point(self):
        blur_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        blur_snip = cv2.GaussianBlur(blur_snip, (radius, radius), 0)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(blur_snip)
        return min_loc

    def calc_shifted_darkest_point(self):
        min_loc = self.calc_darkest_point()
        return min_loc[0] + self.shiftbox["minx"], min_loc[1] + self.shiftbox["miny"]

    def canny_edges(self):
        blur_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        blur_snip = cv2.GaussianBlur(blur_snip, (radius, radius), 0)
        low_threshold = 35
        high_threshold = low_threshold * 3
        eye_edges = cv2.Canny(blur_snip, low_threshold, high_threshold)
        return eye_edges

    def get_thresh(self):
        ret, thresh = cv2.threshold(self.gray_snip,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh
    def get_blur_thresh(self):
   #    blurred = cv2.GaussianBlur(self.gray_snip, (5, 5), 0)
        blurred = cv2.GaussianBlur(self.snip, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def get_segments(self):
        im=self.snip.copy()
        blurred= cv2.GaussianBlur(im, (5, 5), 0)

        params=cv2.SimpleBlobDetector_Params()
#       params.filterByCircularity=True
#       params.minCircularity=0.2
        params.filterByConvexity=True
        params.minConvexity=0.8
        params.filterByColor=True
        params.blobColor= 0
        detector=cv2.SimpleBlobDetector_create(params)
        keyPoints=detector.detect(im)
        x=0
        y=0
        detected=False
        for keypoint in keyPoints:
            detected=True
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size
            r = int(math.floor(s/2))
         #  print x, y
            cv2.circle(im, (x, y), r, (255, 255, 0), 2)
            break
        x,y=trans_point((x,y),self.scope,self.old_scope)
        return  im,(x,y),detected
#def segment_edges(self):
#       img=self.canny_edges()


shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
YELLOW_COLOR = (0, 255, 255)

(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def trans_point(point, old_scope, new_scope, resc=(1, 1)):
    return int(resc[0] * point[0] / old_scope[0] * new_scope[0]), int(resc[1] * point[1] / old_scope[1] * new_scope[1])


def cursor_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x: ' + str(x))
        print('y: ' + str(y))

def main():
    capture_num=2
    height=720
    width=1280
    parser=argparse.ArgumentParser()
    parser.add_argument('-f','--show_frame',help='show frame',action='store_true')
    parser.add_argument('-s',"--show_eye_snip",help='show eye snip',action='store_true')
    parser.add_argument('-c',"--show_contour",help='show contour',action='store_true')
    parser.add_argument('-w','--width',help='width of frame' )
    parser.add_argument('-t','--tallness',help='height of frame')
    input_group=parser.add_mutually_exclusive_group()
    input_group.add_argument('-p','--capture',help='number of capture device')
    input_group.add_argument('-v','--video',help='path to video file')
    args=parser.parse_args()
    if args.capture is not None:
        capture_num=int(args.capture)
    if args.tallness is not None:
        height=args.tallness
    if  args.width is not None:
        width=args.width
    if args.video is not None:
        capture=cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture(capture_num)
    print(capture_num)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if args.show_frame:
        # set click coordinates helper #
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', cursor_position)

    # begin_t = time.time()
    center=None
    while True:
        # print("Iteration time: {}".format(time.time()-begin_t))

        # find face and eyes #
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reye,OK=EyeSnipper.get_from_haar(frame,eye_cascade)
        if reye is None:
            continue
        if not OK and args.show_frame:
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            continue
     #  print("Right eye:\n Retina pos in frame: {} \n Retina pos in snip: {}\n Ear:{}".format(
     #      reye.calc_shifted_darkest_point(), reye.calc_darkest_point(), reye.eye_aspect_ratio))
        toshow=frame.copy()
        cv2.circle(toshow, reye.calc_shifted_darkest_point(),radius,(0, 255, 0))

        # display resized right eye in gray #
        if args.show_frame :
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        if args.show_eye_snip:
            cv2.imshow("reye",reye.snip)
            cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break
        if args.show_contour:
            segframe,ncenter,detected=reye.get_segments()
            if detected:
                center=ncenter
            print("Right retina center :{}\n".format(center))
            cv2.imshow("segments",segframe)
            cv2.waitKey(1)
#       cv2.imshow("contour",reye.get_countur())

if __name__ == '__main__':
    main()
