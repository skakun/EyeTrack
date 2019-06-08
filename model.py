from imutils import face_utils
import numpy as np
import re
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import os
import sys
import math
import copy
import datetime
from enum import Enum
from scipy.spatial import distance
from statistics import mean

from State import external_state

radius = 5
WIDTH, HEIGHT = 640, 480
class SnipMethod(Enum):
    haar='haar'
    convex='convex'
    skip='skip'
    def __str__(self):
        return self.value
class CenterDetectMethod(Enum):
    blob='blob'
    darkestpoint='darkestpoint'
    def __str(self):
        return self.value
def trans_point(point, old_scope, new_scope, resc=(1, 1)):
    return int(resc[0] * point[0] / old_scope[0] * new_scope[0]), int(resc[1] * point[1] / old_scope[1] * new_scope[1])
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
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
        snip.eye_aspect_ratio=ear
        snip.old_scope=frame.shape
        return snip,snip.check_scope()
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
    @staticmethod
    def skip(frame):
        shiftbox={
                "minx":0,
                "maxx":frame.shape[0],
                "miny":0,
                "maxy":frame.shape[1]
                }
        scope= shiftbox["maxx"] - shiftbox["minx"], shiftbox["maxy"] - shiftbox["miny"]
        snip=EyeSnip(frame,'r',shiftbox,scope)
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
        self.radius=5
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
        blur_snip = cv2.GaussianBlur(blur_snip, (self.radius, self.radius), 0)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(blur_snip)
        return min_loc

    def calc_shifted_darkest_point(self):
        min_loc = self.calc_darkest_point()
        return min_loc[0] + self.shiftbox["minx"], min_loc[1] + self.shiftbox["miny"]

    def canny_edges(self):
        blur_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        blur_snip = cv2.GaussianBlur(blur_snip, (self.radius, self.radius), 0)
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
        im=cv2.cvtColor(self.snip,cv2.COLOR_BGR2GRAY)
        blurred= cv2.GaussianBlur(im, (3, 3), 0)
        blurred=cv2.equalizeHist(blurred)
        params=cv2.SimpleBlobDetector_Params()
        params.minThreshold=0
        params.maxThreshold=30
        params.thresholdStep=5
        params.filterByCircularity=True
        params.minCircularity=0.2
        params.filterByConvexity=True
        params.minConvexity=0.3
#       params.filterByColor=True
        params.blobColor= 0
        detector=cv2.SimpleBlobDetector_create(params)
        keyPoints=detector.detect(blurred)
        x=0
        y=0
        detected=False
        s=None
        print(keyPoints)
        for keypoint in keyPoints:
            detected=True
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size
            r = int(math.floor(s/2))
         #  print x, y
            print(x,y)
            cv2.circle(self.snip, (x, y), r, (255, 255, 0), 2)
            break
        x,y=trans_point((x,y),self.scope,self.old_scope)
        print(x, y)
        print(detected)
        return  im,(x,y),detected,s
#def segment_edges(self):
#       img=self.canny_edges()



class Retina_detector :
    def __init__(self,capture):
        if capture.isdigit():
            self.capture=cv2.VideoCapture(int(capture))
        else:
            self.capture=cv2.VideoCapture(capture)
      # self.height=height
      # self.width=width

        self.radius=5
        self.YELLOW_COLOR = (0, 255, 255)
        self.show_frame=False
        self.show_contour=False
        self.show_snip=False
        self.center=None
        self.detected=False
        self.reye=None
        self.prev_reye_detected=False
        self.prev_leye_detected=False
        self.leye=None
        self.winked_frames=0
        self.snip_method=SnipMethod.haar
        self.no_eye_contact=0
        self.detections=0
        self.pupil_positions=[]
        self.detect_streak=0

        self.calibration_frame_count = 0
        self.pupil_positions_MTARNOW = []
        self.pupil_centered = []
        self.cursor_pos = (600, 450)
    def set_display_opt(self,frame,contour,leye,reye):
        self.show_frame=frame
        self.show_contour=contour
#       self.show_snip=snip
        self.show_leye=leye
        self.show_reye=reye
    def reye_winked(self):
        return  self.prev_reye_detected and not self.reye_detected()
    def leye_winked(self):
        return  self.prev_leye_detected and not self.leye_detected()
    def reye_detected(self):
        if self.reye is None:
            return  False
        return self.reye.eye_aspect_ratio<0.15
    def leye_detected(self):
        if self.leye is None:
            return  False
        return self.leye.eye_aspect_ratio<0.15
    def set_cascade(self,path='haarcascade_eye_tree_eyeglasses.xml'):
        self.eye_cascade = cv2.CascadeClassifier(path)
    def set_predictor(self,path="shape_predictor_68_face_landmarks.dat"):
        self.shape_predictor = path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)

        (self.lstart, self.lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rstart, self.rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    def get_state(self):
        state={}
        state["detected"]=self.detected
        state["alarm"]=str(external_state.alarm)
        state["wanna_talk"]=str(external_state.wanna_talk)
        if not self.detected or ( self.reye_detected() and self.leye_detected()):
            self.no_eye_contact+=1
            self.detect_streak=0
        else:
            self.no_eye_contact=0
            self.detect_streak+=1
        state["no_eye_contact_since_frames"]=self.no_eye_contact
        state["time_stamp"]=str(datetime.datetime.now())
        shiftbox={}
        if self.reye is None or self.reye.shiftbox is None:
            shiftbox = {
                "minx": 0,
                "maxx": 0,
                "miny": 0,
                "maxy": 0
            }
        else:
            shiftbox=self.reye.shiftbox
        sbox={"eye_snip_"+key :int(val) for key,val in shiftbox.items()}
        if not self.detected:
            state["center_x"]=None
            state["center_y"]=None
            state["frame_size_x"]=None
            state["frame_size_y"]=None
            state["right_eye_winked"]=None
            state["left_eye_winked"]=None
            state["retina_size"]=None
        #   return state #TODO add empty shiftbox
        else:
            state["center_x"]=self.center[0]
            state["center_y"]=self.center[1]
            state["frame_size_x"]=self.frame.shape[0]
            state["frame_size_y"]=self.frame.shape[1]
            state["right_eye_winked"]=str(self.reye_winked())
            state["left_eye_winked"]=str(self.leye_winked())
            state["retina_size"]=self.retina_size
       #sbox={"eye_snip_"+key :int(val) for key,val in self.reye.shiftbox.items()}
        return  {**state,**sbox} #wtf, python?

    def detect(self):
        _,self.frame=self.capture.read()
        if self.frame is None:
            return self.get_state()
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        #previous
        self.prev_reye_detected=self.reye_detected()
        self.prev_leye_detected=self.leye_detected()

        if self.snip_method==SnipMethod.haar:
            self.reye,self.detected=EyeSnipper.get_from_haar(self.frame,eye_cascade)
        if self.snip_method==SnipMethod.convex:
        #    self.set_predictor()
            rects=self.detector(gray,0)
            if len(rects)<1:
                if self.show_frame:
                    cv2.imshow('frame',self.frame)
                    cv2.waitKey(1)
                self.detected=False
               #print("none found")
                return self.get_state()
            rect=rects[0]
            shape=self.predictor(gray,rect)
            shape=face_utils.shape_to_np(shape)
            self.reye,self.detected=EyeSnipper.get_from_hull(self.frame,
                    shape,'r')
            self.leye,self.detected=EyeSnipper.get_from_hull(self.frame,
                    shape,'l')
            print("reye aspect ratio={}\n".format(self.reye.eye_aspect_ratio))
        if self.snip_method==SnipMethod.skip:
            self.reye,self.detected=EyeSnipper.skip(self.frame)
        if self.reye is None:
            self.detected=False
            return self.get_state()
        if not self.detected and  self.show_frame:
            cv2.imshow('frame',self.frame)
            cv2.waitKey(1)
            return self.get_state()
        if self.center_detec_method==CenterDetectMethod.blob:
            segframe,ncenter,seg_found,self.retina_size=self.reye.get_segments()
            self.detected=self.detected and seg_found
        if self.detected:
            self.center=ncenter
            self.pupil_positions.append(self.center)
        print("center {}\n".format(self.center))
        cv2.circle(self.frame,ncenter, radius,(0, 255, 0))
        if self.show_frame :
            cv2.imshow("frame", self.frame)
            cv2.waitKey(1)
        if  self.show_reye:
            cv2.imshow("reye",self.reye.snip)
            cv2.waitKey(1)
        if  self.show_leye:
            cv2.imshow("leye",self.leye.snip)
            cv2.waitKey(1)
        if self.show_contour:
            cv2.imshow("segments",segframe)
            cv2.waitKey(1)
        return self.get_state()
