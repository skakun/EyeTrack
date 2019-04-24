from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import math
from scipy.spatial import distance

radius = 5
WIDTH, HEIGHT = 640, 480
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
n = 0


class EyeSnipper:
    @staticmethod
    def get_from_haar(frame, cascade):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray)
        if len(eyes) == 0:
            return None, False

        eye = eyes[0]
        minx = eye[0]
        maxx = eye[0] + eye[2]
        miny = eye[1]
        maxy = eye[1] + eye[3]
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
        cropped_frame = frame[miny:maxy, minx:maxx]
        scope = shiftbox["maxx"] - shiftbox["minx"], shiftbox["maxy"] - shiftbox["miny"]
        side = 'r'
        snip = EyeSnip(cropped_frame, side, shiftbox, scope)
        snip.check_scope()
        snip.shiftbox_OK = True
        snip.eye_aspect_ratio = 1
        return snip, True


class EyeSnip:
    def __init__(self, snip, side, shiftbox, scope):
        self.snip = snip
        self.side = side
        self.scope = scope
        self.shiftbox = shiftbox
        self.gray_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        self.segmented = self.get_segments()

    def check_scope(self):
        self.scope_OK = not (self.scope[0] <= 0 or self.scope[1] <= 0)
        return self.scope_OK

    def get_segments(self):
        #im = self.snip.copy()
        gray = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # można popróbować z różnymi blurami, ale prędzej mniejszy, niż większy wg mnie

        #im = cv2.equalizeHist(blurred)  # bardziej uniwersalny, bo rozciąga zakres szarości do stałych granic,
                                         # ale za to trudniej wyróżnić źrenicę na tle tęczówki (dla ciemnych oczu)

        #ret, im = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)  # chamskie thresholdowanie raczej nie działa, ale można coś pokombonować
        im = blurred

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 45  # albo 15-30 z użyciem equalizeHist albo trochę więcej bez niego (bez niego można lepiej
                                  # wyizolować źrenicę od tęczówki, ale trzeba uważać na przypadki, gdy w jakimś
                                  # ultraświetle źrenica byłaby jaśniejsza od tych powiedzmy 30 ustawionych jako maxThreshold

        params.thresholdStep = 5  # thresholduje w górę (bierze wszystko od danego thresholda do 255) od 'min' do 'max' co 'step';
                                  # blob musi się znajdować przynajmniej w 2 z tych obrazów binarnych (po thresholdzie), aby był brany pod uwagę

        #params.minRepeatability  # ilość obrazów binarnych po segmentacji, w jakich musi się znajdować blob, żeby był brany pod uwagę; nie ruszałem

        #params.minInertiaRatio  # płaskość czy coś, to jest defaultowo ustawione na min 0.1 (chyba stosunek wysokości
                                 # do szerokości albo coś takiego); nie ruszałem

        #params.filterByCircularity = True
        #params.minCircularity = 0.3
        params.filterByConvexity = True
        params.minConvexity = 0.7  # wypukłość; musi być nie za duża, ale największa możliwa
        params.filterByArea = True
        params.minArea = 700  # raczej można jeszcze spokojnie zwiększyć
        detector = cv2.SimpleBlobDetector_create(params)
        keyPoints = detector.detect(im)
        global n
        if keyPoints:
            n = 0
        else:
            n += 1
        for keypoint in keyPoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size
            r = int(math.floor(s / 2))
            #  print x, y
            cv2.circle(im, (x, y), r, (255, 255, 0), 2)
        return im


shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
YELLOW_COLOR = (0, 255, 255)

(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def color_at_cursor_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x: ' + str(x))
        print('y: ' + str(y))
        print(param[y, x])


def main():
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('77.mp4')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while capture.isOpened():
        # print("Iteration time: {}".format(time.time()-begin_t))

        # find face and eyes #
        capture.read()
        ret, frame = capture.read()
        if not ret:
            break
        reye, OK = EyeSnipper.get_from_haar(frame, eye_cascade)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not OK:
            continue

        # display resized right eye in gray #
        cv2.imshow("reye", reye.snip)

        cv2.namedWindow("segments", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("segments", color_at_cursor_position, param=reye.segmented)
        cv2.imshow("segments", reye.segmented)


if __name__ == '__main__':
    main()

