from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import math
from statistics import mean
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
        snip = EyeSnip(cropped_frame, [minx, miny], side, shiftbox, scope)
        snip.check_scope()
        snip.shiftbox_OK = True
        snip.eye_aspect_ratio = 1
        return snip, True


class EyeSnip:
    def __init__(self, snip, coords, side, shiftbox, scope):
        self.snip = snip
        self.coords = coords
        self.side = side
        self.scope = scope
        self.shiftbox = shiftbox
        self.gray_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        self.pupil_position = None
        self.segmented = self.get_segments()

    def check_scope(self):
        self.scope_OK = not (self.scope[0] <= 0 or self.scope[1] <= 0)
        return self.scope_OK

    def get_segments(self):
        #im = self.snip.copy()
        gray = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # można popróbować z różnymi blurami, ale prędzej mniejszy, niż większy wg mnie

        im = cv2.equalizeHist(blurred)  # bardziej uniwersalny, bo rozciąga zakres szarości do stałych granic,
                                         # ale za to trudniej wyróżnić źrenicę na tle tęczówki (dla ciemnych oczu)

        #ret, im = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)  # chamskie thresholdowanie raczej nie działa, ale można coś pokombonować
        #im = blurred

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 30  # albo 15-30 z użyciem equalizeHist albo trochę więcej bez niego (bez niego można lepiej
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

        maxsize = 0
        for keypoint in keyPoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size
            r = int(math.floor(s / 2))
            cv2.circle(im, (x, y), r, (255, 255, 0), 2)
            if s > maxsize:
                maxsize = s
                self.pupil_position = [self.coords[0]+x, self.coords[1]+y]
                print('keypoint coordinates: ' + str(self.coords[0]+x), str(self.coords[1]+y))
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


def calibrate_pupil(pupil_positions):
    max_pos = None
    pos_counts = [[0, []] for _ in range(64)]
    for pos in pupil_positions:
        x = pos[0]
        n = x // 20
        pos_counts[n][1].append(pos)
        pos_counts[n][0] += 1
        if max_pos is None or pos_counts[n][0] > max_pos[0]:
            max_pos = pos_counts[n]

    pupil_positions = max_pos[1]
    max_pos = None
    pos_counts = [[0, []] for _ in range(36)]
    for pos in pupil_positions:
        y = pos[1]
        n = y // 20
        pos_counts[n][1].append(pos)
        pos_counts[n][0] += 1
        if max_pos is None or pos_counts[n][0] > max_pos[0]:
            max_pos = pos_counts[n]

    pupil_centered = []
    if max_pos is not None:
        pupil_centered.append(mean([elem[0] for elem in max_pos[1]]))
        pupil_centered.append(mean([elem[1] for elem in max_pos[1]]))
    return pupil_centered


def main():
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('77.mp4')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # for i in range(120):
    #     capture.read()

    i = 0
    pupil_positions = []
    pupil_centered = []
    while capture.isOpened():
        # print("Iteration time: {}".format(time.time()-begin_t))

        # find face and eyes #
        capture.read()  # every second frame
        ret, frame = capture.read()
        if not ret:
            break
        reye, OK = EyeSnipper.get_from_haar(frame, eye_cascade)

        move_left, move_right, move_up, move_down = False, False, False, False
        print('i = ' + str(i))
        if i < 25:
            if reye is not None and reye.pupil_position is not None:
                pupil_positions.append(reye.pupil_position)
                shiftbox_size = [reye.shiftbox['maxx'] - reye.shiftbox['minx'],
                                 reye.shiftbox['maxy'] - reye.shiftbox['miny']]
                print('eye size: ' + str(shiftbox_size))

        if i == 25:
            print(pupil_positions)
            pupil_centered = calibrate_pupil(pupil_positions)
            print(pupil_centered)

        if i >= 25:
            if reye is not None and reye.pupil_position is not None:
                shiftbox_size = [reye.shiftbox['maxx'] - reye.shiftbox['minx'],
                                 reye.shiftbox['maxy'] - reye.shiftbox['miny']]
                shiftbox_center = [(reye.shiftbox['minx'] + reye.shiftbox['maxx']) // 2,
                                   (reye.shiftbox['miny'] + reye.shiftbox['maxy']) // 2]
                print('eye size: ' + str(shiftbox_size))
                x = reye.pupil_position[0]
                y = reye.pupil_position[1]
                x_movement = x - pupil_centered[0]
                y_movement = y - pupil_centered[1]
                if abs(x_movement) < shiftbox_size[0] // 2 and abs(y_movement) < shiftbox_size[1] // 2:
                    if abs(x_movement) > shiftbox_size[0] // 8:
                        if x - pupil_centered[0] < 0:
                            move_left = True
                        else:
                            move_right = True
                    if abs(y_movement) > shiftbox_size[1] // 8:
                        if y_movement < 0:
                            move_up = True
                        else:
                            move_down = True
        i += 1
        print(move_left, move_right, move_up, move_down)

        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", color_at_cursor_position, param=frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(pupil_positions)
            break
        if not OK:
            continue

        # display resized right eye in gray #
        cv2.imshow("reye", reye.snip)

        # cv2.namedWindow("segments", cv2.WINDOW_AUTOSIZE)
        # cv2.setMouseCallback("segments", color_at_cursor_position, param=reye.segmented)
        cv2.imshow("segments", reye.segmented)


if __name__ == '__main__':
    main()

