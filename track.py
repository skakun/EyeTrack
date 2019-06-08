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

radius = 7
WIDTH, HEIGHT = 640, 480
MOVE_STEP = 20

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
        # minx -= marginx
        # maxx += marginx
        maxy += marginy
        miny -= marginy
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
        print(shiftbox)
        scope= shiftbox["maxx"] - shiftbox["minx"], shiftbox["maxy"] - shiftbox["miny"]
        snip=EyeSnip(cropped_frame,side,shiftbox,scope)
        snip.check_scope()
        snip.shiftbox_OK=True
        snip.eye_aspect_ratio=ear
        return snip

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
        snip.old_scope = frame.shape
        snip.check_scope()
        snip.shiftbox_OK = True
        snip.eye_aspect_ratio = 1
        return snip


class EyeSnip:
    # def __init__(self, frame, shape, side):
    #     self.snip, self.shiftbox, self.eye_aspect_ratio = EyeSnip.eye_box(frame, shape, side)
    #     self.scope = self.shiftbox["maxx"] - self.shiftbox["minx"], self.shiftbox["maxy"] - self.shiftbox["miny"]
    #     self.scope_OK = not (self.scope[0] <= 0 or self.scope[1] <= 0)
    #     self.shiftbox_OK = not (self.shiftbox["minx"] < 0 or self.shiftbox["maxx"] > WIDTH or
    #                             self.shiftbox["miny"] < 0 or self.shiftbox["maxy"] > HEIGHT)
    #     self.shiftbox_OK=True
    #     self.gray_snip= cv2.cvtColor(self.snip,cv2.COLOR_BGR2GRAY)
    #
    # def get_countur(self):
    #     blurred = cv2.GaussianBlur(self.gray_snip, (5, 5), 0)
    #     thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
    #     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     image=self.snip.copy()
    #     for c in cnts:
    #     # compute the center of the contour
    #         M = cv2.moments(c)
    #         if M["m00"]==0:
    #             continue
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    #     # cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    #     # cv2.putText(image, "center", (cX - 20, cY - 20),
    #     # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #     return image
    #
    # def calc_darkest_point(self):
    #     blur_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
    #     blur_snip = cv2.GaussianBlur(blur_snip, (radius, radius), 0)
    #     (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(blur_snip)
    #     return min_loc
    #
    # def calc_shifted_darkest_point(self):
    #     min_loc = self.calc_darkest_point()
    #     return min_loc[0] + self.shiftbox["minx"], min_loc[1] + self.shiftbox["miny"]
    #
    # def canny_edges(self):
    #     blur_snip = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)
    #     blur_snip = cv2.GaussianBlur(blur_snip, (radius, radius), 0)
    #     low_threshold = 35
    #     high_threshold = low_threshold * 3
    #     eye_edges = cv2.Canny(blur_snip, low_threshold, high_threshold)
    #     return eye_edges
    #
    # def get_thresh(self):
    #     ret, thresh = cv2.threshold(self.gray_snip,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     return thresh
    # def get_blur_thresh(self):
    #     blurred = cv2.GaussianBlur(self.gray_snip, (5, 5), 0)
    #     ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     return thresh
    #
    # def get_segments(self):
    #     thresh=self.get_thresh()
    #     fg=cv2.erode(thresh,None,iterations=2)
    #     bgt=cv2.dilate(thresh,None,iterations=3)
    #     ret,bg=cv2.threshold(bgt,1,128,1)
    #     marker=cv2.add(fg,bg)
    #     marker32 = np.int32(marker)
    #     cv2.watershed(self.snip,marker32)
    #     m = cv2.convertScaleAbs(marker32)
    #     ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #     res = cv2.bitwise_and(self.snip,self.snip,mask = thresh)
    # #       res[marker ==-1]=[255,0,0]
    #     return res

    def __init__(self,snip,side,shiftbox,scope):
        self.snip=snip
        self.side=side
        self.scope=scope
        self.shiftbox=shiftbox
        self.gray_snip= cv2.cvtColor(self.snip,cv2.COLOR_BGR2GRAY)
        self.pupil_position = None
        self.scope_OK = None

    def check_scope(self):
        self.scope_OK = not (self.scope[0] <= 0 or self.scope[1] <= 0)
        return self.scope_OK

    def get_segments(self):
        # im = self.snip.copy()
        gray = cv2.cvtColor(self.snip, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (3, 3),
                                   0)  # można popróbować z różnymi blurami, ale prędzej mniejszy, niż większy wg mnie

        im = cv2.equalizeHist(blurred)  # bardziej uniwersalny, bo rozciąga zakres szarości do stałych granic,
        # ale za to trudniej wyróżnić źrenicę na tle tęczówki (dla ciemnych oczu)

        # ret, im = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)  # chamskie thresholdowanie raczej nie działa, ale można coś pokombonować
        # im = blurred

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 30  # albo 15-30 z użyciem equalizeHist albo trochę więcej bez niego (bez niego można lepiej
        # wyizolować źrenicę od tęczówki, ale trzeba uważać na przypadki, gdy w jakimś
        # ultraświetle źrenica byłaby jaśniejsza od tych powiedzmy 30 ustawionych jako maxThreshold

        params.thresholdStep = 5  # thresholduje w górę (bierze wszystko od danego thresholda do 255) od 'min' do 'max' co 'step';
        # blob musi się znajdować przynajmniej w 2 z tych obrazów binarnych (po thresholdzie), aby był brany pod uwagę

        # params.minRepeatability  # ilość obrazów binarnych po segmentacji, w jakich musi się znajdować blob, żeby był brany pod uwagę; nie ruszałem

        # params.minInertiaRatio  # płaskość czy coś, to jest defaultowo ustawione na min 0.1 (chyba stosunek wysokości
        # do szerokości albo coś takiego); nie ruszałem

        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.filterByConvexity = True
        params.minConvexity = 0.3  # wypukłość; musi być nie za duża, ale największa możliwa
        # params.filterByArea = True
        # params.minArea = 300  # raczej można jeszcze spokojnie zwiększyć
        params.blobColor = 0
        detector = cv2.SimpleBlobDetector_create(params)
        keyPoints = detector.detect(im)
        # global n
        # if keyPoints:
        #     n = 0
        # else:
        #     n += 1

        maxsize = 0
        for keypoint in keyPoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            print(x, y)
            s = keypoint.size
            r = int(math.floor(s / 2))
            cv2.circle(im, (x, y), r, (255, 255, 0), 2)
            if s > maxsize:
                maxsize = s
                self.pupil_position = [self.shiftbox['minx'] + x, self.shiftbox['miny'] + y]
                print('keypoint coordinates: ' + str(self.shiftbox['minx'] + x), str(self.shiftbox['miny'] + y))
        return im


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


def get_pupil_movement(reye, pupil_centered):
    move_left, move_right, move_up, move_down = False, False, False, False
    if reye is not None and reye.pupil_position is not None:
        shiftbox_size = [reye.shiftbox['maxx'] - reye.shiftbox['minx'],
                         reye.shiftbox['maxy'] - reye.shiftbox['miny']]
        print('eye size: ' + str(shiftbox_size))
        x = reye.pupil_position[0]
        y = reye.pupil_position[1]
        x_movement = x - pupil_centered[0]
        y_movement = y - pupil_centered[1]
        # if abs(x_movement) < shiftbox_size[0] // 2 and abs(y_movement) < shiftbox_size[1] // 2:
        if abs(x_movement) > shiftbox_size[0] // 8:
            if x - pupil_centered[0] < 0:
                move_right = True
            else:
                move_left = True
        if abs(y_movement) > shiftbox_size[1] // 8:
            if y_movement < 0:
                move_up = True
            else:
                move_down = True
    return move_left, move_right, move_up, move_down


def move_cursor(move_left, move_right, move_up, move_down, cursor_pos):
    print(move_left, move_right, move_up, move_down)
    if move_left:
        if cursor_pos[0] > MOVE_STEP + radius:
            cursor_pos = (cursor_pos[0] - MOVE_STEP, cursor_pos[1])
        else:
            cursor_pos = (radius, cursor_pos[1])
    elif move_right:
        if cursor_pos[0] < 1200 - MOVE_STEP - radius:
            cursor_pos = (cursor_pos[0] + MOVE_STEP, cursor_pos[1])
        else:
            cursor_pos = (1200 - radius, cursor_pos[1])
    if move_up:
        if cursor_pos[1] > MOVE_STEP + radius:
            cursor_pos = (cursor_pos[0], cursor_pos[1] - MOVE_STEP)
        else:
            cursor_pos = (cursor_pos[0], radius)
    elif move_down:
        if cursor_pos[1] < 960 - MOVE_STEP - radius:
            cursor_pos = (cursor_pos[0], cursor_pos[1] + MOVE_STEP)
        else:
            cursor_pos = (cursor_pos[0], 960 - radius)
    return cursor_pos


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
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture('LiveRecord/5.mp4')
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # set click coordinates helper #
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', cursor_position)

    calibration_frame_count = 0
    pupil_positions = []
    pupil_centered = []
    cursor_pos = (600, 450)
    while True:
        # find face and eyes #
        # capture.read()
        # capture.read()
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) <= 0:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lstart:lend]
        right_eye = shape[rstart:rend]
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)

        reye=EyeSnipper.get_from_hull(frame,shape,'r')
        leye=EyeSnipper.get_from_hull(frame,shape,'l')
        # reye=EyeSnipper.get_from_haar(frame,eye_cascade)
        if not (reye.scope_OK and reye.shiftbox_OK and
                leye.scope_OK and leye.shiftbox_OK):
            print("not ok\n")
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        print("Right eye ear:{}".format(reye.eye_aspect_ratio))

        # display resized right eye in gray #
        greye_area = cv2.cvtColor(reye.snip, cv2.COLOR_BGR2GRAY)
        dim = (greye_area.shape[1] * 3, greye_area.shape[0] * 3)
        resized_greye_area = cv2.resize(greye_area, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Right eye", resized_greye_area)

        # find pupil in eye region #
        # (canny edges) #
        # reye_edges = reye.canny_edges()
        # dim = (reye_edges.shape[1] * 3, reye_edges.shape[0] * 3)
        # resized_reye_edges = cv2.resize(reye_edges, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow("Edges", resized_reye_edges)

        if (rect.top() < 0 or rect.bottom() < 0 or
            rect.left() < 0 or rect.right() < 0
        ):
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # display resized and mark face rectangle #
        # face = frame[rect.top():rect.bottom(), rect.left():rect.right()]
        # dim = (face.shape[1] * 3, face.shape[0] * 3)
        # face = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow("Face", face)

        segments = reye.get_segments()
        # dim = (segments[1] * 3, segments[0] * 3)
        # segments = cv2.resize(segments, dim, interpolation=cv2.INTER_AREA)
        # dim = (segments.shape[1] * 3, segments.shape[0] * 3)
        # resized_segments = cv2.resize(segments, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow("segments", resized_segments)

        # determine pupil movement #
        if calibration_frame_count < 25:
            if reye is not None and reye.pupil_position is not None:
                pupil_positions.append(reye.pupil_position)
                calibration_frame_count += 1

        if calibration_frame_count == 25:
            pupil_centered = calibrate_pupil(pupil_positions)
            calibration_frame_count += 1

        if calibration_frame_count > 25:
            move_left, move_right, move_up, move_down = get_pupil_movement(reye, pupil_centered)
            cursor_pos = move_cursor(move_left, move_right, move_up, move_down, cursor_pos)

        sshot = cv2.imread('idylla.jpg', 0)
        sshot = cv2.cvtColor(np.array(sshot), cv2.COLOR_GRAY2BGR)
        print(cursor_pos)
        cv2.circle(sshot, cursor_pos, radius, (0, 0, 255), 5)
        cv2.imshow("Screenshot", sshot)

        # mark face rectangle and eye contours #
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), YELLOW_COLOR)
        cv2.drawContours(frame, [left_eye_hull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, YELLOW_COLOR, 1)
        if reye is not None and reye.pupil_position is not None:
            cv2.circle(frame, (reye.pupil_position[0], reye.pupil_position[1]), 5, (0, 255, 0), 2)
        if calibration_frame_count > 25:
            cv2.circle(frame, tuple(pupil_centered), 2, (0, 0, 255), 2)

        # dim = (frame.shape[1] // 2, frame.shape[0] // 2)
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
