from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt

radius = 5
WIDTH, HEIGHT = 640, 480


class EyeSnip:
    EYE_CLOSED_EAR_THRESHOLD = 0.12

    def __init__(self, frame, shape, side):
        self.snip, self.shiftbox, self.eye_aspect_ratio = EyeSnip.eye_box(frame, shape, side)
        self.scope = self.shiftbox["maxx"] - self.shiftbox["minx"], self.shiftbox["maxy"] - self.shiftbox["miny"]
        self.scope_OK = not (self.scope[0] <= 0 or self.scope[1] <= 0)
        self.shiftbox_OK = not (self.shiftbox["minx"] < 0 or self.shiftbox["maxx"] > WIDTH or
                                self.shiftbox["miny"] < 0 or self.shiftbox["maxy"] > HEIGHT)
        self.eye_closed = self.eye_aspect_ratio < self.EYE_CLOSED_EAR_THRESHOLD

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
        low_threshold = 30
        high_threshold = low_threshold * 3
        eye_edges = cv2.Canny(blur_snip, low_threshold, high_threshold)
        return eye_edges

    @staticmethod
    def eye_box(frame, shape, side):
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

        marginx = int(0.2 * (maxx - minx))
        marginy = int(0.3 * (maxy - miny))
        minx -= marginx
        maxx += marginx
        maxy += marginy
        miny -= marginy
        shiftbox = {
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy
        }
        return frame[miny:maxy, minx:maxx], shiftbox, eye_aspect_ratio


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
    capture = cv2.VideoCapture(0)

    # set click coordinates helper #
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', cursor_position)

    rear_list = []
    lear_list = []
    # begin_t = time.time()
    while True:
        # print("Iteration time: {}".format(time.time()-begin_t))

        # find face and eyes #
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
        reye = EyeSnip(frame, shape, 'r')
        leye = EyeSnip(frame, shape, 'l')
        if not (reye.scope_OK and reye.shiftbox_OK and
                leye.scope_OK and leye.shiftbox_OK):
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        #print("Right eye:\n Retina pos in frame: {} \n Retina pos in snip: {}\n Ear:{}".format(
        #    reye.calc_shifted_darkest_point(), reye.calc_darkest_point(), reye.eye_aspect_ratio))

        # check eye aspect ratio #
        if len(rear_list) == 1000:
            rear_list = rear_list[1:]
            lear_list = lear_list[1:]
        rear_list.append(reye.eye_aspect_ratio)
        lear_list.append(leye.eye_aspect_ratio)
        if reye.eye_closed and leye.eye_closed:
            print("Eyes closed\nEAR: " + str(reye.eye_aspect_ratio) + ", " + str(leye.eye_aspect_ratio))

        # display resized right eye in gray #
        greye_area = cv2.cvtColor(reye.snip, cv2.COLOR_BGR2GRAY)
        dim = (greye_area.shape[1] * 3, greye_area.shape[0] * 3)
        resized_greye_area = cv2.resize(greye_area, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Right eye", resized_greye_area)

        # find pupil in eye region #

        # (canny edges) #
        reye_edges = reye.canny_edges()
        dim = (reye_edges.shape[1] * 3, reye_edges.shape[0] * 3)
        resized_reye_edges = cv2.resize(reye_edges, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Edges", resized_reye_edges)

        # (darkest point/eye aspect ratio) #
        cv2.circle(reye.snip, reye.calc_darkest_point(), radius, (0, 255, 0), 2)
        cv2.circle(leye.snip, leye.calc_darkest_point(), radius, (0, 255, 0), 2)
        # cv2.circle(frame, reye.calc_shifted_darkest_point(), radius, (0, 255, 0), 2)
        # cv2.circle(frame, leye.calc_shifted_darkest_point(), radius, (0, 255, 0), 2)

        # mark eye contours #
        cv2.drawContours(frame, [left_eye_hull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, YELLOW_COLOR, 1)
        if (
            rect.top() < 0 or rect.bottom() < 0 or
            rect.left() < 0 or rect.right() < 0
        ):
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # display resized and mark face rectangle #
        face = frame[rect.top():rect.bottom(), rect.left():rect.right()]
        dim = (face.shape[1] * 3, face.shape[0] * 3)
        face = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Face", face)

        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), YELLOW_COLOR)
        cv2.imshow("Frame", frame)

        # sshot=cv2.imread('idylla.jpg',0)
        # sshot = cv2.cvtColor(np.array(sshot), cv2.COLOR_RGB2BGR)
        # cursorPos=transPoint(reye.calc_darkest_point(),reye.scope,sshot.shape[:2],(1,1))
        # cv2.circle(sshot, cursorPos,radius, (0, 0, 255), 2)
        # cv2.imshow("Screenshot", sshot)

        if cv2.waitKey(1) == ord('q'):
            break

    # plot eye aspect ratio data #
    plt.plot(rear_list, label='Right eye')
    plt.plot(lear_list, label='Left eye')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
