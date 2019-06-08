from model import Retina_detector
import pyautogui as pag
import cv2
import numpy as np
from statistics import mean
##########################################################
class Control:
    radius=5
    MOVE_STEP = 20
    def __init__(self):
        self.move_mode_open=True
        pag.FAILSAFE=False
    def calibrate_pupil(pupil_positions):
        max_pos = None
        pos_counts = [[0, []] for _ in range(45)]
        for pos in pupil_positions:
            x = pos[0]
            n = x // 20
            pos_counts[n][1].append(pos)
            pos_counts[n][0] += 1
            if max_pos is None or pos_counts[n][0] > max_pos[0]:
                max_pos = pos_counts[n]

        pupil_positions = max_pos[1]
        max_pos = None
        pos_counts = [[0, []] for _ in range(80)]
        for pos in pupil_positions:
            y = pos[1]
            n = y // 20
            print(y, n)
            pos_counts[n][1].append(pos)
            pos_counts[n][0] += 1
            if max_pos is None or pos_counts[n][0] > max_pos[0]:
                max_pos = pos_counts[n]

        pupil_centered = []
        if max_pos is not None:
            pupil_centered.append(mean([elem[0] for elem in max_pos[1]]))
            pupil_centered.append(mean([elem[1] for elem in max_pos[1]]))
        return pupil_centered


    def get_pupil_movement(reye, pupil_position, pupil_centered):
        print(pupil_position)
        move_left, move_right, move_up, move_down = False, False, False, False
        if reye is not None and pupil_position is not None:
            shiftbox_size = [reye.shiftbox['maxx'] - reye.shiftbox['minx'],
                             reye.shiftbox['maxy'] - reye.shiftbox['miny']]
            print('eye size: ' + str(shiftbox_size))
            x = pupil_position[0]
            y = pupil_position[1]
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
        print(pupil_position)
        return move_left, move_right, move_up, move_down


    def move_cursor(move_left, move_right, move_up, move_down, cursor_pos):
        radius=Control.radius
        MOVE_STEP=Control.MOVE_STEP
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
        pag.moveTo(cursor_pos)
        return cursor_pos
    def proc_control(self,detector):
            radius=Control.radius
            self.move_mode_open=self.move_mode_open  != detector.leye_winked()
            if detector.leye_winked():
                print("MODE CHANGED")
            if not self.move_mode_open:
                return
            if detector.center is None:
                return
            if detector.reye_winked():
                print("CLICK")
                pag.click()
            if detector.calibration_frame_count < 25:
                print(detector.center)
                detector.pupil_positions_MTARNOW.append(detector.center)
                detector.calibration_frame_count += 1

            if detector.calibration_frame_count == 25:
                detector.pupil_centered = Control.calibrate_pupil(detector.pupil_positions_MTARNOW)
                detector.calibration_frame_count += 1

            if detector.calibration_frame_count > 25:
                move_left, move_right, move_up, move_down = Control.get_pupil_movement(detector.reye, detector.center, detector.pupil_centered)
                detector.cursor_pos = Control.move_cursor(move_left, move_right, move_up, move_down, detector.cursor_pos)

#           sshot = cv2.imread('idylla.jpg', 0)
#           sshot = cv2.cvtColor(np.array(sshot), cv2.COLOR_GRAY2BGR)
#           cv2.circle(sshot, detector.cursor_pos, radius, (0, 0, 255), 5)
#           cv2.imshow("Screenshot", sshot)
#           cv2.waitKey(1)
            if detector.calibration_frame_count > 25:
                print(detector.pupil_centered)
                cv2.circle(detector.frame, (int(detector.pupil_centered[0]), int(detector.pupil_centered[1])), 2, (0, 0, 255), 2)
