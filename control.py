from model.py import Retina_detector
def move_coursor(detector):
    move_left, move_right, move_up, move_down = False, False, False, False
        if len(detector.pupil_positions)<25:
            if reye is not None and reye.pupil_position is not None:
                shiftbox_size = [detector.reye.shiftbox['maxx'] - detector.reye.shiftbox['minx'],
                                 detector.reye.shiftbox['maxy'] - detector.reye.shiftbox['miny']]
        if len(detector.pupil_positions)>25
            if reye is not None and reye.pupil_position is not None:
                shiftbox_size = [reye.shiftbox['maxx'] - reye.shiftbox['minx'],
                                 reye.shiftbox['maxy'] - reye.shiftbox['miny']]
                print('shiftbox size: ' + str(shiftbox_size[0]), str(shiftbox_size[1]))
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

        print(move_left, move_right, move_up, move_down)
        # cursorPos=transPoint(reye.calc_darkest_point(),reye.scope,sshot.shape[:2],(1,1))
        if move_left:
            if cursorPos[0] > MOVE_STEP:
                cursorPos = (cursorPos[0] - MOVE_STEP, cursorPos[1])
            else:
                cursorPos = (0, cursorPos[1])
        elif move_right:
            if cursorPos[0] < 1200 - MOVE_STEP:
                cursorPos = (cursorPos[0] + MOVE_STEP, cursorPos[1])
            else:
                cursorPos = (1200, cursorPos[1])
        if move_up:
            if cursorPos[1] > MOVE_STEP:
                cursorPos = (cursorPos[0], cursorPos[1] - MOVE_STEP)
            else:
                cursorPos = (cursorPos[0], 0)
        elif move_down:
            if cursorPos[0] < 1176 - MOVE_STEP:
                cursorPos = (cursorPos[0], cursorPos[1] + MOVE_STEP)
            else:
                cursorPos = (cursorPos[0], 1176)


