import sys
from PyQt5 import QtCore, QtWidgets, Qt, QtGui
from PyQt5.QtWidgets import QMainWindow, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter, QFont, QBrush
import json
import time
import ast


class MyQThread(QtCore.QThread):
    # Signals to relay thread progress to the main GUI thread
    moveModeSignal = QtCore.pyqtSignal(bool)

    def run(self):
        while True:
            time.sleep(3)
            with open('cexch.pkl') as f:
                json_data = json.load(f)
            move_mode_open = json_data['move_mode_open']
            print(move_mode_open)
            self.moveModeSignal.emit(move_mode_open)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(1280, 720))
        self.setWindowTitle("Sentinel")
        self.alarm = False
        self.inform = False
        self.move_mode_open = False

        self.alarm_button = QPushButton('Zaalarmuj personel', self)
        self.inform_button = QPushButton('Powiadom personel', self)

        self.alarm_button.clicked.connect(self.on_alarm_click)
        self.inform_button.clicked.connect(self.on_inform_click)
        font = self.alarm_button.font()
        font.setPointSize(24)
        print(font.defaultFamily())
        self.alarm_button.setFont(font)
        self.alarm_button.setStyleSheet("QPushButton {background:#ffbfbc; color: #3d3c3c} QPushButton:hover {background:#ffa499}")

        self.inform_button.setFont(font)
        self.inform_button.setStyleSheet("QPushButton {background:#cceeff; color: #3d3c3c} QPushButton:hover {background: #9dd4f9}")

        self.alarm_button.resize(640, 560)
        self.alarm_button.move(0, 160)
        self.inform_button.resize(640, 560)
        self.inform_button.move(640, 160)

        # Initialize the thread
        self.myThread = MyQThread()
        if not self.myThread.isRunning():
            self.myThread.moveModeSignal.connect(self.handle_move_mode_signal)
            self.myThread.start()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(Qt.QPen(Qt.QColor('#3d3c3c')))
        painter.setFont(QFont("MS Shell Dlg 2", 16))

        if self.move_mode_open:
            painter.fillRect(Qt.QRectF(0, 0, 1280, 80), Qt.QColor('#d6ffdd'))
        else:
            painter.fillRect(Qt.QRectF(0, 0, 1280, 80), QtCore.Qt.white)

        if self.alarm or self.inform:
            self.alarm_button.resize(640, 560)
            self.alarm_button.move(0, 160)
            self.inform_button.resize(640, 560)
            self.inform_button.move(640, 160)
        else:
            self.alarm_button.resize(640, 640)
            self.alarm_button.move(0, 80)
            self.inform_button.resize(640, 640)
            self.inform_button.move(640, 80)

        if self.alarm:
            painter.fillRect(Qt.QRectF(0, 80, 1280, 80), Qt.QColor('#ffbfbc'))
            painter.drawText(Qt.QRectF(0, 80, 1280, 80), QtCore.Qt.AlignCenter, 'opiekun jest alarmowany')
        elif self.inform:
            painter.fillRect(Qt.QRectF(0, 80, 1280, 80), Qt.QColor('#cceeff'))
            painter.drawText(Qt.QRectF(0, 80, 1280, 80), QtCore.Qt.AlignCenter, 'opiekun jest powiadamiany')
        painter.drawText(Qt.QRectF(0, 0, 1280, 80), QtCore.Qt.AlignCenter,
                         'sterowanie kursorem jest ' + ('włączone' if self.move_mode_open else 'wyłączone'))
        painter.end()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            raise KeyboardInterrupt

    def on_alarm_click(self):
        print('Alarm')
        if not self.alarm:
            self.alarm = True
            self.inform = False
        else:
            self.alarm = False

        with open('alert_data.pkl', 'w') as f:
            f.truncate(0)
            alert_data = {'alarm': self.alarm, 'inform': self.inform}
            json.dump(alert_data, f)
        self.update()

    @QtCore.pyqtSlot(bool)
    def handle_move_mode_signal(self, e):
        self.move_mode_open = e
        self.update()

    def on_inform_click(self):
        print('Powiadomienie')
        if not self.inform:
            self.inform = True
            self.alarm = False
        else:
            self.inform = False

        with open('alert_data.pkl', 'w') as f:
            f.truncate(0)
            alert_data = {'alarm': self.alarm, 'inform': self.inform}
            json.dump(alert_data, f)
        self.update()


def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
