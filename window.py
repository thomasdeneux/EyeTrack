from PyQt4.QtGui import *
from PyQt4.QtCore import *
import cv2
import numpy as np

import os.path
from time import clock, sleep

import tracking


class EyetrackMenu(QDialog):
    DOREC = False

    def __init__(self):
        QDialog.__init__(self, None)
        self.setWindowTitle("EYETRACK")

        # Properties
        # (state)
        self.running = False
        self.dotrack = False
        self.roi_selected = False

        # (acquisition)
        self.filename = ''
        self.acqlen = 3
        self.numacq = 0

        # (camera grabbing)
        self.film, self.frame = [], []
        self.status = ''
        self.t0, self.nprocessed, self.fps = 0, 0, 0
        self.idle = False

        # (eye)
        self.roi, self.eye = [], []

        # Init graphics
        self.init_graphics()

        # Init camera and grab continuously
        self.init_camera()

    def __del__(self):
        print 'cleaning up'
        cv2.destroyAllWindows()
        self.acq_timer.stop()
        self.film.release()

    # GRAPHICS
    def init_graphics(self):
        # init window
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        row = 0
        self.buttons = {}

        # ROI and pupil tracking
        row += 1
        b = QPushButton("Select ROI")
        b.clicked.connect(self.select_roi)
        self.layout.addWidget(b, row, 1)

        b = QPushButton("TRACK")
        b.setCheckable(True)
        b.toggled.connect(self.toggletrack)
        self.layout.addWidget(b, row, 2)
        self.buttons['track'] = b

        # File name
        row += 1
        b = QPushButton("file name:")
        b.clicked.connect(self.select_filename)
        self.layout.addWidget(b, row, 1)

        b = QLineEdit("")
        self.layout.addWidget(b, row, 2)
        self.buttons['filename'] = b

        # Acquisition length
        row += 1
        b = QLabel("length (s)")
        self.layout.addWidget(b, row, 1)

        b = QLineEdit(str(self.acqlen))
        self.layout.addWidget(b, row, 2)

        # Start
        row += 1
        b = QPushButton("RUN")
        b.setCheckable(True)
        b.toggled.connect(self.startstop)
        self.layout.addWidget(b, row, 1)
        self.buttons['startstop'] = b

        # Status
        row += 1
        self.statusbar = QLabel("")
        self.layout.addWidget(self.statusbar, row, 1, row, 2)

    def startstop(self):
        b = self.buttons['startstop']
        self.running = b.isChecked()
        if self.running:
            if (not self.roi_selected) or (self.filename == ""):
                QMessageBox.warning(None,"EYETRACK","Select ROI and file name first")
            b.setStyleSheet("color: red; font-weight: bold")
        else:
            b.setStyleSheet("")

    def toggletrack(self):
        b = self.buttons['track']
        self.dotrack = b.isChecked()
        if self.dotrack:
            b.setStyleSheet("color: blue; font-weight: bold")
        else:
            b.setStyleSheet("")

    def select_filename(self):
        f = str(QFileDialog.getSaveFileName())
        self.filename, ext = os.path.splitext(f)
        self.buttons['filename'].setText(self.filename)

    # CAMERA
    def init_camera(self):
        # start camera
        if self.DOREC:
            self.film = cv2.VideoCapture(2)
        else:
            self.film = cv2.VideoCapture("mouseeyetracking.avi")
            # spare time for setting ROI and file name
            self.roi = {'x1': 222, 'y1': 163, 'x2': 268, 'y2': 210}
            self.roi_selected = True
            self.filename = "C:/Users/THomas/PycharmProjects/EyeTrack/data"

        # grab one frame
        ret, self.frame = self.film.read()
        while self.frame is None:
            ret, self.frame = film.read()
        print ret
        print self.film.isOpened()
        print self.frame.shape
        if self.frame.ndim == 3:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Main loop: grab and process frames continuously
        self.t0 = clock()
        self.acq_timer = QTimer()
        self.acq_timer.timeout.connect(self.process_one_frame)
        self.acq_timer.start()

    # ACTION
    def select_roi(self):
        self.roi = tracking.select_roi(self.frame)
        self.roi_selected = True

    # MAIN LOOP
    def process_one_frame(self):

        ret, self.frame = self.film.read()

        # no frame -> indicate that there is some "idle" time
        if not ret:
            if self.DOREC:
                if not self.idle:
                    self.idle = True
                    print 'some idle time'
                sleep(.001)
                return
            else:
                self.film.release()
                self.film = cv2.VideoCapture("mouseeyetracking.avi")
                ret, self.frame = self.film.read()
        else:
            self.idle = False

        # follow speed of processing frames
        t = clock()
        self.nprocessed += 1
        if t>self.t0+1:
            self.fps = self.nprocessed/(t-self.t0)
            self.t0 = t
            self.nprocessed = 0

        # make frame single channel
        if self.frame.ndim == 3:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # display frame
        cv2.imshow('movie', self.frame)

        # display eye
        if self.roi_selected:
            self.eye = tracking.resize_roi(self.frame,self.roi)
            cv2.imshow('eye', np.repeat(np.repeat(self.eye, 4, axis=0), 4, axis=1))

        # update status
        strfps = " (%.1ffps)" % self.fps
        self.statusbar.setText(self.status+strfps)



def launch_menu():
    app = QApplication([])
    b = EyetrackMenu()
    b.show()
    app.exec_()


if __name__ == "__main__":
    launch_menu()

