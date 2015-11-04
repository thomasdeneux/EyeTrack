from PyQt4.QtGui import *
from PyQt4.QtCore import *
import cv2
import numpy as np
from scipy.io import savemat

import os.path
from time import clock, sleep
from math import floor

import tracking, audiotrigger


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

        # (camera grabbing)
        self.film, self.frame = [], []
        self.status = ''
        self.t0, self.nprocessed, self.fps = 0, 0, 0
        self.idle = False
        self.acq_timer = []

        # (acquisition)
        self.filename = ''
        self.acqlen = 3
        self.numacq = 1
        self.acqstate = 'off'  # 'off', 'wait' or 'on'
        self.curacq = 0
        self.acqstart = 0

        # (eye)
        self.roi, self.roisave, self.eye = [], [], []
        self.nxsave, self.nysave = 0, 0

        # (audio trigger)
        self.audio = audiotrigger.AudioTrigger()

        # (saving)
        self.curname = ''
        self.out, self.timevector = [], []

        # (eye tracking)
        self.tracker = []

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
        b.textEdited.connect(lambda: self.select_filename('edit'))
        self.layout.addWidget(b, row, 2)
        self.buttons['filename'] = b

        # Acquisition length
        row += 1
        b = QLabel("length (s)")
        self.layout.addWidget(b, row, 1)

        b = QLineEdit(str(self.acqlen))
        b.textChanged.connect(lambda: self.readinput('acqlen'))
        self.layout.addWidget(b, row, 2)
        self.buttons['acqlen'] = b

        # Acquisition number
        row += 1
        b = QLabel("# repeat (0=Inf.)")
        self.layout.addWidget(b, row, 1)

        b = QLineEdit(str(self.numacq))
        b.textChanged.connect(lambda: self.readinput('numacq'))
        self.layout.addWidget(b, row, 2)
        self.buttons['numacq'] = b

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

    def readinput(self, field):
        st = self.buttons[field].text()
        num = int(st)
        self.buttons[field].setText(str(num))
        if field == 'numacq':
            self.numacq = num
        elif field == 'acqlen':
            self.acqlen = num

    def startstop(self):
        b = self.buttons['startstop']
        if b.isChecked() and (not self.roi_selected or self.filename == ""):
            b.setChecked(False)
            QMessageBox.warning(None, "EYETRACK", "Select ROI and file name first")
            return
        self.running = b.isChecked()
        if self.running:
            b.setStyleSheet("color: red; font-weight: bold")
        else:
            b.setStyleSheet("")

    def toggletrack(self):
        b = self.buttons['track']
        if b.isChecked() and not self.roi_selected:
            b.setChecked(False)
            QMessageBox.warning(None, "EYETRACK", "Select ROI first")
            return
        self.dotrack = b.isChecked()
        if self.dotrack:
            b.setStyleSheet("color: blue; font-weight: bold")
            self.tracker = tracking.Tracker(self.eye)
        else:
            b.setStyleSheet("")
            self.tracker = []
            cv2.destroyWindow('preproc')
            cv2.destroyWindow('controls')

    def select_filename(self, f=None):
        if f is None:
            f = str(QFileDialog.getSaveFileName())
        elif f == 'edit':
            f = str(self.buttons['filename'].text())
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
            self.roisave = {'x1': 212, 'y1': 153, 'x2': 278, 'y2': 220}
            self.nxsave, self.nysave = self.roisave['x2'] - self.roisave['x1'], self.roisave['y2'] - self.roisave['y1']
            self.roi_selected = True
            self.select_filename("C:/Users/THomas/PycharmProjects/EyeTrack/data")

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

    # SELECT ROI
    def select_roi(self):
        self.roi = tracking.select_roi(self.frame)

        # ROI for saving must be at least 65x65 otherwise buffer might be to small
        roisave = self.roi.copy()
        nx, ny = roisave['x2'] - roisave['x1'], roisave['y2'] - roisave['y1']
        if nx < 64:
            roisave['x1'] -= floor((64 - nx) / 2)
            nx = 64
            roisave['x2'] = roisave['x1'] + nx
        if ny < 64:
            roisave['y1'] -= floor((64 - ny) / 2)
            ny = 64
            roisave['y2'] = roisave['y1'] + ny

        self.roisave = roisave
        self.nxsave, self.nysave = nx, ny
        self.roi_selected = True

        # Re-init tracker if necessary
        if self.dotrack:
            self.tracker = tracking.Tracker(self.eye)

    # MAIN LOOP
    def process_one_frame(self):

        ret, frame3 = self.film.read()

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
                ret, frame3 = self.film.read()
        else:
            self.idle = False

        # follow speed of processing frames
        t = clock()
        self.nprocessed += 1
        if t > self.t0 + 1:
            self.fps = self.nprocessed / (t - self.t0)
            self.t0 = t
            self.nprocessed = 0

        # make frame single channel
        if frame3.ndim == 3:
            self.frame = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        else:
            self.frame = frame3

        # display frame
        cv2.imshow('movie', self.frame)

        # change between acquisition states
        # note that acquisition will never start as long as ROI and filename are not selected
        if self.running and self.acqstate == 'off':
            self.acqstate = 'wait'
            self.curacq += 1
            self.status = 'waiting for trigger ' + str(self.curacq) + '/' + str(self.numacq)

            # open audio stream to detect trigger
            self.audio.load()

            # open movie for writing
            fourcc = cv2.VideoWriter_fourcc(*'i420')
            self.curname = self.filename + '_%.3i' % self.curacq
            self.out = cv2.VideoWriter(self.curname + '.avi', fourcc, 60.0, (self.nysave, self.nxsave))

            # prepare for saving time vector and tracking results
            self.timevector = []

        if self.acqstate == 'wait':
            if self.audio.check():  # once trigger will be detected, audio stream will be automatically closed
                self.acqstate = 'on'
                self.acqstart = clock()
                self.status = 'ACQUISITION ' + str(self.curacq) + '/' + str(self.numacq)
                if self.dotrack:
                    self.tracker.startsave()
        elif self.acqstate == 'on' and (clock() - self.acqstart) > self.acqlen:
            self.acqstate = 'off'
            self.status = ''
            if self.dotrack:
                self.tracker.dosave = False

            # close output movie, save time vector and tracking results
            self.out.release()
            savedata = {'timevector': self.timevector}
            if self.dotrack:
                savedata['xshift'] = self.tracker.xshift
                savedata['yshift'] = self.tracker.yshift
                savedata['rshift'] = self.tracker.rshift
            savemat(self.curname + '.mat', savedata)
            if self.dotrack:
                pass

            # finished repetition, or user interrupted them?
            if self.curacq == self.numacq or not self.running:
                self.buttons['startstop'].setChecked(False)
                self.curacq = 0

        # save
        if self.acqstate == 'on':
            eyesave = tracking.resize_roi(frame3, self.roisave)
            self.timevector.append(clock() - self.acqstart)
            self.out.write(eyesave)

        # track
        if self.dotrack:
            self.tracker.track(self.eye)

        # display eye
        if self.roi_selected:
            self.eye = tracking.resize_roi(self.frame, self.roi)
            scale = 4
            img = np.repeat(np.repeat(self.eye, scale, axis=0), scale, axis=1)
            if self.dotrack:
                circle = self.tracker.fit*scale
                cv2.circle(img, (int(circle[0]),int(circle[1])),int(circle[2]),255,1)
            cv2.imshow('eye', img)

        # update status
        strfps = " (%.1ffps)" % self.fps
        self.statusbar.setText(self.status + strfps)


def launch_menu():
    app = QApplication([])
    b = EyetrackMenu()
    b.show()
    app.exec_()


if __name__ == "__main__":
    launch_menu()
