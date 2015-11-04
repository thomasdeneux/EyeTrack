# Sub-functions involved in realtime pupil tracking

import cv2
import pylab as plt
from time import clock

import numpy as np
from scipy.optimize import fmin, minimize, fmin_cg


################################################################################
# Select ROI: mouse callback function
################################################################################

def select_roi(frame):
    roi = {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
    state = ["point1"]

    def draw_rect(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi['x1'], roi['y1'] = x, y
            state[0] = "point2"
        elif event == cv2.EVENT_LBUTTONUP:
            state[0] = "done"

        if state[0] != "point2":
            return

        roi['x2'], roi['y2'] = x, y
        img = frame.copy()
        cv2.rectangle(img, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), 255, 1)
        cv2.imshow('roi', img)

    cv2.namedWindow('roi')
    cv2.setMouseCallback('roi', draw_rect)
    cv2.imshow('roi', frame)

    while True:
        cv2.waitKey(20)  # this line allows other event to process
        if state[0] == "done":
            cv2.destroyWindow("roi")
            break

    roi['x1'], roi['x2'] = sorted((roi['x1'], roi['x2']))
    roi['y1'], roi['y2'] = sorted((roi['y1'], roi['y2']))
    return roi


################################################################################
################################################################################

def resize_roi(frame,roi):
    eye = frame[roi['y1']:roi['y2'],roi['x1']:roi['x2']]
    return eye


class Tracker:
    def __init__(self, eye):

        self.first = True

        # size info
        self.ny, self.nx = eye.shape

        # Tracking parameters
        self.mini = 0
        self.threshold = 50
        self.maxi = 70
        self.xdrift, self.xdriftmax = 30., 100.
        self.maxcontour = 300
        self.alpha = .5
        self.dotrack = True

        # Init graphics
        def nothing(x):
            pass
        # (create windows)
        cv2.namedWindow('preproc')  #, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('controls') #, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # (create trackbars)
        cv2.createTrackbar('mini', 'controls', self.mini, 100, nothing)
        cv2.createTrackbar('threshold', 'controls', self.threshold, 100, nothing)
        cv2.createTrackbar('maxi', 'controls', self.maxi, 100, nothing)
        cv2.createTrackbar('xdrift', 'controls', int((self.xdrift/self.xdriftmax + 1)*50), 100, nothing)
        cv2.createTrackbar('maxcontour', 'controls', self.maxcontour, 800, nothing)
        cv2.createTrackbar('alpha', 'controls', int(self.alpha*100), 100, nothing)
        cv2.createTrackbar('dotrack', 'controls', self.dotrack, 1, nothing)

        # TODO: drift dependant of the size of the image
        # TODO: optimization parameters
        # TODO: allow only small variation once flexible mode in ON

        # Tracking results
        self.fit = (self.nx/2, self.ny/2, 7)
        self.dosave = False
        self.xshift = []
        self.yshift = []
        self.rshift = []

    def startsave(self):
        self.dosave = True
        self.xshift = []
        self.yshift = []
        self.rshift = []

    def preprocess(self, eye):
        # correct for drift
        linx = np.linspace(-self.xdrift, self.xdrift, self.nx)
        #linx = np.linspace(0, self.xdrift, self.nx)

        liny = np.linspace(0, 1, self.ny)
        xv, yv = np.meshgrid(linx, liny)
        xv = np.array(xv, np.uint8)
        img = eye + xv
        img = cv2.GaussianBlur(img,(5,5),0)

        # clip image, smooth and center on threshold
        eye2 = np.clip(img, self.mini, self.maxi)
        eye2 = eye2.astype(float) - self.threshold

        # edge detection
        ix = cv2.Scharr(img,cv2.CV_32F,1,0)
        iy = cv2.Scharr(img,cv2.CV_32F,0,1)
        eyecontour = np.sqrt(ix**2 + iy**2)
        eyecontour = np.minimum(eyecontour, np.array(self.maxcontour))

        return eye2, eyecontour

    def circularmask(self, x, y, r):
        my, mx = np.ogrid[0:self.ny, 0:self.nx]
        mx, my = mx-x, my-y
        mask = np.sqrt(mx*mx + my*my)
        mask = np.maximum(mask, r)
        mask = np.minimum(mask, r+1)
        mask -= np.min(mask)
        mask *= 1.0/np.max(mask)
        return 1-mask

    def energycalc(self, param, eye2, eyecontour, showimage=False):
        x, y, r = param
        mask = self.circularmask(x, y, r)
        eyein = mask*eye2
        energyin = np.sum(eyein)

        mask *= 255
        mask = np.array(mask,dtype=np.uint8)
        maskcontour = cv2.Canny(mask,100,200)
        energyborder = -(np.sum(eyecontour*maskcontour)/np.sum(maskcontour))

        if showimage:
            scale = 4

            # eye2
            img1 = eye2/(2*max(eye2.max(),-eye2.min())) + .5  # bring eye2 between 0 and 1
            img1 = img1*.6 + np.greater(img1,.5)*.4  # move values away from .5
            img1 = (img1*255).astype('uint8')
            # img1 = imresize(img1, (self.ny*scale, self.nx*scale))
            img1 = np.repeat(np.repeat(img1, scale, axis=0), scale, axis=1)
            cv2.circle(img1, (int(x*scale), int(y*scale)), int(r*scale), 128, 1)

            # eyecontour
            img2 = (eyecontour*(255./self.maxcontour)).astype('uint8')
            # img2 = imresize(img2, (self.ny*scale, self.nx*scale))
            img2 = np.repeat(np.repeat(img2, scale, axis=0), scale, axis=1)
            cv2.circle(img2, (int(x*scale), int(y*scale)), int(r*scale), 0, 1)

            # display both
            img = np.hstack((img1, img2))
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            cv2.imshow('preproc', img)

        energy = (1-self.alpha)*energyin + self.alpha*energyborder
        return energy

    def track(self, eye):
        self.threshold = cv2.getTrackbarPos('threshold', 'controls')
        self.mini = min(cv2.getTrackbarPos('mini', 'controls'),self.threshold-1)
        self.maxi = max(cv2.getTrackbarPos('maxi', 'controls'),self.threshold+1)
        self.xdrift = (cv2.getTrackbarPos('xdrift', 'controls')/50.-1)*self.xdriftmax
        self.maxcontour = cv2.getTrackbarPos('maxcontour', 'controls')
        self.alpha = cv2.getTrackbarPos('alpha', 'controls')/100.
        self.dotrack = cv2.getTrackbarPos('dotrack', 'controls')

        eye2, eyecontour = self.preprocess(eye)

        # opt = dict(xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, disp=False)
        opt = dict(xtol=0.1, ftol=0.1, maxiter=10, maxfun=None, disp=True)
        if self.dotrack:
            self.fit = fmin(self.energycalc, self.fit, (eye2, eyecontour), **opt)
        else:
            self.fit = fmin(self.energycalc, (self.nx/2, self.ny/2, 10), (eye2, eyecontour), **opt)

        self.fit = np.maximum(self.fit, 0)
        self.fit = np.minimum(self.fit, np.max([self.nx, self.ny]))

        self.energycalc(self.fit, eye2, eyecontour, showimage=True)

        if self.dosave:
            self.xshift.append(self.fit[1])
            self.yshift.append(self.fit[0])
            self.rshift.append(self.fit[2])

    def summary(self):
        plt.plot(self.xshift)
        plt.plot(self.yshift)
        plt.plot(self.rshift)
        plt.legend()
        plt.show()
