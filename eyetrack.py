import cv2
import numpy as np
import pylab as plt
import matplotlib.cm as cm
import scipy.ndimage.filters as filt
import pyqtgraph as pg

import scipy.stats as st
from scipy.ndimage.interpolation import shift
import numpy as np
from scipy.misc import imresize
from scipy.optimize import fmin, minimize, fmin_cg


film = cv2.VideoCapture("mouseeyetracking.avi")

# skip first frames (bad)
for i in range(100):
    ret, frame = film.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print ret
print film.isOpened()
print frame.shape

drawingimg= np.zeros_like(frame,dtype=np.uint8)
ix,iy = -1,-1
radius = 0
draw=False

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,radius,drawingimg,draw
    if event == cv2.EVENT_LBUTTONDOWN:
        draw=True
        ix,iy=x,y
        cv2.circle(drawingimg,(ix,iy),1,255,-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw==True:
            drawingimg= np.zeros_like(frame,dtype=np.uint8)
            cv2.circle(drawingimg,(ix,iy),1,255,-1)
            radius = int(np.sqrt((y-iy)*(y-iy) + (x-ix)*(x-ix)))
            cv2.circle(drawingimg,(ix,iy),radius,255,1)

    elif event == cv2.EVENT_LBUTTONUP:
        draw=False
        radius = int(np.sqrt((y-iy)*(y-iy) + (x-ix)*(x-ix)))
        cv2.circle(drawingimg,(ix,iy),radius,255,1)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


while(True):
    img = cv2.add(frame,drawingimg)
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


################################################################################
################################################################################

def resizeROI(frame,ix,iy,radius):
    # Crop the frame in a rectangle around the ROI
    ny,nx = frame.shape
    ymin = np.max([0,iy-radius])
    xmin = np.max([0,ix-radius])
    ymax = np.min([ny,iy+radius])
    xmax = np.min([nx,ix+radius])

    # eye is the ROI in rectangle
    eye = frame[ymin:ymax,xmin:xmax]
    x = ix-xmin
    y = iy-ymin
    return eye

def preprocess(img, mini = 30, maxi = 90, drift=50):
    # Preprocess
    linx = np.linspace(0,drift*0.01*nx,nx)
    liny = np.linspace(0,1,ny)
    xv, yv = np.meshgrid(linx,liny)
    xv = np.array(xv, np.uint8)
    img = cv2.addWeighted(img,1,xv,1,0)

    img = np.minimum(img,maxi)
    img = np.maximum(img,mini)
    img = cv2.equalizeHist(img)

    img = cv2.GaussianBlur(img,(5,5),0)
    return img



def circularmask(size, pos, r):
    x,y = pos
    nx,ny = size
    mx,my = np.ogrid[-x:(nx-x), -y:(ny-y)]
    mask = np.sqrt(mx*mx + my*my)
    mask = np.maximum(mask, r)
    mask = np.minimum(mask, r+1)
    mask -= np.min(mask)
    mask *= 1.0/np.max(mask)
    mask = mask[0:nx,0:ny]
    return 1-mask


def energycalc(param, img, Ith, alpha=0.5):
    x,y,r = param
    nx,ny = img.shape
    mask = circularmask((nx,ny), (x,y), r)
    eyein = mask*img
    energyin = 2*(np.sum(eyein)-Ith*np.sum(mask))

    mask *= 255
    mask = np.array(mask,dtype=np.uint8)
    maskcontour = cv2.Canny(mask,100,200)
    eyecontour = cv2.Canny(eye2,100,200)
    energyborder = -(np.sum(eyecontour*maskcontour)/np.sum(maskcontour))

    energy = alpha*energyin + (1-alpha)*energyborder
    return energy


eye = resizeROI(frame,ix,iy,radius)
nx,ny=eye.shape
fit=(ny/2,nx/2,7)




def nothing(x):
    pass

# Create windows
cv2.namedWindow('image')

# create trackbars
cv2.createTrackbar('mini','image',0,255,nothing)
cv2.createTrackbar('maxi','image',0,255,nothing)
cv2.createTrackbar('drift','image',0,100,nothing)
cv2.createTrackbar('alpha','image',0,100,nothing)
switch = '0 : Fixed \n1 : Flexible'
cv2.createTrackbar(switch, 'image',0,1,nothing)
mini = 10
# TODO: set the parameters right depending on the size of the ROI
maxi = 90
drift = 75
alpha = 50
cv2.setTrackbarPos('mini','image',mini)
cv2.setTrackbarPos('maxi','image',maxi)
cv2.setTrackbarPos('drift','image',drift)
cv2.setTrackbarPos('alpha','image',alpha)

xshift = []
yshift = []
rshift = []


# TODO: drift dependant of the size of the image
# TODO: optimization parameters
# TODO: allow only small variation once flexible mode in ON


while(film.isOpened()):
    ret, frame = film.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = resizeROI(frame,ix,iy,radius)
    eye2 = preprocess(eye, mini=mini, maxi=maxi, drift=drift)

    s = cv2.getTrackbarPos(switch,'image')
    if s == True:
        fit=fmin(energycalc, fit, (eye2, 200, alpha/100.), disp= False)
    else :
        fit=fmin(energycalc, (nx/2,ny/2,10), (eye2, 200, alpha/100.), disp= False)

    fit = np.maximum(fit,0)
    fit = np.minimum(fit,np.max([nx,ny]))

    img = eye2
    scale = 2
    img=imresize(img,(nx*scale,ny*scale))
    circle=fit*scale
    cv2.circle(img,(int(circle[1]),int(circle[0])),int(circle[2]),255,1)
    cv2.imshow('image',img)

    # eye3 = cv2.Canny(eye2,100,200)
    img2 = eye
    scale = 4
    img2 = imresize(img2,(nx*scale,ny*scale))
    circle = fit*scale
    cv2.circle(img2,(int(circle[1]),int(circle[0])),int(circle[2]),255,1)
    cv2.imshow('image2',img2)

    xshift.append(fit[1])
    yshift.append(fit[0])
    rshift.append(fit[2])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    mini = cv2.getTrackbarPos('mini','image')
    maxi = cv2.getTrackbarPos('maxi','image')
    drift = cv2.getTrackbarPos('drift','image')
    alpha = cv2.getTrackbarPos('alpha','image')

# When everything done, release the capture
film.release()
cv2.destroyAllWindows()


plt.plot(xshift)
plt.plot(yshift)
plt.plot(rshift)
plt.legend()
plt.show()
