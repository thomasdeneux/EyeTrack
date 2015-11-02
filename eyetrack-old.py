import numpy as np
import cv2

#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture('mouseeyetracking.avi')
print 'camera connected'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'i420')
xsize, ysize = cap.get(3), cap.get(4)
out = cv2.VideoWriter('output3.avi',fourcc, 60.0, (int(xsize),int(ysize)))

print(xsize,ysize)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        continue

    # Our operations on the frame come here
    gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    # Save the resulting frame in the VideoWriter
    out.write(gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
