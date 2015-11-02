import numpy as np
import cv2

cap = cv2.VideoCapture("output.avi")
print 'camera connected'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        continue

    # Our operations on the frame come here
    gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
