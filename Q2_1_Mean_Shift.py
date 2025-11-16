import numpy as np
import cv2
import sys
import tracking_utils as utils

win_manager = utils.WindowManager()
cap, _ = utils.setup_video_and_args()
video_filename = sys.argv[1]

#take first frame of the video
ret,frame = cap.read()

#select ROI
r, c, h, w = utils.select_roi(frame.copy())
win_manager.set_static_roi_frame(frame, r, c, h, w)
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #Q2.1: displaying the sequences of hue images
        hue_channel = hsv[:,:,0]
        win_manager.show('Hue Channel', hue_channel)
        # Backproject the model histogram roi_hist onto the 
        # current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))

		#Q2.1: weight images corresponding to the back-projection of the hue histogram
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        win_manager.show('Back Projection', dst)

        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a blue rectangle on the current image
        r,c,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        win_manager.show('Sequence',frame_tracked)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            win_manager.save_composite_image("Q2.1", video_filename, cpt)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()