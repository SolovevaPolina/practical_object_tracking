import numpy as np
import cv2
import sys
import tracking_utils as utils


#Q3: threshold on the gradient magnitude
GRADIENT_THRESHOLD = 80
#Q4: bins for angle
NUM_ANGLE_BINS = 16

win_manager = utils.WindowManager()
cap, GRADIENT_THRESHOLD = utils.setup_video_and_args(default_thresh=GRADIENT_THRESHOLD)
video_filename = sys.argv[1]

ret,frame = cap.read()

r, c, h, w = utils.select_roi(frame.copy())
win_manager.set_static_roi_frame(frame, r, c, h, w)
 
#create model from ROI
roi = frame[c:c+w, r:r+h] # c=y, r=x, w=height, h=width
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
r_table = utils.build_r_table(roi_gray, w, h, GRADIENT_THRESHOLD, NUM_ANGLE_BINS) # w=height, h=width

cpt = 1
while(1):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Q3: the local orientation and the gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

    #Q4: calculate the associated Hough transform
    hough_space = np.zeros(gray.shape, dtype=np.float32)

    for y_coord in range(gray.shape[0]):
        for x_coord in range(gray.shape[1]):
            #Q3: threshold on the gradient magnitude to mask pixels
            if magnitude[y_coord, x_coord] > GRADIENT_THRESHOLD:
                angle_bin = int((angle[y_coord, x_coord] / 360.0) * NUM_ANGLE_BINS)
                
                if angle_bin in r_table:
                    for dx, dy in r_table[angle_bin]:
                        center_x, center_y = x_coord + dx, y_coord + dy
                        if 0 <= center_y < hough_space.shape[0] and 0 <= center_x < hough_space.shape[1]:
                            hough_space[center_y, center_x] += 1

    #Q4: calculate the straightforward tracking
    _, _, _, max_loc = cv2.minMaxLoc(hough_space) # (x, y)
    
    r_track = max_loc[0] - h//2 # h is width
    c_track = max_loc[1] - w//2 # w is height
    
    frame_tracked = frame.copy()
    cv2.rectangle(frame_tracked, (r_track, c_track), (r_track+h, c_track+w), (255, 0, 0), 2)

    #Q4: showing Hough Transform
    hough_display = cv2.normalize(hough_space, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    win_manager.show('Sequence', frame_tracked)
    win_manager.show('Hough Transform H(x)', hough_display)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
    elif key == ord('s'):
        win_manager.save_composite_image("Q4", video_filename, cpt)
    cpt += 1

cap.release()
cv2.destroyAllWindows()