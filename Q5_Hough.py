import numpy as np
import cv2
import sys
import tracking_utils as utils

#Q3: threshold on the gradient magnitude
GRADIENT_THRESHOLD = 80 
#Q4: bins for angle
NUM_ANGLE_BINS = 16

#Q5: (1) simple prediction strategy
USE_PREDICTION = True
SEARCH_WINDOW_SCALE = 2 # x object size

#Q5: (2) an update strategy of the model
USE_MODEL_UPDATE = True
LEARNING_RATE = 0.1 # 10% new, 90% old

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

#initial position for tracking
r_track = r
c_track = c

cpt = 1
while(1):
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape
    
    #Q5: (1) simple prediction strategy
    if USE_PREDICTION:
        search_h = h * SEARCH_WINDOW_SCALE # search width
        search_w = w * SEARCH_WINDOW_SCALE # search height
        
        search_r = max(0, r_track + h//2 - search_h//2) # top-left x
        search_c = max(0, c_track + w//2 - search_w//2) # top-left y

        #ensure search window is inside frame
        search_h = min(search_h, frame_width - search_r)
        search_w = min(search_w, frame_height - search_c)

        #crop search area
        gray_search = frame[search_c:search_c+search_w, search_r:search_r+search_h]
        if gray_search.size == 0: continue
        gray_search = cv2.cvtColor(gray_search, cv2.COLOR_BGR2GRAY)
        
        search_offset_r, search_offset_c = search_r, search_c
    else:
        #if no prediction, search the whole frame
        gray_search = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_offset_r, search_offset_c = 0, 0


    #Q3: the local orientation and the gradient magnitude (on search area)
    sobel_x = cv2.Sobel(gray_search, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_search, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

    #Q4: calculate the associated Hough transform (on search area)
    hough_space = np.zeros(gray_search.shape, dtype=np.float32)

    for y_coord in range(gray_search.shape[0]):
        for x_coord in range(gray_search.shape[1]):
            #Q3: threshold on the gradient magnitude to mask pixels
            if magnitude[y_coord, x_coord] > GRADIENT_THRESHOLD:
                angle_bin = int((angle[y_coord, x_coord] / 360.0) * NUM_ANGLE_BINS)
                
                if angle_bin in r_table:
                    for dx, dy in r_table[angle_bin]:
                        center_x, center_y = x_coord + dx, y_coord + dy
                        if 0 <= center_y < hough_space.shape[0] and 0 <= center_x < hough_space.shape[1]:
                            hough_space[center_y, center_x] += 1

    #Q4: calculate the straightforward tracking
    _, max_val, _, max_loc = cv2.minMaxLoc(hough_space) # (x, y) relative to search area
    
    #Convert local (search area) coordinates to global (frame) coordinates
    center_x_abs = max_loc[0] + search_offset_r
    center_y_abs = max_loc[1] + search_offset_c
    
    r_track = center_x_abs - h//2 # h is width
    c_track = center_y_abs - w//2 # w is height
    
    #Q5: (2) an update strategy of the model
    if USE_MODEL_UPDATE and max_val > 0:
        #get new roi
        new_roi = frame[c_track:c_track+w, r_track:r_track+h]
        #check if roi is valid
        if new_roi.shape[0] == w and new_roi.shape[1] == h:
            new_roi_gray = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
            new_r_table = utils.build_r_table(new_roi_gray, w, h, GRADIENT_THRESHOLD, NUM_ANGLE_BINS)
            
            #Simple weighted update
            for bin_key in r_table: # iterate old keys to avoid growing table
                old_vectors = r_table.get(bin_key, [])
                new_vectors = new_r_table.get(bin_key, []) #get new vectors for same bin
                
                num_new = int(len(new_vectors) * LEARNING_RATE)
                num_old = len(old_vectors) - num_new
                if num_old < 0: num_old = 0

                #replace part of old list with part of new list
                r_table[bin_key] = old_vectors[:num_old] + new_vectors[:num_new]
    
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
        win_manager.save_composite_image("Q5", video_filename, cpt)
    cpt += 1

cap.release()
cv2.destroyAllWindows()