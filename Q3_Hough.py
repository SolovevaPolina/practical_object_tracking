import cv2
import sys
import tracking_utils as utils

GRADIENT_THRESHOLD = 30

win_manager = utils.WindowManager()
cap, GRADIENT_THRESHOLD = utils.setup_video_and_args(default_thresh=GRADIENT_THRESHOLD)
video_filename = sys.argv[1]

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

    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    norm_angle = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    #Q3: threshold on the gradient magnitude to mask pixels
    _, mask = cv2.threshold(norm_magnitude, GRADIENT_THRESHOLD, 255, cv2.THRESH_BINARY)

    #Q3: sequence of orientations, where the masked pixels appear in red
    selected_orientations = cv2.cvtColor(norm_angle, cv2.COLOR_GRAY2BGR)
    selected_orientations[mask == 0] = [0, 0, 255]

    win_manager.show('Original', frame)
    win_manager.show('Gradient norm', norm_magnitude)
    win_manager.show('Gradient orientation', norm_angle)
    win_manager.show('Selected orientations', selected_orientations)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
    elif key == ord('s'):
        win_manager.save_composite_image("Q3", video_filename, cpt)
    cpt += 1

cap.release()
cv2.destroyAllWindows()