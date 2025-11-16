import cv2
import sys
import os
import numpy as np
from collections import defaultdict

_r, _c, _w, _h = 0, 0, 0, 0
_roi_defined = False

def define_ROI(event, x, y, flags, param):
	"""
	"""
	global _r, _c, _w, _h, _roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		_r, _c = x, y
		_roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		_h = abs(r2-_r) # h is width
		_w = abs(c2-_c) # w is height
		_r = min(_r,r2)
		_c = min(_c,c2)  
		_roi_defined = True

def setup_video_and_args(default_thresh=None):
	"""

	"""
	threshold = default_thresh
	
	if len(sys.argv) < 2:
		print("Add video name")
		print(f"Example: python {sys.argv[0]} Red_Ball.mp4")
		sys.exit()

	if len(sys.argv) > 2 and default_thresh is not None:
		try:
			threshold = int(sys.argv[2])
		except ValueError:
			print("Error: Threshold must be an integer")
			sys.exit()
	
	video_filename = sys.argv[1]
	video_path = os.path.join('Sequences', video_filename)
	cap = cv2.VideoCapture(video_path)

	if not cap.isOpened():
		print(f"Error opening video file '{video_path}'")
		sys.exit()
		
	if threshold is not None:
		print(f"Using gradient threshold: {threshold}")
		
	return cap, threshold

def select_roi(frame):
	"""
	take first frame of the video
    load the image, clone it, and setup the mouse callback function
	keep looping until the 'q' key is pressed
	draw a green rectangle around the region of interest
	"""
	global _r, _c, _w, _h, _roi_defined
	
	clone = frame.copy()
	cv2.namedWindow("First image")
	cv2.setMouseCallback("First image", define_ROI)

	while True:
		cv2.imshow("First image", frame)
		key = cv2.waitKey(1) & 0xFF
		if (_roi_defined):
			cv2.rectangle(frame, (_r,_c), (_r+_h,_c+_w), (0, 255, 0), 2)
		else:
			frame = clone.copy()
		if key == ord("q"):
			break
	
	cv2.destroyWindow("First image")
	return _r, _c, _h, _w



def build_r_table(roi_gray, w, h, gradient_threshold, num_angle_bins):
    """
    Q4: build a model of the initial object (R-Table)
    """
    #Q3: the local orientation and the gradient magnitude (for ROI)
    sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

    center_x, center_y = h // 2, w // 2 # h=width, w=height
    r_table = defaultdict(list)

    for y_coord in range(w): # iter over height (w)
        for x_coord in range(h): # iter over width (h)
            #Q3: threshold on the gradient magnitude
            if magnitude[y_coord, x_coord] > gradient_threshold:
                #quantize angle
                angle_bin = int((angle[y_coord, x_coord] / 360.0) * num_angle_bins)

                dx = center_x - x_coord
                dy = center_y - y_coord
                r_table[angle_bin].append((dx, dy))
    return r_table


# --- New class for managing windows ---

class WindowManager:
    """
    Manages cv2 windows and saves composite screenshots from all windows.
    """
    def __init__(self, max_row_width=3, save_directory="Figures"):
        self.windows = {} # Dictionary to store {name: image}
        self.max_row_width = max_row_width
        self.save_dir = save_directory # Directory to save images
        
        self.static_panel = None # This will store our pre-rendered ROI panel
        
        # Settings for titles
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_color = (0, 0, 0) # Black
        self.line_type = 1
        self.title_bar_height = 40
        self.std_height = 480 # The target height for all panels

    def set_static_roi_frame(self, frame, r, c, h, w):
        """
        Creates and stores a prepared panel of the initial ROI.
        This will be prepended to all saved composite images.
        """
        # Draw the green ROI box
        roi_frame = frame.copy()
        cv2.rectangle(roi_frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
        
        # Use _prepare_panel to create the styled panel ONCE
        self.static_panel = self._prepare_panel("Initial ROI (Static)", roi_frame)

    def show(self, window_name, image_data):
        """
        Displays an image and "registers" it for saving.
        Replaces cv2.imshow()
        """
        # Store the latest version of the image
        self.windows[window_name] = image_data
        # Show it as usual
        cv2.imshow(window_name, image_data)

    def _prepare_panel(self, title, image):
        """
        (Internal function)
        Creates a single "panel": image + title on top.
        Scales all images to std_height.
        """
        # 1. Convert Grayscale to BGR if needed
        if len(image.shape) == 2:
            panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            panel = image.copy()
            
        # 2. Scale to standard height, preserving aspect ratio
        orig_h, orig_w = panel.shape[:2]
        new_w = int((orig_w / orig_h) * self.std_height)
        new_h = self.std_height
        panel_resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. Create a white bar for the title
        title_bar = np.full((self.title_bar_height, new_w, 3), 255, dtype=np.uint8)
        
        # 4. Write the title text
        (text_w, text_h), _ = cv2.getTextSize(title, self.font, self.font_scale, self.line_type)
        text_x = (new_w - text_w) // 2 # Center text
        text_y = (self.title_bar_height + text_h) // 2
        cv2.putText(title_bar, title, (text_x, text_y), self.font, self.font_scale, self.font_color, self.line_type)

        # 5. Stack title bar and image
        final_panel = cv2.vconcat([title_bar, panel_resized])
        return final_panel

    def save_composite_image(self, q_name, video_name, frame_num):
        """
        Assembles all "registered" windows into a grid and saves the file.
        Dynamically chooses layout (2x2 for 4 images, 3xN for all others).
        """
        
        # 1. Create a list of prepared "panels"
        if self.static_panel is not None:
            prepared_panels = [self.static_panel]
        else:
            prepared_panels = []

        # --- MODIFIED SECTION ---
        # Add the dynamic (current) panels
        for title, image in self.windows.items():
            panel_title = title # Default title
            
            # Check if this is the 'Sequence' panel
            if title == "Sequence":
                panel_title = f"Sequence (Frame {frame_num})" # Create dynamic title
                
            prepared_panels.append(self._prepare_panel(panel_title, image))
        # --- END MODIFIED SECTION ---

        if not prepared_panels:
            print("WindowManager: No windows to save.")
            return

        num_windows = len(prepared_panels)

        # --- Dynamic Layout Logic ---
        if num_windows == 4:
            row_width = 2
        else:
            row_width = self.max_row_width
        # --- End of New Logic ---

        # 2. Group panels into rows
        rows = []
        for i in range(0, num_windows, row_width):
            row_panels = prepared_panels[i:i + row_width]
            rows.append(np.hstack(row_panels))
        
        # 3. Equalize row widths
        max_w = max(r.shape[1] for r in rows) # Find the widest row
        final_rows = []
        
        for row in rows:
            if row.shape[1] < max_w:
                # Add white padding to the right if row is shorter
                padding = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
                final_rows.append(np.hstack([row, padding]))
            else:
                final_rows.append(row)
                
        # 4. Stack all rows into one final image
        final_image = np.vstack(final_rows)
        
        # 5. Check/Create save directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")
        
        # 6. Generate filename and full path
        video_base_name = os.path.splitext(video_name)[0]
        base_filename = f"{q_name}_{video_base_name}_{frame_num:04d}.png"
        full_path = os.path.join(self.save_dir, base_filename)
        
        # 7. Save
        cv2.imwrite(full_path, final_image)
        print(f"Saved: {full_path}")