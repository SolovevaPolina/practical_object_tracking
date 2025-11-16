import numpy as np
import cv2
import sys
import torch
import torchvision.models as models
import torchvision.transforms as T
import tracking_utils as utils

# --- Settings ---
LAYER_TO_ANALYZE_INDEX = 19
NUM_CHANNELS_TO_USE = 8
IMG_SIZE = 224 # VGG16 input size
FEATURE_SIZE = 28 # conv4_2 output feature size

# --- VGG16 Model Setup ---
VGG_MODEL = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
VGG_MODEL.eval()
feature_extractor = VGG_MODEL.features[:LAYER_TO_ANALYZE_INDEX + 1]

# --- Helper Functions ---
def preprocess_image(frame_224):
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(frame_224).unsqueeze(0)

def get_features(model, preprocessed_frame):
    with torch.no_grad():
        features = model(preprocessed_frame)
    return features.squeeze(0) # [C, H, W]

def scale_coords_224_to_28(r, c, h, w):
    scale = FEATURE_SIZE / IMG_SIZE # (28 / 224 = 0.125)
    r_s = int(r * scale)
    c_s = int(c * scale)
    h_s = int(h * scale)
    w_s = int(w * scale)
    
    if h_s == 0: h_s = 1
    if w_s == 0: w_s = 1
    return (r_s, c_s, h_s, w_s)

# --- Main Script Init ---
win_manager = utils.WindowManager()
cap, _ = utils.setup_video_and_args()
video_filename = sys.argv[1]

ret, frame_orig = cap.read()
frame_224 = cv2.resize(frame_orig, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

print("Please select ROI on the SQUASHED (224x224) image.")
r, c, h, w = utils.select_roi(frame_224.copy())
win_manager.set_static_roi_frame(frame_224, r, c, h, w)

# --- Model Creation (Frame 1) ---
print("Processing Frame 1 to find golden channels...")
prep_frame = preprocess_image(frame_224)
frame_features = get_features(feature_extractor, prep_frame)
num_channels_total = frame_features.shape[0]

r_s, c_s, h_s, w_s = scale_coords_224_to_28(r, c, h, w)

roi_mask_tensor = torch.zeros(FEATURE_SIZE, FEATURE_SIZE)
roi_mask_tensor[c_s : c_s+w_s, r_s : r_s+h_s] = 1
bg_mask_tensor = 1 - roi_mask_tensor

scores = []
for i in range(num_channels_total):
    channel_map = frame_features[i]
    activated_map = torch.nn.functional.relu(channel_map)
    mean_roi = (activated_map * roi_mask_tensor).sum() / (roi_mask_tensor.sum() + 1e-6)
    mean_bg = (activated_map * bg_mask_tensor).sum() / (bg_mask_tensor.sum() + 1e-6)
    score = mean_roi - mean_bg
    scores.append( (score.item(), i) )
scores.sort(key=lambda x: x[0], reverse=True)
top_k_indices = [idx for score, idx in scores[:NUM_CHANNELS_TO_USE]]
print(f"Using top {NUM_CHANNELS_TO_USE} channels: {top_k_indices}")

golden_features_map = frame_features[top_k_indices]
roi_features = golden_features_map[:, c_s:c_s+w_s, r_s:r_s+h_s]
template_vector = roi_features.mean(dim=[1, 2])

# --- Setup MeanShift ---
track_window = (r, c, h, w) 
prev_r, prev_c = r, c
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
cpt = 1
print("Model created. Starting tracking...")

# --- Tracking Loop ---
while(1):
    ret ,frame_orig = cap.read()

    frame_224 = cv2.resize(frame_orig, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    prep_frame_n = preprocess_image(frame_224)
    frame_features_n = get_features(feature_extractor, prep_frame_n)
    golden_features_n = frame_features_n[top_k_indices] # [8, 28, 28]

    template = template_vector.unsqueeze(-1).unsqueeze(-1) # [8, 1, 1]
    dst_tensor = torch.nn.functional.cosine_similarity(golden_features_n, template, dim=0) # [28, 28]

    dst_norm = cv2.normalize(dst_tensor.numpy(), None, 0, 255, cv2.NORM_MINMAX)
    dst_8u = dst_norm.astype(np.uint8)

    dst_fullsize = cv2.resize(dst_8u, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    dst_smooth = cv2.GaussianBlur(dst_fullsize, (11, 11), 0) 

    vel_r = r - prev_r
    vel_c = c - prev_c
    guess_r = r + vel_r
    guess_c = c + vel_c
    
    guess_r = np.clip(guess_r, 0, IMG_SIZE - h)
    guess_c = np.clip(guess_c, 0, IMG_SIZE - w)
    
    track_window_guess = (guess_r, guess_c, h, w)
    
    ret, track_window = cv2.meanShift(dst_smooth, track_window_guess, term_crit) 
    
    r, c, h, w = track_window
    prev_r, prev_c = r, c

    frame_tracked = cv2.rectangle(frame_224.copy(), (r, c), (r+h, c+w), (255,0,0) ,2)

    win_manager.show('Sequence (224x224)', frame_tracked)
    win_manager.show('Back Projection (224x224)', dst_smooth)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        win_manager.save_composite_image("Q6", video_filename, cpt)
    cpt += 1

cv2.destroyAllWindows()
cap.release()