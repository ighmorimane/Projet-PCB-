# ═══════════════════════════════════════════════════════════════
# PER-CLASS THRESHOLD INFERENCE – BOOST OPEN_CIRCUIT DETECTION
# (No retraining, only detection parameters)
# ═══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from collections import defaultdict
import random
import subprocess

# ------------------------------------------------------------------
# 1️⃣ PATHS – ADJUST THESE TO YOUR SETUP
# ------------------------------------------------------------------
MODEL_PATH = "/srv/groups/group7/outputs/pcb_defect_classifier_colab.keras"

BASE_DIR = Path("/srv/groups/group7/data/pcb-defects/PCB_DATASET")
DATASET_IMAGES = BASE_DIR / "images"

OUTPUT_DIR = Path("/srv/groups/group7/outputs/detection_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Results will be saved to: {OUTPUT_DIR}")

# ------------------------------------------------------------------
# 2️⃣ CONFIGURATION (must match your model's training)
# ------------------------------------------------------------------
PATCH_SIZES = [96, 128]
STRIDES     = [64, 96]
NMS_IOU_THRESHOLD = 0.35
BATCH_SIZE = 32

DEFECT_CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
CLASSES = DEFECT_CLASSES + ['background']
BG_IDX = CLASSES.index('background')

CLASS_COLORS = {
    'missing_hole'   : (0,   0,   255),
    'mouse_bite'     : (0,   140, 255),
    'open_circuit'   : (0,   255, 255),
    'short'          : (255, 0,   0  ),
    'spur'           : (255, 0,   255),
    'spurious_copper': (0,   200, 0  ),
}

FOLDER_NAME_MAP = {
    'missing_hole':'Missing_hole', 'mouse_bite':'Mouse_bite', 'open_circuit':'Open_circuit',
    'short':'Short', 'spur':'Spur', 'spurious_copper':'Spurious_copper'
}

# ------------------------------------------------------------------
# 3️⃣ PER-CLASS THRESHOLDS
# ------------------------------------------------------------------
PER_CLASS_THRESHOLDS = {
    'missing_hole':      (0.75, 0.30),
    'mouse_bite':        (0.55, 0.25),
    'open_circuit':      (0.01, 0.05),
    'short':             (0.70, 0.25),
    'spur':              (0.70, 0.25),
    'spurious_copper':   (0.60, 0.15),
}

# ------------------------------------------------------------------
# 4️⃣ HELPER FUNCTIONS
# ------------------------------------------------------------------
def normalize_patch(patch_bgr):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_float = patch_rgb.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    return (patch_float - mean) / std

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

def apply_nms_class_aware(detections, iou_thresh):
    if not detections: return []
    by_class = defaultdict(list)
    for d in detections:
        by_class[d[4]].append(d)
    kept = []
    for cls, dets in by_class.items():
        dets.sort(key=lambda x: x[5], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if compute_iou(best[:4], d[:4]) < iou_thresh]
    return kept

def multi_scale_inference_per_class(model, image):
    h, w = image.shape[:2]
    all_detections = []
    target_size = PATCH_SIZES[0]
    for psize, stride in zip(PATCH_SIZES, STRIDES):
        patches = []
        positions = []
        for y in range(0, h - psize + 1, stride):
            for x in range(0, w - psize + 1, stride):
                patch = image[y:y+psize, x:x+psize]
                if patch.shape[:2] != (psize, psize): continue
                patch_resized = cv2.resize(patch, (target_size, target_size))
                patches.append(normalize_patch(patch_resized))
                positions.append((x, y, psize))
        if not patches: continue
        patches_arr = np.array(patches, dtype='float32')
        preds = model.predict(patches_arr, batch_size=BATCH_SIZE*2, verbose=0)
        for i, (x, y, psize) in enumerate(positions):
            prob = preds[i]
            class_idx = np.argmax(prob)
            cls_name = CLASSES[class_idx]
            conf = prob[class_idx]
            bg_conf = prob[BG_IDX]
            margin = conf - bg_conf
            thr_conf, thr_margin = PER_CLASS_THRESHOLDS.get(cls_name, (0.65, 0.20))
            if class_idx != BG_IDX and conf >= thr_conf and margin >= thr_margin:
                all_detections.append((x, y, x+psize, y+psize, cls_name, conf))
    return apply_nms_class_aware(all_detections, NMS_IOU_THRESHOLD)

def annotate_image(image, detections):
    annotated = image.copy()
    for (xmin, ymin, xmax, ymax, cls, conf) in detections:
        color = CLASS_COLORS.get(cls, (0,0,255))
        label = f"{cls} ({conf:.2f})"
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(annotated, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated

# ------------------------------------------------------------------
# 5️⃣ LOAD MODEL
# ------------------------------------------------------------------
print("Loading model...")
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# ------------------------------------------------------------------
# 6️⃣ GET RANDOM IMAGES FROM DATASET
# ------------------------------------------------------------------
all_images = []
for defect_class in DEFECT_CLASSES:
    folder = DATASET_IMAGES / FOLDER_NAME_MAP.get(defect_class, defect_class)
    if folder.exists():
        all_images.extend(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

if not all_images:
    raise FileNotFoundError(f"No images found in {DATASET_IMAGES}. Check BASE_DIR.")

num_test = 5
test_images = random.sample(all_images, min(num_test, len(all_images)))
print(f"Testing on {len(test_images)} random images from dataset.\n")

# ------------------------------------------------------------------
# 7️⃣ RUN DETECTION AND SAVE RESULTS
# ------------------------------------------------------------------
saved_paths = []
for idx, img_path in enumerate(test_images):
    print(f"[{idx+1}/{len(test_images)}] Processing: {img_path.name}")
    image = cv2.imread(str(img_path))
    if image is None:
        print("  Failed to read, skipping.")
        continue

    detections = multi_scale_inference_per_class(model, image)
    annotated = annotate_image(image, detections)

    out_filename = f"detected_{img_path.stem}.jpg"
    out_path = OUTPUT_DIR / out_filename
    cv2.imwrite(str(out_path), annotated)
    saved_paths.append(out_path)
    print(f"  Defects found: {len(detections)} | Saved to: {out_path}")

# ------------------------------------------------------------------
# 8️⃣ PACKAGE RESULTS INTO ZIP
# ------------------------------------------------------------------
zip_path = OUTPUT_DIR / "detection_results.zip"
subprocess.run(["zip", "-r", str(zip_path), str(OUTPUT_DIR)])
print(f"\nAll results saved in: {OUTPUT_DIR}")
print(f"Zip file: {zip_path}")