import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════
# CONFIGURATION (tuned for better performance)
# ═════════════════════════════════════════════
BASE_DIR            = Path("/srv/groups/group7/data/pcb-defects/PCB_DATASET/PCB_DATASET")
DATASET_IMAGES      = BASE_DIR / "images"
DATASET_ANNOTATIONS = BASE_DIR / "Annotations"

OUTPUT_DIR        = Path("/srv/groups/group7/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH   = OUTPUT_DIR / "pcb_defect_classifier_optimized.keras"
RESULT_IMAGE_PATH = OUTPUT_DIR / "pcb_annotated_result_optimized.jpg"

# Multi‑scale sliding window
PATCH_SIZES = [96, 128]          # two scales to catch small & medium defects
STRIDES     = [64, 96]           # stride ~ half patch size
BATCH_SIZE  = 32
EPOCHS      = 40                 # fewer epochs but with hard mining
CONF_THRESHOLD    = 0.65         # confident defect
BG_CONF_THRESH    = 0.30         # background confidence must be low
NMS_IOU_THRESHOLD = 0.35         # tighter NMS, per‑class

LEARNING_RATE     = 0.0005
IMAGE_SPLIT_RATIO = 0.2
RANDOM_SEED       = 42

# Data balancing
DEFECT_PATCHES_PER_BOX = 5      # augmented crops per defect box
MAX_BG_ATTEMPTS        = 100    # increase to find more pure background
HARD_NEGATIVE_ITER     = 2      # number of hard negative mining rounds

# Augmentation for training patches
ROTATION_RANGE   = 30
SHIFT_RANGE      = 0.15
ZOOM_RANGE       = 0.15
BRIGHTNESS_RANGE = [0.7, 1.3]

DEFECT_CLASSES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
]
CLASSES = DEFECT_CLASSES + ['background']
BG_IDX  = CLASSES.index('background')

# Folder name mapping (as provided)
FOLDER_NAME_MAP = {
    'missing_hole'   : 'Missing_hole',
    'mouse_bite'     : 'Mouse_bite',
    'open_circuit'   : 'Open_circuit',
    'short'          : 'Short',
    'spur'           : 'Spur',
    'spurious_copper': 'Spurious_copper',
}

CLASS_MAP = {
    'missing_hole'   : 'missing_hole',
    'mouse_bite'     : 'mouse_bite',
    'mouse-bite'     : 'mouse_bite',
    'open_circuit'   : 'open_circuit',
    'open-circuit'   : 'open_circuit',
    'short'          : 'short',
    'short_circuit'  : 'short',
    'spur'           : 'spur',
    'spurious_copper': 'spurious_copper',
    'spurious-copper': 'spurious_copper',
}

CLASS_COLORS = {
    'missing_hole'   : (0,   0,   255),
    'mouse_bite'     : (0,   140, 255),
    'open_circuit'   : (0,   255, 255),
    'short'          : (255, 0,   0  ),
    'spur'           : (255, 0,   255),
    'spurious_copper': (0,   200, 0  ),
}

# ═════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════
def normalize_patch(patch_bgr):
    """Normalise patch using global mean/std (ImageNet style)."""
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_float = patch_rgb.astype('float32') / 255.0
    # Use fixed stats for stability (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    patch_norm = (patch_float - mean) / std
    return patch_norm

def augment_patch(patch_bgr, rng, n=5):
    """Generate n augmented versions (used for defect oversampling)."""
    results = [patch_bgr]
    h, w = patch_bgr.shape[:2]
    for _ in range(n - 1):
        aug = patch_bgr.copy()
        angle = rng.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.random() > 0.5:
            aug = cv2.flip(aug, 1)
        if rng.random() > 0.5:
            aug = cv2.flip(aug, 0)
        factor = rng.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        aug = np.clip(aug.astype('float32') * factor, 0, 255).astype('uint8')
        results.append(aug)
    return results

def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            raw_name = obj.find('name').text.lower().strip()
            mapped = CLASS_MAP.get(raw_name)
            if mapped not in DEFECT_CLASSES:
                continue
            bnd = obj.find('bndbox')
            xmin = int(float(bnd.find('xmin').text))
            ymin = int(float(bnd.find('ymin').text))
            xmax = int(float(bnd.find('xmax').text))
            ymax = int(float(bnd.find('ymax').text))
            boxes.append((mapped, xmin, ymin, xmax, ymax))
        return boxes
    except Exception as e:
        print(f"Warning: Could not parse {xml_path}: {e}")
        return []

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# ═════════════════════════════════════════════
# DATA SPLIT (image‑level, no leakage)
# ═════════════════════════════════════════════
def split_image_files():
    print("\n[1/6] Splitting PCB images at image level...")
    if not DATASET_IMAGES.exists() or not DATASET_ANNOTATIONS.exists():
        raise FileNotFoundError("Dataset folders missing.")
    
    all_records = []
    for class_name in DEFECT_CLASSES:
        folder_name = FOLDER_NAME_MAP.get(class_name, class_name)
        img_folder = DATASET_IMAGES / folder_name
        xml_folder = DATASET_ANNOTATIONS / folder_name
        if not xml_folder.exists():
            continue
        for xml_file in sorted(xml_folder.glob("*.xml")):
            img_file = img_folder / (xml_file.stem + ".jpg")
            if not img_file.exists():
                img_file = img_folder / (xml_file.stem + ".png")
            if not img_file.exists():
                continue
            all_records.append((xml_file, img_file, class_name))
    
    if not all_records:
        raise ValueError("No records found.")
    
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(all_records))
    all_records = [all_records[i] for i in indices]
    split_at = int(len(all_records) * (1 - IMAGE_SPLIT_RATIO))
    train_records = all_records[:split_at]
    test_records = all_records[split_at:]
    
    print(f"  Train images: {len(train_records)} | Test images: {len(test_records)}")
    return train_records, test_records

# ═════════════════════════════════════════════
# PATCH EXTRACTION (balanced + optional hard negatives)
# ═════════════════════════════════════════════
def extract_patches(records, hard_negatives=None):
    """
    Extract defect patches (augmented) and background patches.
    If hard_negatives is provided (list of (patch, label)), add them as background.
    """
    print(f"\n[2/6] Extracting patches from {len(records)} images...")
    X, y = [], []
    cls_counts = Counter()
    rng = np.random.default_rng(RANDOM_SEED)
    
    for idx, (xml_path, img_path, class_name) in enumerate(records):
        if idx % 50 == 0:
            print(f"  {idx+1}/{len(records)}")
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = parse_xml(xml_path)
        
        # Defect patches
        defect_patches_this = 0
        for (mapped_class, xmin, ymin, xmax, ymax) in boxes:
            xmin, xmax = max(0, xmin), min(w, xmax)
            ymin, ymax = max(0, ymin), min(h, ymax)
            if xmax - xmin < 10 or ymax - ymin < 10:
                continue
            patch = image[ymin:ymax, xmin:xmax]
            if patch.size == 0:
                continue
            patch_resized = cv2.resize(patch, (PATCH_SIZES[0], PATCH_SIZES[0]), interpolation=cv2.INTER_AREA)
            for aug_patch in augment_patch(patch_resized, rng, n=DEFECT_PATCHES_PER_BOX):
                X.append(normalize_patch(aug_patch))
                y.append(CLASSES.index(mapped_class))
                cls_counts[mapped_class] += 1
                defect_patches_this += 1
        
        # Background patches – match defect count for this image
        bg_target = max(5, defect_patches_this)   # at least 5 per image
        bg_sampled, attempts = 0, 0
        while bg_sampled < bg_target and attempts < MAX_BG_ATTEMPTS:
            bx = rng.integers(0, max(1, w - PATCH_SIZES[0]))
            by = rng.integers(0, max(1, h - PATCH_SIZES[0]))
            patch_rect = [bx, by, bx + PATCH_SIZES[0], by + PATCH_SIZES[0]]
            overlap = any(compute_iou(patch_rect, [x1,y1,x2,y2]) > 0.1 for (_,x1,y1,x2,y2) in boxes)
            if not overlap:
                patch = image[by:by+PATCH_SIZES[0], bx:bx+PATCH_SIZES[0]]
                if patch.shape[:2] == (PATCH_SIZES[0], PATCH_SIZES[0]):
                    X.append(normalize_patch(patch))
                    y.append(BG_IDX)
                    cls_counts['background'] += 1
                    bg_sampled += 1
            attempts += 1
    
    # Add hard negatives if provided
    if hard_negatives is not None:
        for patch_norm, _ in hard_negatives:
            X.append(patch_norm)
            y.append(BG_IDX)
            cls_counts['background'] += 1
    
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    print(f"\n  Total patches: {len(X)}")
    for cls in CLASSES:
        cnt = cls_counts.get(cls,0)
        print(f"    {cls:20s}: {cnt:5d}")
    bg_ratio = cls_counts['background'] / len(y)
    print(f"  Background ratio: {bg_ratio:.1%}")
    return X, y

# ═════════════════════════════════════════════
# DATA GENERATOR WITH AUGMENTATION
# ═════════════════════════════════════════════
class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=ROTATION_RANGE if augment else 0,
            width_shift_range=SHIFT_RANGE if augment else 0,
            height_shift_range=SHIFT_RANGE if augment else 0,
            zoom_range=ZOOM_RANGE if augment else 0,
            brightness_range=BRIGHTNESS_RANGE if augment else None,
            horizontal_flip=augment,
            vertical_flip=augment,
            fill_mode='reflect'
        )
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X = self.X[batch_idx]
        batch_y = self.y[batch_idx]
        if self.augment:
            # Apply augmentation
            batch_X = self.datagen.flow(batch_X, batch_y, batch_size=len(batch_X), shuffle=False)[0][0]
        return batch_X, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ═════════════════════════════════════════════
# FOCAL LOSS
# ═════════════════════════════════════════════
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.exp(-ce)
        focal = alpha * (1 - p_t)**gamma * ce
        return tf.reduce_mean(focal)
    return loss_fn

# ═════════════════════════════════════════════
# BUILD MODEL WITH PRETRAINED BACKBONE
# ═════════════════════════════════════════════
def build_model():
    # MobileNetV2 pretrained on ImageNet
    base = MobileNetV2(input_shape=(PATCH_SIZES[0], PATCH_SIZES[0], 3),
                       include_top=False, weights='imagenet')
    base.trainable = False   # freeze initially
    
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    
    model = keras.Model(inputs=base.input, outputs=outputs)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, decay_steps=1000, decay_rate=0.9, staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    return model, base

# ═════════════════════════════════════════════
# HARD NEGATIVE MINING
# ═════════════════════════════════════════════
def mine_hard_negatives(model, records, top_k=500):
    """Extract background patches from training images that model misclassifies as defect."""
    print("\n  Hard negative mining...")
    hard_patches = []
    rng = np.random.default_rng(RANDOM_SEED)
    
    for xml_path, img_path, _ in records:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = parse_xml(xml_path)
        
        # Sample many random background patches
        candidates = []
        for _ in range(100):   # per image
            bx = rng.integers(0, max(1, w - PATCH_SIZES[0]))
            by = rng.integers(0, max(1, h - PATCH_SIZES[0]))
            patch_rect = [bx, by, bx+PATCH_SIZES[0], by+PATCH_SIZES[0]]
            overlap = any(compute_iou(patch_rect, [x1,y1,x2,y2]) > 0.1 for (_,x1,y1,x2,y2) in boxes)
            if not overlap:
                patch = image[by:by+PATCH_SIZES[0], bx:bx+PATCH_SIZES[0]]
                if patch.shape[:2] == (PATCH_SIZES[0], PATCH_SIZES[0]):
                    patch_norm = normalize_patch(patch)
                    # Predict
                    prob = model.predict(patch_norm.reshape(1,*patch_norm.shape), verbose=0)[0]
                    pred_class = np.argmax(prob)
                    if pred_class != BG_IDX and prob[pred_class] > 0.5:
                        candidates.append((patch_norm, prob[pred_class]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        hard_patches.extend(candidates[:min(10, len(candidates))])
        if len(hard_patches) >= top_k:
            break
    
    print(f"  Found {len(hard_patches)} hard negatives (top {top_k})")
    return hard_patches[:top_k]

# ═════════════════════════════════════════════
# TRAINING WITH HARD MINING CYCLES
# ═════════════════════════════════════════════
def train_with_hard_mining(X_init, y_init, val_data):
    X_train, y_train = X_init, y_init
    model, base = build_model()
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    for mining_round in range(HARD_NEGATIVE_ITER + 1):
        print(f"\n[3/6] Training round {mining_round+1}/{HARD_NEGATIVE_ITER+1}")
        train_gen = BalancedDataGenerator(X_train, y_train, BATCH_SIZE, augment=True)
        val_gen = BalancedDataGenerator(val_data[0], val_data[1], BATCH_SIZE, augment=False)
        
        epochs_this = EPOCHS // (HARD_NEGATIVE_ITER+1) if mining_round < HARD_NEGATIVE_ITER else EPOCHS // 2
        model.fit(train_gen, epochs=epochs_this, validation_data=val_gen, callbacks=callbacks, verbose=1)
        
        if mining_round < HARD_NEGATIVE_ITER:
            # Unfreeze top layers after first round
            if mining_round == 0:
                base.trainable = True
                for layer in base.layers[:100]:
                    layer.trainable = False
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=focal_loss(), metrics=['accuracy'])
                print("  Unfroze top part of MobileNetV2 for fine-tuning.")
            
            # Mine hard negatives from training records (need access to original records)
            hard_neg = mine_hard_negatives(model, train_records, top_k=500)
            if hard_neg:
                # Add to training set
                X_hard = np.array([hn[0] for hn in hard_neg])
                y_hard = np.full(len(X_hard), BG_IDX)
                X_train = np.concatenate([X_train, X_hard], axis=0)
                y_train = np.concatenate([y_train, y_hard], axis=0)
                print(f"  Added {len(X_hard)} hard negatives, new training size: {len(X_train)}")
    
    return model

# ═════════════════════════════════════════════
# MULTI‑SCALE INFERENCE
# ═════════════════════════════════════════════
def multi_scale_inference(model, image):
    h, w = image.shape[:2]
    all_detections = []
    
    for psize, stride in zip(PATCH_SIZES, STRIDES):
        # Resize image if needed? No, we slide over original.
        # But if psize != PATCH_SIZES[0], we need to resize patches to model input size.
        target_size = PATCH_SIZES[0]   # model input size
        patches = []
        positions = []
        for y in range(0, h - psize + 1, stride):
            for x in range(0, w - psize + 1, stride):
                patch = image[y:y+psize, x:x+psize]
                if patch.shape[:2] != (psize, psize):
                    continue
                # Resize to model input size
                patch_resized = cv2.resize(patch, (target_size, target_size))
                patches.append(normalize_patch(patch_resized))
                positions.append((x, y, psize))
        
        if not patches:
            continue
        patches_arr = np.array(patches, dtype='float32')
        preds = model.predict(patches_arr, batch_size=BATCH_SIZE*2, verbose=0)
        
        for i, (x, y, psize) in enumerate(positions):
            prob = preds[i]
            class_idx = np.argmax(prob)
            conf = prob[class_idx]
            bg_conf = prob[BG_IDX]
            if class_idx != BG_IDX and conf >= CONF_THRESHOLD and bg_conf < BG_CONF_THRESH:
                all_detections.append((x, y, x+psize, y+psize, CLASSES[class_idx], conf))
    
    return all_detections

def apply_nms_class_aware(detections, iou_thresh):
    if not detections:
        return []
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

def run_inference(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    detections = multi_scale_inference(model, image)
    detections = apply_nms_class_aware(detections, NMS_IOU_THRESHOLD)
    return image, detections

# ═════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════
def plot_training_history(history):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png')
    plt.close()

# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
def main():
    print("="*60)
    print("  OPTIMIZED CNN-BASED PCB DEFECT DETECTOR")
    print("  Features: Pretrained backbone, focal loss, hard mining, multi‑scale")
    print("="*60)
    
    global train_records, test_records
    train_records, test_records = split_image_files()
    
    # Initial patch extraction
    X, y = extract_patches(train_records, hard_negatives=None)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    val_data = (X_val, y_val)
    
    # Train with hard negative mining
    model = train_with_hard_mining(X_train, y_train, val_data)
    
    # Evaluate on test set (first 20 images for speed, you can increase)
    print("\n[4/6] Evaluating on test images...")
    total_defects = 0
    for i, (xml_path, img_path, _) in enumerate(test_records[:20]):
        _, dets = run_inference(model, img_path)
        total_defects += len(dets)
        print(f"  {i+1:2d}. {img_path.name}: {len(dets)} defects")
    print(f"  Average defects per image: {total_defects/20:.1f}")
    
    # Save a result image
    print("\n[5/6] Generating annotated result...")
    best_img, best_dets = None, []
    for _, img_path, _ in test_records[:10]:
        img, dets = run_inference(model, img_path)
        if len(dets) > len(best_dets):
            best_img, best_dets = img, dets
            best_path = img_path
    if best_img is not None:
        annotated = best_img.copy()
        for (xmin, ymin, xmax, ymax, cls, conf) in best_dets:
            color = CLASS_COLORS.get(cls, (0,0,255))
            label = f"{cls} ({conf:.2f})"
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(annotated, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(str(RESULT_IMAGE_PATH), annotated)
        print(f"  Saved result: {RESULT_IMAGE_PATH}")
    
    print("\n[6/6] Done.")
    print(f"  Model saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

