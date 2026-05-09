import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
# Dataset 1 — green PCB (XML annotations)
DATASET1_IMAGES      = Path("/srv/groups/group7/data/pcb-defects/PCB_DATASET/images")
DATASET1_ANNOTATIONS = Path("/srv/groups/group7/data/pcb-defects/PCB_DATASET/Annotations")

# Dataset 2 — yellow PCB (COCO JSON annotation)
DATASET2_IMAGES      = Path("/srv/groups/group7/data/pcb-defects2/images")
DATASET2_JSON        = Path("/srv/groups/group7/data/pcb-defects2/annotation/_annotations.coco.json")

# Output directory
OUTPUT_DIR = Path("/srv/groups/group7/projects/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classes — unified across both datasets
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit',
           'short', 'spur', 'spurious_copper']

# Dataset 2 uses 'missing_pad' instead of 'missing_hole' → map it
CLASS_MAP = {
    'missing_pad'    : 'missing_hole',
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

PATCH_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS     = 20

print("=" * 55)
print(" PCB Defect Classifier — Two Dataset Training")
print("=" * 55)
print(f" Dataset 1 : {DATASET1_IMAGES}")
print(f" Dataset 2 : {DATASET2_IMAGES}")
print(f" Classes   : {CLASSES}")
print(f" Output    : {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 1. NORMALIZATION FUNCTION
# Applied to every patch from both datasets
# ─────────────────────────────────────────────
def normalize_patch(patch, target_mean=128.0):
    """
    Normalize a patch so both datasets look comparable:
    1. Grayscale       — removes color difference (green vs yellow PCB)
    2. Brightness fix  — dark patches get brighter, bright ones get darker
                         both land at target_mean (128 = middle gray)
    3. 3 channels      — CNN needs 3 channels
    4. Scale to [0,1]  — pixel values between 0 and 1
    """
    # Step 1 — Grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Step 2 — Brightness targeting
    current_mean = gray.mean()
    scale        = target_mean / (current_mean + 1e-6)
    gray         = np.clip(gray * scale, 0, 255).astype(np.uint8)

    # Step 3 — Back to 3 channels
    patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Step 4 — Scale to [0, 1]
    patch = patch.astype('float32') / 255.0

    return patch

# ─────────────────────────────────────────────
# 2. DATASET 1 PARSER — XML (green PCB)
# ─────────────────────────────────────────────
def parse_xml(xml_path):
    """
    Read Pascal VOC XML annotation file.
    Returns list of (class_name, xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name   = obj.find('name').text.lower().strip()
        bndbox = obj.find('bndbox')
        xmin   = int(bndbox.find('xmin').text)
        ymin   = int(bndbox.find('ymin').text)
        xmax   = int(bndbox.find('xmax').text)
        ymax   = int(bndbox.find('ymax').text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def extract_patches_xml(images_path, annotations_path, classes):
    """
    Extract and normalize patches from Dataset 1 (XML annotations).
    """
    X, y = [], []

    folder_map = {
        f.name.lower(): f.name
        for f in images_path.iterdir()
        if f.is_dir()
    }

    for class_name in classes:
        folder_name = folder_map.get(class_name)
        if not folder_name:
            print(f"  WARNING: folder not found for {class_name}")
            continue

        img_folder = images_path      / folder_name
        xml_folder = annotations_path / folder_name
        xml_files  = sorted(xml_folder.glob("*.xml"))

        print(f"  [XML] {class_name} — {len(xml_files)} files")

        for xml_file in xml_files:
            img_file = img_folder / (xml_file.stem + ".jpg")
            if not img_file.exists():
                continue

            image = cv2.cvtColor(
                cv2.imread(str(img_file)),
                cv2.COLOR_BGR2RGB
            )
            boxes = parse_xml(xml_file)

            for (name, xmin, ymin, xmax, ymax) in boxes:
                mapped = CLASS_MAP.get(name)
                if mapped not in classes:
                    continue
                patch = image[ymin:ymax, xmin:xmax]
                if patch.size == 0:
                    continue
                patch = cv2.resize(patch, PATCH_SIZE)
                patch = normalize_patch(patch)
                X.append(patch)
                y.append(classes.index(mapped))

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 3. DATASET 2 PARSER — COCO JSON (yellow PCB)
# ─────────────────────────────────────────────
def extract_patches_json(images_path, json_path, classes):
    """
    Extract and normalize patches from Dataset 2 (COCO JSON annotation).
    COCO bbox format: [x, y, width, height]
    """
    X, y = [], []

    with open(json_path, 'r') as f:
        data = json.load(f)

    # category_id → class name
    cat_map = {
        cat['id']: cat['name'].lower().strip()
        for cat in data['categories']
    }

    # image_id → file_name
    img_map = {
        img['id']: img['file_name']
        for img in data['images']
    }

    class_counts = {cls: 0 for cls in classes}

    for ann in data['annotations']:
        # Get class name
        raw_name = cat_map.get(ann['category_id'], '')
        mapped   = CLASS_MAP.get(raw_name)
        if mapped not in classes:
            continue

        # Get image filename
        file_name = img_map.get(ann['image_id'])
        if not file_name:
            continue

        img_file = images_path / file_name
        if not img_file.exists():
            continue

        # COCO bbox: [x, y, width, height] → convert to xmin ymin xmax ymax
        x, y_coord, w, h = ann['bbox']
        xmin = int(x)
        ymin = int(y_coord)
        xmax = int(x + w)
        ymax = int(y_coord + h)

        image = cv2.cvtColor(
            cv2.imread(str(img_file)),
            cv2.COLOR_BGR2RGB
        )
        patch = image[ymin:ymax, xmin:xmax]
        if patch.size == 0:
            continue

        patch = cv2.resize(patch, PATCH_SIZE)
        patch = normalize_patch(patch)
        X.append(patch)
        y.append(classes.index(mapped))
        class_counts[mapped] += 1

    print(f"  [JSON] patches per class: {class_counts}")
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 4. EXTRACT PATCHES FROM BOTH DATASETS
# ─────────────────────────────────────────────
print("\n[1/7] Extracting patches from Dataset 1 (XML — green PCB)...")
X1, y1 = extract_patches_xml(DATASET1_IMAGES, DATASET1_ANNOTATIONS, CLASSES)
print(f"      X1 shape : {X1.shape}")
print(f"      y1 dist  : {np.bincount(y1)}")

print("\n[2/7] Extracting patches from Dataset 2 (JSON — yellow PCB)...")
X2, y2 = extract_patches_json(DATASET2_IMAGES, DATASET2_JSON, CLASSES)
print(f"      X2 shape : {X2.shape}")
print(f"      y2 dist  : {np.bincount(y2)}")

# ─────────────────────────────────────────────
# 5. COMBINE BOTH DATASETS
# ─────────────────────────────────────────────
print("\n[3/7] Combining datasets...")
X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)
print(f"      Combined X shape : {X.shape}")
print(f"      Combined y dist  : {np.bincount(y)}")

# ─────────────────────────────────────────────
# 6. VISUALISE SAMPLE PATCHES
# ─────────────────────────────────────────────
print("\n      Saving sample patch grid...")
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for col, class_name in enumerate(CLASSES):
    idx  = np.where(y == col)[0][0]
    idx2 = np.where(y == col)[0][1]
    axes[0, col].imshow(X[idx])
    axes[0, col].set_title(class_name, fontsize=7)
    axes[0, col].axis('off')
    axes[1, col].imshow(X[idx2])
    axes[1, col].axis('off')
plt.suptitle('Sample normalized patches — both datasets combined')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_patches.png', dpi=100)
plt.close()
print(f"      Saved → sample_patches.png")

# ─────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[4/7] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"      X_train : {X_train.shape}")
print(f"      X_test  : {X_test.shape}")
print(f"      Train dist : {np.bincount(y_train)}")
print(f"      Test  dist : {np.bincount(y_test)}")

# ─────────────────────────────────────────────
# 8. DATA AUGMENTATION
# ─────────────────────────────────────────────
print("\n      Configuring data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
datagen.fit(X_train)
print("      Augmentation configured")

# ─────────────────────────────────────────────
# 9. BUILD THE CNN MODEL
# ─────────────────────────────────────────────
print("\n[5/7] Building CNN model...")
model = keras.models.Sequential([
    # Block 1
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),

    # Block 2
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),

    # Block 3
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

    # Classifier head
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(6, activation='softmax')  # 6 defect classes
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────────
# 10. TRAIN THE MODEL
# ─────────────────────────────────────────────
print("\n[6/7] Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─────────────────────────────────────────────
# 11. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
print("\n      Saving training curves...")
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(history.history['loss'],     label='Train Loss', color='tab:blue')
ax1.plot(history.history['val_loss'], label='Val Loss',   color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.plot(history.history['accuracy'],     label='Train Acc', color='tab:orange')
ax2.plot(history.history['val_accuracy'], label='Val Acc',   color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper right')

plt.title('Training Loss and Accuracy — Two Datasets')
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=100)
plt.close()
print(f"      Saved → training_curves.png")

# ─────────────────────────────────────────────
# 12. EVALUATE THE MODEL
# ─────────────────────────────────────────────
print("\n[7/7] Evaluating model...")
score = model.evaluate(X_test, y_test, verbose=0)
print(f"      Test loss     : {score[0]:.4f}")
print(f"      Test accuracy : {score[1]:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=-1)

print("\n" + classification_report(
    y_test, y_pred,
    target_names=CLASSES,
    digits=4
))

# ─────────────────────────────────────────────
# 13. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("      Saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix — Two Datasets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=100)
plt.close()
print(f"      Saved → confusion_matrix.png")

# ─────────────────────────────────────────────
# 14. SAVE THE MODEL
# ─────────────────────────────────────────────
print("\n      Saving model...")
model.save(OUTPUT_DIR / 'pcb_defect_classifier.keras')
print(f"      Saved → pcb_defect_classifier.keras")

print("\n" + "=" * 55)
print(" Training complete! Output files:")
print(f"  {OUTPUT_DIR / 'sample_patches.png'}")
print(f"  {OUTPUT_DIR / 'training_curves.png'}")
print(f"  {OUTPUT_DIR / 'confusion_matrix.png'}")
print(f"  {OUTPUT_DIR / 'pcb_defect_classifier.keras'}")
print("=" * 55)