import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# ─────────────────────────────────────────────
# 1. DOWNLOAD DATASET
# ─────────────────────────────────────────────
print("=" * 50)
print("Downloading PCB defect dataset...")
print("=" * 50)

path = kagglehub.dataset_download("akhatova/pcb-defects")

IMAGES_PATH      = Path(path) / "PCB_DATASET" / "images"
ANNOTATIONS_PATH = Path(path) / "PCB_DATASET" / "Annotations"
CLASSES          = ['missing_hole', 'mouse_bite', 'open_circuit',
                    'short', 'spur', 'spurious_copper']

print("Images path     :", IMAGES_PATH)
print("Annotations path:", ANNOTATIONS_PATH)
print("Classes         :", CLASSES)

# ─────────────────────────────────────────────
# 2. PARSE XML ANNOTATIONS
# ─────────────────────────────────────────────
def parse_xml(xml_path):
    """
    Read one Pascal VOC XML annotation file.
    Returns a list of (class_name, xmin, ymin, xmax, ymax) tuples.
    One image can contain multiple defects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name   = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin   = int(bndbox.find('xmin').text)
        ymin   = int(bndbox.find('ymin').text)
        xmax   = int(bndbox.find('xmax').text)
        ymax   = int(bndbox.find('ymax').text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes

# ─────────────────────────────────────────────
# 3. EXTRACT PATCHES
# ─────────────────────────────────────────────
PATCH_SIZE = (100, 100)

def extract_patches(images_path, annotations_path, classes):
    """
    For each class folder:
      - Read each XML → get bounding boxes
      - Crop the patch from the image
      - Resize to 100x100
      - Label with class index
    Returns X (patches) and y (labels).
    """
    X = []
    y = []

    # map lowercase class name → actual folder name (Missing_hole etc.)
    folder_map = {f.name.lower(): f.name for f in images_path.iterdir() if f.is_dir()}

    for class_name in classes:
        folder_name = folder_map[class_name]
        img_folder  = images_path      / folder_name
        xml_folder  = annotations_path / folder_name

        xml_files = sorted(xml_folder.glob("*.xml"))
        print(f"  [{class_name}] — {len(xml_files)} annotation files")

        for xml_file in xml_files:
            img_file = img_folder / (xml_file.stem + ".jpg")
            if not img_file.exists():
                continue

            image = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
            boxes = parse_xml(xml_file)

            for (name, xmin, ymin, xmax, ymax) in boxes:
                patch         = image[ymin:ymax, xmin:xmax]
                patch_resized = cv2.resize(patch, PATCH_SIZE)
                X.append(patch_resized)
                y.append(classes.index(name))

    return np.array(X), np.array(y)


print("\nExtracting patches...")
X, y = extract_patches(IMAGES_PATH, ANNOTATIONS_PATH, CLASSES)

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ─────────────────────────────────────────────
# 4. VISUALISE SAMPLE PATCHES
# ─────────────────────────────────────────────
print("\nSaving sample patch grid...")

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for col, class_name in enumerate(CLASSES):
    idx  = np.where(y == col)[0][0]
    idx2 = np.where(y == col)[0][1]
    axes[0, col].imshow(X[idx])
    axes[0, col].set_title(class_name, fontsize=7)
    axes[0, col].axis('off')
    axes[1, col].imshow(X[idx2])
    axes[1, col].axis('off')

plt.suptitle('Sample patches per defect class')
plt.tight_layout()
plt.savefig('sample_patches.png', dpi=100)
plt.close()
print("  Saved → sample_patches.png")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Train class distribution : {np.bincount(y_train)}")
print(f"Test  class distribution : {np.bincount(y_test)}")

# ─────────────────────────────────────────────
# 6. DATA AUGMENTATION
# ─────────────────────────────────────────────
print("\nConfiguring data augmentation...")

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
print("  Augmentation configured")

# ─────────────────────────────────────────────
# 7. BUILD THE CNN MODEL
# ─────────────────────────────────────────────
print("\nBuilding CNN model...")

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
# 8. TRAIN THE MODEL
# ─────────────────────────────────────────────
print("\nTraining model...")

BATCH_SIZE = 32
EPOCHS     = 20

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─────────────────────────────────────────────
# 9. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
print("\nSaving training curves...")

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

plt.title('Training Loss and Accuracy')
fig.tight_layout()
plt.savefig('training_curves.png', dpi=100)
plt.close()
print("  Saved → training_curves.png")

# ─────────────────────────────────────────────
# 10. EVALUATE THE MODEL
# ─────────────────────────────────────────────
print("\nEvaluating model on test set...")

score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss     : {score[0]:.4f}")
print(f"Test accuracy : {score[1]:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=-1)

print("\n" + classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ─────────────────────────────────────────────
# 11. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\nSaving confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()
print("  Saved → confusion_matrix.png")

# ─────────────────────────────────────────────
# 12. SAVE THE MODEL
# ─────────────────────────────────────────────
print("\nSaving model...")

model.save('pcb_defect_classifier.keras')
print("  Saved → pcb_defect_classifier.keras")

print("\n" + "=" * 50)
print("Training complete!")
print("  sample_patches.png")
print("  training_curves.png")
print("  confusion_matrix.png")
print("  pcb_defect_classifier.keras")
print("=" * 50)