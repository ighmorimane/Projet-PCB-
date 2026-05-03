import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras

# ─────────────────────────────────────────────
# CONFIGURATION — change these values
# ─────────────────────────────────────────────

# Path to the saved trained model
MODEL_PATH = "/srv/groups/group7/projects/pcb_defect_classifier.keras"

# Path to the new PCB image you want to test
IMAGE_PATH = "/srv/groups/group7/data/pcb-defects/PCB_DATASET/images/Short/01_short_03.jpg"  # ← change this

# Bounding box of the defect region you want to test
# Format: (xmin, ymin, xmax, ymax)
# How to find these values:
#   - Open the image in VS Code
#   - Hover over the defect area to get pixel coordinates
XMIN = 787  # ← change this
YMIN = 1206 # ← change this
XMAX = 841  # ← change this
YMAX = 1263 # ← change this

# Output folder for the prediction result image
OUTPUT_DIR = Path("/srv/groups/group7/projects/")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CLASSES    = ['missing_hole', 'mouse_bite', 'open_circuit',
              'short', 'spur', 'spurious_copper']
PATCH_SIZE = (100, 100)

# ─────────────────────────────────────────────
# 1. LOAD THE MODEL
# ─────────────────────────────────────────────
print("=" * 50)
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"  Model loaded from: {MODEL_PATH}")

# ─────────────────────────────────────────────
# 2. LOAD THE IMAGE
# ─────────────────────────────────────────────
print("\nLoading image...")
image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
print(f"  Image shape : {image.shape}")
print(f"  Image path  : {IMAGE_PATH}")

# ─────────────────────────────────────────────
# 3. CROP THE PATCH
# ─────────────────────────────────────────────
print("\nCropping patch...")
patch = image[YMIN:YMAX, XMIN:XMAX]
print(f"  Raw patch shape : {patch.shape}")

patch_resized = cv2.resize(patch, PATCH_SIZE)
print(f"  Resized to      : {patch_resized.shape}")

# ─────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────
print("\nPredicting...")

# Add batch dimension: (100,100,3) → (1,100,100,3)
inp   = np.expand_dims(patch_resized, axis=0)
probs = model.predict(inp, verbose=0)[0]  # shape: (6,)

pred_index = np.argmax(probs)
pred_class = CLASSES[pred_index]
confidence = probs[pred_index] * 100

# ─────────────────────────────────────────────
# 5. PRINT RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)
print(f"  Predicted class : {pred_class}")
print(f"  Confidence      : {confidence:.2f}%")
print("\nAll class probabilities:")
for cls, prob in zip(CLASSES, probs):
    bar = "█" * int(prob * 30)
    print(f"  {cls:<20} {prob*100:6.2f}%  {bar}")
print("=" * 50)

# ─────────────────────────────────────────────
# 6. SAVE RESULT IMAGE
# ─────────────────────────────────────────────
print("\nSaving result image...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: full image with bounding box drawn
axes[0].imshow(image)
import matplotlib.patches as mpatches
rect = mpatches.Rectangle(
    (XMIN, YMIN), XMAX - XMIN, YMAX - YMIN,
    linewidth=2, edgecolor='red', facecolor='none'
)
axes[0].add_patch(rect)
axes[0].set_title('Full PCB Image\n(red box = tested region)')
axes[0].axis('off')

# Right: the cropped patch with prediction
axes[1].imshow(patch_resized)
axes[1].set_title(f'Patch\nPredicted: {pred_class}\nConfidence: {confidence:.1f}%')
axes[1].axis('off')

plt.tight_layout()
output_path = OUTPUT_DIR / 'prediction_result.png'
plt.savefig(output_path, dpi=100)
plt.close()
print(f"  Saved → {output_path}")
print("\nDone!")
