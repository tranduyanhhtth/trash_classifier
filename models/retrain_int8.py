#!/usr/bin/env python3
"""
retrain_int8.py  –  Rebuild + Fine-tune → Full INT8 for TrashNet
=================================================================
Bám sát notebook gốc (pruning-and-quantization-in-keras.ipynb):
  - preprocess_input (MobileNet chuẩn [-1,1]) thay vì rescale=1/255
  - Augmentation mạnh hơn (rotation_range=40, shear, shift, zoom)
  - 2-phase training:
      Phase 1: frozen base, train Dense(6) head  (~15 epochs)
      Phase 2: unfreeze all, fine-tune toàn bộ   (~15 epochs)
  - Representative dataset dùng preprocess_input để calibrate đúng

Tại sao preprocess_input quan trọng:
  MobileNetV1 imagenet weights được train với input [-1,1].
  Dùng rescale=1/255 ([0,1]) → "out-of-distribution" → accuracy thấp hơn ~20-25%.

ESP32 preprocessing:
  uint8 [0,255] → int8 [-128,127] via XOR 0x80 (= pixel - 128)
  Với model [-1,1]: scale≈0.00784, zp=0 → float = int8 * 0.00784 ≈ [-1,1] ✅
  (Xem tính toán trong TRAINING.md)

Usage:
    cd trash_classifier/models
    python3 retrain_int8.py
"""

import os, sys, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    sys.exit("ERROR: pip install tensorflow")

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: pip install pillow")

from tensorflow.keras.applications.mobilenet import preprocess_input

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_DIR       = "training/dataset-resized/dataset-resized"
OUTPUT_MODEL      = "trash_model_int8.tflite"
IMAGE_SIZE        = (224, 224)
BATCH_SIZE        = 16       # khớp notebook gốc (16, không phải 32)
EPOCHS_HEAD       = 15       # Phase 1: train chỉ Dense(6) head
EPOCHS_FINETUNE   = 15       # Phase 2: fine-tune toàn bộ (tăng từ 5 lên 15)
NUM_CALIB_SAMPLES = 300      # calibrate INT8 activations

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

print(f"\nDataset : {DATASET_DIR}")
print(f"Output  : {OUTPUT_MODEL}")
print(f"Preprocessing: preprocess_input (MobileNet [-1, 1])\n")

# ── Verify dataset ─────────────────────────────────────────────────────────
if not os.path.isdir(DATASET_DIR):
    sys.exit(f"ERROR: dataset not found: {DATASET_DIR}\n"
             f"  Run: cd training && unzip dataset-resized.zip")

for cls in CLASSES:
    path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(path):
        sys.exit(f"ERROR: missing class dir: {path}")
    n = len([f for f in os.listdir(path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {cls:12s}: {n} images")

# ── Data pipeline ──────────────────────────────────────────────────────────
print("\nBuilding data pipeline...")

# Training: augmentation mạnh (khớp notebook gốc)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ← QUAN TRỌNG: [-1,1] cho MobileNet
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=40,          # notebook: 40
    shear_range=0.2,            # notebook: 0.2
    zoom_range=0.2,             # notebook: 0.2
    width_shift_range=0.2,      # notebook: 0.2
    height_shift_range=0.2,     # notebook: 0.2
    fill_mode='nearest',        # notebook: 'nearest'
)

# Validation: chỉ preprocess (không augment)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True,
)
val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
)

print(f"  Train: {train_gen.samples} images  ({len(train_gen)} batches)")
print(f"  Val  : {val_gen.samples} images  ({len(val_gen)} batches)")
print(f"  Classes: {train_gen.class_indices}")

# ── Build model (khớp notebook gốc) ───────────────────────────────────────
print("\nBuilding model (MobileNetV1 + Flatten + Dense(6))...")

base = tf.keras.applications.MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=IMAGE_SIZE + (3,),
)
base.trainable = False  # Phase 1: freeze

model = tf.keras.Sequential([
    base,
    tf.keras.layers.Flatten(),          # khớp notebook (không dùng GlobalAvgPool)
    tf.keras.layers.Dense(6, activation='softmax'),
], name="trashnet")

model.summary(line_length=72)

# ── Phase 1: Train chỉ Dense(6) head ──────────────────────────────────────
print(f"\nPhase 1: Training Dense(6) head only ({EPOCHS_HEAD} epochs, lr=1e-3)...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

cb_head = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1),
]

h1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_HEAD, callbacks=cb_head, verbose=1,
)
val_acc_head = max(h1.history['val_accuracy'])
print(f"\n✓ Phase 1 done. Best val accuracy: {val_acc_head:.4f}")

# ── Phase 2: Fine-tune toàn bộ model ──────────────────────────────────────
if EPOCHS_FINETUNE > 0:
    print(f"\nPhase 2: Fine-tuning all layers ({EPOCHS_FINETUNE} epochs, lr=1e-5)...")
    print(f"  (Khớp notebook: Adam lr=1e-5 cho fine-tune)")
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # khớp notebook lr
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    cb_ft = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FINETUNE, callbacks=cb_ft, verbose=1,
    )
    val_acc_ft = max(h2.history['val_accuracy'])
    print(f"\n✓ Phase 2 done. Best val accuracy: {val_acc_ft:.4f}")

# ── Representative dataset (dùng preprocess_input) ────────────────────────
print(f"\nBuilding calibration dataset ({NUM_CALIB_SAMPLES} samples, preprocess_input)...")

calib_paths = []
for cls in CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls)
    imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('.')]
    calib_paths.extend(imgs)

np.random.shuffle(calib_paths)
calib_paths = calib_paths[:NUM_CALIB_SAMPLES]

def representative_dataset():
    """
    Calibration data dùng cùng preprocessing với training:
      preprocess_input: [0,255] → [-1, 1]
    """
    for p in calib_paths:
        img = Image.open(p).convert("RGB").resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)   # [0,255] → [-1,1], KHỚP với training
        yield [arr[np.newaxis, ...]]  # shape: [1, 224, 224, 3]

# ── Convert → full INT8 ───────────────────────────────────────────────────
print("\nConverting to full INT8 TFLite (calibrating with representative data)...")
print("Expected input quantization: scale≈0.00784, zp≈0 (MobileNet [-1,1] range)")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(OUTPUT_MODEL, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"\n✓ Saved: {OUTPUT_MODEL}  ({size_kb:.0f} KB)")

# ── Verify output ──────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings('ignore')
interp = tf.lite.Interpreter(model_path=OUTPUT_MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print(f"\nVerification:")
print(f"  Input  : {inp['shape'].tolist()}  {inp['dtype'].__name__}  "
      f"scale={inp['quantization'][0]:.6f}  zp={inp['quantization'][1]}")
print(f"  Output : {out['shape'].tolist()}  {out['dtype'].__name__}  "
      f"scale={out['quantization'][0]:.6f}  zp={out['quantization'][1]}")

zp = inp['quantization'][1]
scale = inp['quantization'][0]
print(f"\nESP32 XOR 0x80 mapping verification:")
print(f"  pixel=0  → int8={0^0x80-256}  → float={(0^0x80-256 - zp)*scale:.4f}  (preprocess_input(0)=-1.0)")
print(f"  pixel=128→ int8={128^0x80}    → float={(128^0x80 - zp)*scale:.4f}  (preprocess_input(128)≈0.0)")
print(f"  pixel=255→ int8={255^0x80-256}→ float={(255^0x80-256 - zp)*scale:.4f}  (preprocess_input(255)=1.0)")

if inp['dtype'] == np.int8 and out['dtype'] == np.int8:
    print("\n✓ SUCCESS: Full INT8 – compatible with TFLite Micro esp_nn!")
    print("\nNext steps:")
    print(f"  1. python3 models/check_ops.py          ← verify op coverage")
    print(f"  2. bash models/generate_model_cc.sh {OUTPUT_MODEL}")
    print(f"  3. cd .. && idf build && idf flash")
else:
    print(f"\n⚠ WARNING: Not full INT8 – input={inp['dtype']}, output={out['dtype']}")
