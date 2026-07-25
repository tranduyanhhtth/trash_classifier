#!/usr/bin/env python3
"""
retrain_int8.py  –  Rebuild + Fine-tune → Full INT8 for TrashNet
=================================================================
Vì model gốc (.h5) đã mất (notebook dùng tempfile), script này:
1. Load MobileNetV1 với imagenet weights (frozen)
2. Thêm Flatten + Dense(6) head
3. Fine-tune chỉ head ~15 epochs (nhanh, ~5-10 phút)
4. Convert thẳng sang full INT8 (không qua hybrid)

Usage:
    cd trash_classifier/models
    python3 retrain_int8.py

Output: trash_model_int8.tflite  (nhúng vào firmware bằng generate_model_cc.sh)
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
except ImportError:
    sys.exit("ERROR: pip install tensorflow")

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: pip install pillow")

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_DIR   = "training/dataset-resized/dataset-resized"
OUTPUT_MODEL  = "trash_model_int8.tflite"
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_HEAD   = 15   # train only Dense(6), fast
EPOCHS_FINETUNE = 5  # optional: fine-tune all layers, slow but better accuracy
NUM_CALIB_SAMPLES = 300  # for INT8 calibration

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

print(f"\nDataset : {DATASET_DIR}")
print(f"Output  : {OUTPUT_MODEL}")

# ── Verify dataset ─────────────────────────────────────────────────────────
if not os.path.isdir(DATASET_DIR):
    sys.exit(f"ERROR: dataset not found: {DATASET_DIR}")

for cls in CLASSES:
    path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(path):
        sys.exit(f"ERROR: missing class dir: {path}")
    n = len([f for f in os.listdir(path)
             if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"  {cls:12s}: {n} images")

# ── Data pipeline ──────────────────────────────────────────────────────────
print("\nBuilding data pipeline...")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True,
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
)

print(f"  Train batches : {len(train_gen)}  ({train_gen.samples} images)")
print(f"  Val   batches : {len(val_gen)}  ({val_gen.samples} images)")
print(f"  Class indices : {train_gen.class_indices}")

# ── Build model (same architecture as notebook) ────────────────────────────
print("\nBuilding model (MobileNetV1 + Dense(6))...")

base = tf.keras.applications.MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=IMAGE_SIZE + (3,),
)
base.trainable = False   # freeze for head training

model = tf.keras.Sequential([
    base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='softmax'),
], name="trashnet")

model.summary(line_length=72)

# ── Phase 1: Train only Dense(6) head ─────────────────────────────────────
print(f"\nPhase 1: Training head only ({EPOCHS_HEAD} epochs)...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1),
]

h1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_HEAD, callbacks=cb, verbose=1,
)

val_acc_head = max(h1.history['val_accuracy'])
print(f"\n✓ Head training done. Best val accuracy: {val_acc_head:.4f}")

# ── Phase 2: Fine-tune all layers (optional but improves accuracy) ─────────
if EPOCHS_FINETUNE > 0:
    print(f"\nPhase 2: Fine-tuning all layers ({EPOCHS_FINETUNE} epochs)...")
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # lower LR for fine-tune
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FINETUNE, callbacks=cb, verbose=1,
    )
    val_acc_ft = max(h2.history['val_accuracy'])
    print(f"\n✓ Fine-tune done. Best val accuracy: {val_acc_ft:.4f}")

# ── Representative dataset for INT8 calibration ───────────────────────────
print(f"\nBuilding calibration dataset ({NUM_CALIB_SAMPLES} samples)...")

calib_paths = []
for cls in CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls)
    imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))]
    calib_paths.extend(imgs)

np.random.shuffle(calib_paths)
calib_paths = calib_paths[:NUM_CALIB_SAMPLES]

def representative_dataset():
    for p in calib_paths:
        img = Image.open(p).convert("RGB").resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        yield [arr[np.newaxis, ...]]

# ── Convert to full INT8 ───────────────────────────────────────────────────
print("\nConverting to full INT8 (calibrating activations)...")
print("This may take 2-5 minutes...")

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
interp = tf.lite.Interpreter(model_path=OUTPUT_MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"\nVerification:")
print(f"  Input  : {inp['shape'].tolist()}  {inp['dtype'].__name__}  "
      f"scale={inp['quantization'][0]:.6f}  zp={inp['quantization'][1]}")
print(f"  Output : {out['shape'].tolist()}  {out['dtype'].__name__}  "
      f"scale={out['quantization'][0]:.6f}  zp={out['quantization'][1]}")

if inp['dtype'] == np.int8 and out['dtype'] == np.int8:
    print("\n✓ SUCCESS: Full INT8 – compatible with TFLite Micro esp_nn!")
    print("\nNext step:")
    print(f"  bash generate_model_cc.sh {OUTPUT_MODEL}")
else:
    print(f"\n⚠ WARNING: Still FLOAT I/O – not compatible with esp_nn kernels")
