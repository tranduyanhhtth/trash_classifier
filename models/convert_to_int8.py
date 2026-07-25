#!/usr/bin/env python3
"""
convert_to_int8.py  –  Full Integer Quantization for TrashNet
==============================================================
Converts hybrid-quantized (FLOAT32 I/O) model to full-INT8
required by TFLite Micro esp_nn kernels on ESP32-S3.

WHY:
  Hybrid model: weights INT8, I/O FLOAT32
  → "Hybrid models are not supported on TFLite Micro" (esp_nn/conv.cc)

  Full INT8 model: weights + activations + I/O all INT8
  → Works on TFLite Micro, faster inference, smaller arena

USAGE:
  # With real images (best accuracy):
  python3 convert_to_int8.py \\
      --model quantized_and_pruned_model.tflite \\
      --images training/dataset-resized/dataset-resized \\
      --output trash_model_int8.tflite

  # With random data (quick test, ~2% accuracy drop):
  python3 convert_to_int8.py \\
      --model quantized_and_pruned_model.tflite \\
      --random --output trash_model_int8.tflite

AFTER conversion:
  bash generate_model_cc.sh trash_model_int8.tflite
"""

import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="quantized_and_pruned_model.tflite")
parser.add_argument("--images",      default=None)
parser.add_argument("--random",      action="store_true")
parser.add_argument("--output",      default="trash_model_int8.tflite")
parser.add_argument("--num-samples", type=int, default=200)
args = parser.parse_args()

# ── TensorFlow ─────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("ERROR: pip install tensorflow")
    sys.exit(1)

# ── Load + inspect input model ─────────────────────────────────────────────
if not os.path.exists(args.model):
    print(f"ERROR: {args.model} not found")
    sys.exit(1)

print(f"\nLoading: {args.model}  ({os.path.getsize(args.model)/1024:.0f} KB)")
interp = tf.lite.Interpreter(model_path=args.model)
interp.allocate_tensors()
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"  Input  : {in_det['shape'].tolist()}  {in_det['dtype'].__name__}")
print(f"  Output : {out_det['shape'].tolist()}  {out_det['dtype'].__name__}")

if in_det['dtype'] == np.int8:
    print("\nModel already has INT8 input. Checking output...")
    if out_det['dtype'] == np.int8:
        print("✓ Model is already full INT8. Nothing to do.")
        sys.exit(0)

# ── Representative dataset ─────────────────────────────────────────────────
INPUT_SHAPE = (224, 224, 3)

if args.random:
    print(f"\nUsing RANDOM calibration ({args.num_samples} samples).")
    print("Warning: real images give better quantization accuracy.")
    def representative_dataset():
        for _ in range(args.num_samples):
            yield [np.random.rand(1, *INPUT_SHAPE).astype(np.float32)]
else:
    if not args.images:
        print("ERROR: provide --images <dir>  or  --random")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: pip install pillow")
        sys.exit(1)

    paths = []
    for root, _, files in os.walk(args.images):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, f))
    if not paths:
        print(f"ERROR: no images found in {args.images}")
        sys.exit(1)

    np.random.shuffle(paths)
    paths = paths[:args.num_samples]
    print(f"\nUsing {len(paths)} real images from {args.images}")

    def representative_dataset():
        for p in paths:
            img = Image.open(p).convert("RGB").resize(
                (INPUT_SHAPE[1], INPUT_SHAPE[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            yield [arr[np.newaxis, ...]]

# ── Convert ────────────────────────────────────────────────────────────────
print("\nConverting to full INT8 (may take 1-5 min)...")

with open(args.model, 'rb') as f:
    model_bytes = f.read()

converter = tf.lite.TFLiteConverter.from_tflite_model_content(model_bytes)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# ── Save ───────────────────────────────────────────────────────────────────
with open(args.output, 'wb') as f:
    f.write(tflite_model)
print(f"\n✓ Saved: {args.output}  ({len(tflite_model)/1024:.0f} KB)")

# ── Verify ─────────────────────────────────────────────────────────────────
interp2 = tf.lite.Interpreter(model_path=args.output)
interp2.allocate_tensors()
in2  = interp2.get_input_details()[0]
out2 = interp2.get_output_details()[0]
print(f"\nVerifying output:")
print(f"  Input  : {in2['shape'].tolist()}  {in2['dtype'].__name__}")
print(f"  Output : {out2['shape'].tolist()}  {out2['dtype'].__name__}")

if in2['dtype'] == np.int8 and out2['dtype'] == np.int8:
    print("\n✓ SUCCESS: Full INT8 – compatible with TFLite Micro esp_nn!")
    print(f"\nNext step:")
    print(f"  bash generate_model_cc.sh {args.output}")
else:
    print("\n⚠ WARNING: still FLOAT I/O – may not work on TFLite Micro")
    print(f"  Input dtype : {in2['dtype'].__name__}")
    print(f"  Output dtype: {out2['dtype'].__name__}")
