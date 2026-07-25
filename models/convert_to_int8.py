#!/usr/bin/env python3
"""
convert_to_int8.py  –  Full Integer Quantization for TrashNet
==============================================================
Converts the hybrid-quantized (FLOAT32 I/O) model to a proper
full-integer-quantized (INT8 I/O) model required by TFLite Micro.

WHY is this needed?
  Your current model (quantized_and_pruned_model.tflite) has:
    Input  : [1, 224, 224, 3]  FLOAT32  ← TFLite Micro cannot handle this
    Output : [1, 6]            FLOAT32  ← same issue

  Full-INT8 quantization requires a representative dataset to calibrate
  the activation scale/zero_point for every layer.

REQUIREMENTS:
  pip install tensorflow>=2.10 numpy pillow

USAGE:
  python3 convert_to_int8.py \
      --model quantized_and_pruned_model.tflite \
      --images /path/to/trashnet/dataset \
      --output trash_model_int8.tflite

  OR (minimal, uses random data as calibration – accuracy may suffer slightly):
  python3 convert_to_int8.py --model quantized_and_pruned_model.tflite --random

AFTER conversion:
  xxd -i trash_model_int8.tflite > main/trash_model_data.cc
  # Then manually rename array to g_trash_model_data and length to g_trash_model_data_len
"""

import argparse
import os
import sys
import numpy as np

# ── 1. Parse arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Full INT8 quantization for TrashNet")
parser.add_argument("--model",  default="quantized_and_pruned_model.tflite",
                    help="Input TFLite model (hybrid or float)")
parser.add_argument("--images", default=None,
                    help="Path to representative image directory (TrashNet structure)")
parser.add_argument("--random", action="store_true",
                    help="Use random data for calibration (faster, slight accuracy drop)")
parser.add_argument("--output", default="trash_model_int8.tflite",
                    help="Output INT8 TFLite model filename")
parser.add_argument("--num-samples", type=int, default=200,
                    help="Number of calibration samples (default 200)")
args = parser.parse_args()

# ── 2. Import TensorFlow ───────────────────────────────────────────────────
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("ERROR: TensorFlow not found. Install with:\n  pip install tensorflow")
    sys.exit(1)

# ── 3. Load model ──────────────────────────────────────────────────────────
if not os.path.exists(args.model):
    print(f"ERROR: Model file not found: {args.model}")
    sys.exit(1)

print(f"Loading model: {args.model}  ({os.path.getsize(args.model)/1024:.0f} KB)")

# Check current model I/O types
interp = tf.lite.Interpreter(model_path=args.model)
interp.allocate_tensors()
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"  Input  : shape={in_det['shape'].tolist()}, dtype={in_det['dtype'].__name__}")
print(f"  Output : shape={out_det['shape'].tolist()}, dtype={out_det['dtype'].__name__}")

# ── 4. Build representative dataset generator ──────────────────────────────
INPUT_SHAPE = (224, 224, 3)  # H × W × C

if args.random:
    print(f"\nUsing RANDOM calibration data ({args.num_samples} samples)")
    print("Warning: accuracy may be 2-5% lower than real-image calibration.")

    def representative_dataset():
        for _ in range(args.num_samples):
            # Random image normalised to [0, 1] as float32
            img = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
            yield [img]

else:
    if not args.images:
        print("ERROR: Provide --images <path> or use --random for calibration.")
        sys.exit(1)
    if not os.path.isdir(args.images):
        print(f"ERROR: Image directory not found: {args.images}")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not found. Install with:\n  pip install pillow")
        sys.exit(1)

    # Collect image paths (TrashNet has subdirs per class)
    image_paths = []
    for root, dirs, files in os.walk(args.images):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, fname))

    if not image_paths:
        print(f"ERROR: No images found in {args.images}")
        sys.exit(1)

    # Shuffle and cap
    np.random.shuffle(image_paths)
    image_paths = image_paths[:args.num_samples]
    print(f"\nUsing {len(image_paths)} real images from {args.images}")

    def representative_dataset():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0,1]
            arr = arr[np.newaxis, ...]                        # add batch dim
            yield [arr]

# ── 5. Convert with full INT8 quantization ─────────────────────────────────
print("\nConverting to full INT8 quantization...")
print("(This may take 1-5 minutes depending on model size and dataset)")

converter = tf.lite.TFLiteConverter.from_saved_model(args.model) \
    if args.model.endswith('.pb') else \
    tf.lite.TFLiteConverter.from_tflite_model(tf.lite.Interpreter(model_path=args.model))

# Actually we need to use the file path API directly
converter = tf.lite.TFLiteConverter.from_tflite_model_file(args.model) \
    if hasattr(tf.lite.TFLiteConverter, 'from_tflite_model_file') else None

# Fallback: use Interpreter-based approach
if converter is None:
    # Read model bytes and use from_keras_model if possible, else from_tflite_model_content
    with open(args.model, 'rb') as f:
        model_content = f.read()

    converter = tf.lite.TFLiteConverter.from_tflite_model_content(model_content) \
        if hasattr(tf.lite.TFLiteConverter, 'from_tflite_model_content') else None

    if converter is None:
        print("Attempting alternative converter path...")
        # Load as flatbuffer and reconstruct
        converter = tf.lite.TFLiteConverter.experimental_from_tflite_model(model_content)

# Configure quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force INT8 for ALL tensors including inputs and outputs
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8   # input  tensor → INT8
converter.inference_output_type = tf.int8   # output tensor → INT8

# Convert
tflite_quant_model = converter.convert()

# ── 6. Save output ────────────────────────────────────────────────────────
with open(args.output, 'wb') as f:
    f.write(tflite_quant_model)

out_size = os.path.getsize(args.output)
print(f"\n✓ Saved: {args.output}  ({out_size/1024:.0f} KB)")

# ── 7. Verify output model ─────────────────────────────────────────────────
print("\nVerifying output model:")
interp2 = tf.lite.Interpreter(model_path=args.output)
interp2.allocate_tensors()
in2  = interp2.get_input_details()[0]
out2 = interp2.get_output_details()[0]
print(f"  Input  : shape={in2['shape'].tolist()}, dtype={in2['dtype'].__name__}")
print(f"  Output : shape={out2['shape'].tolist()}, dtype={out2['dtype'].__name__}")

if in2['dtype'] == np.int8 and out2['dtype'] == np.int8:
    print("\n✓ SUCCESS: Model is fully INT8-quantized and ready for TFLite Micro!")
    print(f"\nNext steps:")
    print(f"  xxd -i {args.output} > main/trash_model_data.cc")
    print(f"  # Edit main/trash_model_data.cc:")
    print(f"  #   rename array to:  g_trash_model_data[]")
    print(f"  #   rename length to: g_trash_model_data_len")
    print(f"  #   add at top:       #include \"trash_model_data.h\"")
else:
    print(f"\n⚠ WARNING: Output model still has FLOAT I/O.")
    print(f"  Input  dtype: {in2['dtype'].__name__}")
    print(f"  Output dtype: {out2['dtype'].__name__}")
    print(f"  This may not be compatible with TFLite Micro.")
    print(f"  Consider retraining the model from scratch with TF2 quantization-aware training.")
