#!/usr/bin/env python3
"""
check_ops.py – Kiểm tra xem main_functions.cc đã đăng ký đủ ops cho model chưa.

Usage:
    cd trash_classifier
    python3 models/check_ops.py

Exit code:
    0 = tất cả ops đã đăng ký → có thể build/flash
    1 = thiếu ops            → phải sửa main_functions.cc trước
"""

import os, re, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "trash_model_int8.tflite")
CC_PATH     = os.path.join(SCRIPT_DIR, "../main/main_functions.cc")

if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: Model not found: {MODEL_PATH}\n"
             f"  Run: python3 models/retrain_int8.py")
if not os.path.exists(CC_PATH):
    sys.exit(f"ERROR: Source file not found: {CC_PATH}")

# ── 1. Extract ops from model ─────────────────────────────────────────────────
try:
    from tensorflow.lite.python import schema_py_generated as schema_fb
except ImportError:
    sys.exit("ERROR: pip install tensorflow")

with open(MODEL_PATH, 'rb') as f:
    buf = bytearray(f.read())

model      = schema_fb.ModelT.InitFromPackedBuf(buf, 0)
enum_names = {v: k for k, v in vars(schema_fb.BuiltinOperator).items()
              if isinstance(v, int)}

model_ops = sorted({
    enum_names[model.operatorCodes[op.opcodeIndex].builtinCode]
    for sg in model.subgraphs for op in sg.operators
})

# ── 2. Extract registered ops from main_functions.cc ─────────────────────────
cc_source        = open(CC_PATH).read()
registered_calls = re.findall(r'micro_op_resolver\.Add(\w+)\(\)', cc_source)

def pascal_to_upper(s):
    """Conv2D → CONV_2D, DepthwiseConv2D → DEPTHWISE_CONV_2D, etc."""
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', s)  # CamelCase → Snake_Case
    s = re.sub(r'(?<=[a-z])(?=\d)',    '_', s)  # conv2 → conv_2
    return s.upper()

registered_ops = sorted([pascal_to_upper(c) for c in registered_calls])

# ── 3. Compare ────────────────────────────────────────────────────────────────
model_set      = set(model_ops)
registered_set = set(registered_ops)
missing        = model_set - registered_set
extra          = registered_set - model_set

count_match    = re.search(r'MicroMutableOpResolver<(\d+)>', cc_source)
resolver_cap   = int(count_match.group(1)) if count_match else 0

# ── 4. Report ─────────────────────────────────────────────────────────────────
print(f"Model : {os.path.basename(MODEL_PATH)}")
print(f"Source: main/main_functions.cc")
print()
print("=== Op coverage ===")
for op in model_ops:
    status = "✅" if op in registered_set else "❌ MISSING"
    print(f"  {status:12s} {op}")

if extra:
    print()
    print("=== Registered but not in model (harmless, can remove) ===")
    for op in sorted(extra):
        print(f"  ⚠️  {op}")

print()
print(f"MicroMutableOpResolver capacity : <{resolver_cap}>")
print(f"Ops registered                  : {len(registered_calls)}")

cap_ok = resolver_cap >= len(registered_calls)
print(f"Capacity                        : {'✅ OK' if cap_ok else f'❌ INCREASE to <{len(registered_calls)}>'}")

print()
if not missing and cap_ok:
    print("✅ ALL OPS COVERED – safe to build and flash!")
    sys.exit(0)
else:
    if missing:
        print(f"❌ {len(missing)} op(s) MISSING: {sorted(missing)}")
        print("   Fix: add to MicroMutableOpResolver in main/main_functions.cc:")
        for op in sorted(missing):
            # Guess the AddXxx() call name
            func = "Add" + op.replace("_", "").title().replace("2D", "2D")
            print(f"   micro_op_resolver.{func}();")
    if not cap_ok:
        print(f"❌ Resolver capacity too small: increase <{resolver_cap}> → <{len(registered_calls)}>")
    sys.exit(1)
