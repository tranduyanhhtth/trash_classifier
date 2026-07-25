# Hướng dẫn Training & Nhúng Model – TrashNet ESP32-S3

Tài liệu này mô tả toàn bộ quá trình từ **dataset thô → model full INT8 → firmware chạy được trên ESP32-S3**.

---

## Tại sao cần retrain?

Model gốc `quantized_and_pruned_model.tflite` từ notebook là **hybrid quantized**:
- Weights: INT8
- I/O (input/output): FLOAT32

ESP32 esp_nn kernels **chỉ hỗ trợ full INT8** (tất cả đều INT8). Chạy hybrid model sẽ gặp lỗi ngay tại `AllocateTensors()`:

```
Hybrid models are not supported on TFLite Micro.
Node CONV_2D (number 2) failed to prepare with status 1
```

**Full INT8 model** yêu cầu:
- Weights: INT8 ✓
- Input tensor: INT8 ✓  
- Output tensor: INT8 ✓
- Mọi activation buffer: INT8 ✓

---

## Yêu cầu môi trường

```bash
# Python 3.10+ (đang dùng 3.12)
python3 --version

# Cài thư viện (một lần)
pip install tensorflow pillow numpy --break-system-packages

# Kiểm tra
python3 -c "import tensorflow as tf; print(tf.__version__)"
# → 2.21.0 hoặc mới hơn
```

---

## Cấu trúc thư mục

```
trash_classifier/models/
├── TRAINING.md                          ← file này
├── retrain_int8.py                      ← script chính: train + convert INT8
├── convert_to_int8.py                   ← convert từ float model (nếu có sẵn)
├── generate_model_cc.sh                 ← nhúng .tflite → C array cho firmware
├── quantized_and_pruned_model.tflite    ← model gốc (hybrid, KHÔNG dùng trực tiếp)
├── trash_model_int8.tflite              ← OUTPUT: model full INT8 (gitignored)
└── training/
    ├── pruning-and-quantization-in-keras.ipynb  ← notebook gốc (tham khảo)
    ├── dataset-resized.zip                      ← archive dataset (43MB, gitignored)
    └── dataset-resized/
        └── dataset-resized/
            ├── cardboard/   (403 ảnh)
            ├── glass/       (501 ảnh)
            ├── metal/       (410 ảnh)
            ├── paper/       (594 ảnh)
            ├── plastic/     (482 ảnh)
            └── trash/       (137 ảnh)
```

---

## Bước 1: Chuẩn bị dataset

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

# Nếu chưa giải nén:
cd training
unzip dataset-resized.zip
cd ..

# Kiểm tra cấu trúc dataset:
ls training/dataset-resized/dataset-resized/
# → cardboard  glass  metal  paper  plastic  trash

# Đếm ảnh mỗi class:
for cls in cardboard glass metal paper plastic trash; do
    count=$(find training/dataset-resized/dataset-resized/$cls -name "*.jpg" | wc -l)
    echo "  $cls: $count"
done
```

Kết quả mong đợi:
```
  cardboard: 403
  glass: 501
  metal: 410
  paper: 594
  plastic: 482
  trash: 137
```

---

## Bước 2: Chạy retrain + convert INT8

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

python3 retrain_int8.py
```

Script thực hiện tuần tự:

| Phase | Mô tả | Thời gian |
|-------|--------|-----------|
| Build model | MobileNetV1 (imagenet) + Flatten + Dense(6) | <1 phút |
| Phase 1 | Train chỉ Dense(6) head, MobileNet frozen | ~5–8 phút |
| Phase 2 | Fine-tune toàn bộ model với LR thấp | ~8–12 phút |
| INT8 convert | Calibrate activations với 300 ảnh | ~2–3 phút |
| **Tổng** | | **~15–23 phút** |

### Tuỳ chỉnh `retrain_int8.py` (nếu cần)

Mở file và chỉnh các biến ở đầu:

```python
DATASET_DIR    = "training/dataset-resized/dataset-resized"  # đường dẫn dataset
OUTPUT_MODEL   = "trash_model_int8.tflite"                   # tên file output
IMAGE_SIZE     = (224, 224)   # kích thước input model
BATCH_SIZE     = 32           # giảm nếu RAM máy tính không đủ
EPOCHS_HEAD    = 15           # epochs train head (giảm → nhanh hơn, kém hơn)
EPOCHS_FINETUNE= 5            # epochs fine-tune toàn bộ (0 = bỏ qua)
NUM_CALIB_SAMPLES = 300       # số ảnh calibrate INT8 (tăng → chính xác hơn)
```

### Log thành công

```
TensorFlow 2.21.0 loaded

Dataset : training/dataset-resized/dataset-resized
  cardboard   : 403 images
  glass       : 501 images
  ...

Phase 1: Training head only (15 epochs)...
Epoch 1/15
64/64 ━━━━━━━━━━━━━━━━━━━━ 45s - accuracy: 0.6823 - val_accuracy: 0.7201
...
✓ Head training done. Best val accuracy: 0.9124

Phase 2: Fine-tuning all layers (5 epochs)...
...
✓ Fine-tune done. Best val accuracy: 0.9312

Converting to full INT8 (calibrating activations)...
✓ Saved: trash_model_int8.tflite  (3735 KB)

Verification:
  Input  : [1, 224, 224, 3]  int8  scale=0.003922  zp=-128
  Output : [1, 6]            int8  scale=0.003906  zp=-128

✓ SUCCESS: Full INT8 – compatible with TFLite Micro esp_nn!
```

> **Quan trọng:** Phải thấy `int8` ở cả Input và Output. Nếu thấy `float32` → có lỗi trong quá trình convert.

---

## Bước 3: Nhúng model vào firmware

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

bash generate_model_cc.sh trash_model_int8.tflite
```

Script tự động:
1. Chạy `xxd -i trash_model_int8.tflite` để tạo C hex array
2. Đổi tên symbol thành `g_trash_model_data` và `g_trash_model_data_len`
3. Thêm `#include "trash_model_data.h"` header
4. Ghi ra `../main/trash_model_data.cc`

```
Converting trash_model_int8.tflite → ../main/trash_model_data.cc ...
  Raw array name: trash_model_int8_tflite
  Raw length var: trash_model_int8_tflite_len
✓ Done! Written to: ../main/trash_model_data.cc

File size check:
3824888 trash_model_int8.tflite
```

Kiểm tra file output:

```bash
# Số dòng (318K dòng cho model ~3.7MB)
wc -l ../main/trash_model_data.cc

# Đầu file: phải có include và khai báo array
head -8 ../main/trash_model_data.cc
# →  #include "trash_model_data.h"
# →  const unsigned char g_trash_model_data[] = {

# Cuối file: kích thước array
tail -3 ../main/trash_model_data.cc
# →  const unsigned int g_trash_model_data_len = 3824888;
```

---

## Bước 4: Build firmware

```bash
cd /home/danz/Downloads/trash/trash_classifier

# Load ESP-IDF
source idf_env.sh
# → ✓ ESP-IDF v5.5.1 loaded (ESP32-S3 N16R8). Use 'idf' command.

# (Chỉ cần lần đầu hoặc sau khi đổi sdkconfig.defaults)
rm -f sdkconfig

# Build
idf build
```

Kết quả build thành công:

```
[x/x] Generating binary image from built executable
Generated /home/danz/.../build/trash_classifier.bin
trash_classifier.bin binary size 0x4A0000 bytes.
Smallest app partition is 0x500000 bytes. 0x60000 bytes (12%) free.
Project build complete.
```

> **Partition:** OTA slot đã được tăng lên **5MB (0x500000)** → ~12% headroom.
> Nếu gặp `Warning: Nearly full` → tăng tiếp trong `partitions.csv`.

---

## Bước 5: Flash lên ESP32-S3

### Flash toàn bộ (lần đầu hoặc sau khi đổi partition)

Khi đổi `partitions.csv`, **phải flash cả 4 files**:

```bash
# Linux (thay /dev/ttyUSB0 bằng cổng thực tế)
python -m esptool --chip esp32s3 -b 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0      build/bootloader/bootloader.bin \
    0x8000   build/partition_table/partition-table.bin \
    0x13000  build/ota_data_initial.bin \
    0x20000  build/trash_classifier.bin

# Windows (thay COM8 bằng cổng thực tế)
python -m esptool --chip esp32s3 -b 460800 --port COM8 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0      build/bootloader/bootloader.bin \
    0x8000   build/partition_table/partition-table.bin \
    0x13000  build/ota_data_initial.bin \
    0x20000  build/trash_classifier.bin
```

### Chỉ cập nhật firmware (partition không đổi)

```bash
# Chỉ flash app binary
python -m esptool --chip esp32s3 -b 460800 \
    write_flash 0x20000 build/trash_classifier.bin
```

### Bảng địa chỉ flash

| File | Địa chỉ | Mô tả |
|------|---------|-------|
| `bootloader.bin` | `0x0` | Bootloader |
| `partition-table.bin` | `0x8000` | Bảng phân vùng |
| `ota_data_initial.bin` | `0x13000` | Trỏ boot vào ota_0 |
| `trash_classifier.bin` | `0x20000` | Firmware chính |

> **Nếu board không vào boot mode:** Giữ **BOOT** → nhấn **RESET** → thả **BOOT** → chạy lại flash.

---

## Bước 6: Xem log và kiểm tra

```bash
# Linux
idf monitor -p /dev/ttyUSB0
# hoặc
screen /dev/ttyUSB0 115200

# Windows: mở Serial Monitor tại COM8, baud rate 115200
# Thoát idf monitor: Ctrl+]
```

Log boot thành công (full INT8):

```
I (xxx) main_functions: ==============================================
I (xxx) main_functions:  TrashNet Classifier  –  ESP32-S3
I (xxx) main_functions: ==============================================
I (xxx) main_functions: Model loaded from flash (3824888 bytes)
I (xxx) main_functions: Tensor arena: 828 KB allocated from PSRAM
I (xxx) main_functions: Input  tensor: [1,224,224,3] INT8  scale=0.004  zp=-128
I (xxx) main_functions: Output tensor: [1,6]         INT8  scale=0.004  zp=-128
I (xxx) main_functions: Setup complete – HTTP inference server ready
I (xxx) wifi_ap: WiFi AP started  SSID: TrashNet-ESP32  IP: 192.168.4.1
I (xxx) http_infer: HTTP server started on port 80
```

---

## Bước 7: Test inference

```bash
# Kết nối WiFi "TrashNet-ESP32" (không có mật khẩu)

# Test bằng curl (Linux/Mac/Windows)
curl -X POST http://192.168.4.1/infer \
     -H "Content-Type: image/jpeg" \
     --data-binary @/path/to/photo.jpg

# Kết quả JSON:
{
  "top_label": "Plastic",
  "top_score": 0.847,
  "inference_ms": 285,
  "scores": {
    "Cardboard": 0.021,
    "Glass":     0.043,
    "Metal":     0.018,
    "Paper":     0.062,
    "Plastic":   0.847,
    "Trash":     0.009
  }
}
```

Log trên serial khi inference:

```
I (xxx) http_infer: Received 45231 bytes JPEG
I (xxx) http_infer: Image: 640×480, decoded size: 921600 bytes
I (xxx) http_infer: Resizing 640×480 → 224×224...
I (xxx) http_infer: Calling run_inference()...
I (xxx) main_functions: Inference: Plastic (84.7%)  285 ms
```

---

## Troubleshooting

### `Hybrid models are not supported on TFLite Micro`
Model trong `trash_model_data.cc` là hybrid (FLOAT32 I/O).
```bash
# Giải pháp: retrain và nhúng lại
cd models
python3 retrain_int8.py
bash generate_model_cc.sh trash_model_int8.tflite
# → Build lại firmware
```

### `AllocateTensors() FAILED` + `Didn't find op for FULLY_CONNECTED`
Op chưa đăng ký. Kiểm tra `main/main_functions.cc` phải có:
```cpp
micro_op_resolver.AddFullyConnected();
```

### `FATAL: Input tensor type is NOT INT8 (got 1)`
Model output vẫn là FLOAT32. Kiểm tra log training:
```
  Input  : [1, 224, 224, 3]  float32   ← SAI
```
→ Chạy lại `retrain_int8.py`, kiểm tra phần Verification output.

### Không thấy log sau `entry 0x403c8938`
Console đang output sang USB Serial/JTAG, không phải UART.
Kiểm tra `sdkconfig.defaults.esp32s3`:
```
CONFIG_ESP_CONSOLE_UART_DEFAULT=y
# CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y
```
Nếu chưa có → thêm vào, xoá `sdkconfig`, build lại.

### `Warning: Nearly full` khi build
Tăng OTA slot trong `partitions.csv`:
```
ota_0, app, ota_0, 0x20000,  0x500000,   ← 5MB
ota_1, app, ota_1, 0x520000, 0x500000,   ← 5MB
```
Nhớ cập nhật offset của các partition sau đó theo công thức:
`offset_tiep_theo = offset_hien_tai + size_hien_tai`

### Val accuracy thấp (<80%) sau training
- Thêm epochs: tăng `EPOCHS_HEAD` và `EPOCHS_FINETUNE`
- Giảm learning rate cho fine-tune (trong `retrain_int8.py` line `Adam(1e-5)`)
- Kiểm tra dataset: class `trash` chỉ có 137 ảnh → dễ bị overfit

---

## Tóm tắt lệnh (chạy tuần tự từ đầu)

```bash
# ── 0. Chuẩn bị ────────────────────────────────────────────────────────────
cd /home/danz/Downloads/trash/trash_classifier/models
pip install tensorflow pillow numpy --break-system-packages

# Giải nén dataset nếu chưa có
cd training && unzip dataset-resized.zip && cd ..

# ── 1. Train + Convert INT8 (~15–23 phút) ──────────────────────────────────
python3 retrain_int8.py
# → trash_model_int8.tflite

# ── 2. Nhúng model vào firmware ─────────────────────────────────────────────
bash generate_model_cc.sh trash_model_int8.tflite
# → ../main/trash_model_data.cc

# ── 3. Build firmware ────────────────────────────────────────────────────────
cd ..
source idf_env.sh
rm -f sdkconfig   # bắt buộc khi đổi partition hoặc sdkconfig.defaults
idf build
# → build/trash_classifier.bin

# ── 4. Flash (thay COM8/ttyUSB0 bằng cổng thực tế) ──────────────────────────
python -m esptool --chip esp32s3 -b 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0      build/bootloader/bootloader.bin \
    0x8000   build/partition_table/partition-table.bin \
    0x13000  build/ota_data_initial.bin \
    0x20000  build/trash_classifier.bin

# ── 5. Xem log ───────────────────────────────────────────────────────────────
idf monitor -p /dev/ttyUSB0   # Linux
# Windows: mở Serial Monitor tại COM8, baud 115200

# ── 6. Test ──────────────────────────────────────────────────────────────────
# Kết nối WiFi "TrashNet-ESP32"
curl -X POST http://192.168.4.1/infer \
     -H "Content-Type: image/jpeg" \
     --data-binary @test.jpg
```
