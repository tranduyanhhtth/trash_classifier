# TrashNet 6-Class Classifier – ESP32-S3 N16R8

Bộ phân loại rác thải 6 nhóm chạy trên **ESP32-S3 N16R8**
(16MB Flash · 8MB Octal PSRAM), sử dụng **MobileNetV1 1.0 (full INT8)**
và **TFLite Micro** của Espressif với esp_nn optimized kernels.

Inference được kích hoạt qua **HTTP POST** — chip tạo WiFi hotspot,
người dùng kết nối và upload ảnh JPEG để nhận kết quả phân loại.

| Thông số        | Giá trị                                        |
| --------------- | ----------------------------------------------- |
| SoC             | ESP32-S3 (Xtensa LX7 dual-core 240 MHz)         |
| Flash           | 16 MB – DIO 80 MHz                              |
| PSRAM           | 8 MB Octal SPI 80 MHz                           |
| Framework       | ESP-IDF v5.5.1 + TFLite Micro (esp_nn)          |
| Model           | MobileNetV1 1.0 – **full INT8** (224×224×3)     |
| Inference time  | ~150–300 ms (từ PSRAM)                          |
| WiFi mode       | Access Point – SSID: `TrashNet-ESP32`           |
| HTTP endpoint   | `POST http://192.168.4.1/infer` (JPEG body)     |

> ⚠️ **Yêu cầu bắt buộc: Model phải là full INT8**
> esp_nn kernels của ESP32 **không hỗ trợ hybrid quantized model**
> (weights INT8, I/O FLOAT32). Dùng `models/retrain_int8.py` để tạo model đúng.

---

## Cấu trúc dự án

```
trash/
├── esp-tflite-micro/                        ← thư viện TFLite Micro (đã clone)
└── trash_classifier/                        ← DỰ ÁN CHÍNH
    ├── idf_env.sh                           ← load ESP-IDF v5.5.1 nhanh
    ├── partitions.csv                       ← partition table 16MB (4.75MB/OTA)
    ├── sdkconfig.defaults                   ← cấu hình N16R8 (PSRAM, 240MHz)
    ├── sdkconfig.defaults.esp32s3           ← console UART0, camera module
    ├── models/
    │   ├── retrain_int8.py                  ← rebuild + convert → full INT8 ✅
    │   ├── convert_to_int8.py               ← convert từ float model (nếu có)
    │   ├── generate_model_cc.sh             ← .tflite → C array (xxd)
    │   ├── quantized_and_pruned_model.tflite← model gốc (hybrid, KHÔNG dùng)
    │   └── training/
    │       ├── pruning-and-quantization-in-keras.ipynb
    │       └── dataset-resized/dataset-resized/  ← 2527 ảnh, 6 class
    └── main/
        ├── main.cc                          ← FreeRTOS entry, WiFi + HTTP init
        ├── main_functions.cc                ← CORE: TFLite setup() + run_inference()
        ├── model_settings.h/cc              ← nhãn 6 class + kConfidenceThreshold
        ├── image_provider.cc/h              ← camera: JPEG→RGB→resize 224×224
        ├── classification_responder.cc/h    ← in kết quả ra serial
        ├── wifi_ap.cc/h                     ← WiFi Access Point "TrashNet-ESP32"
        ├── http_infer.cc/h                  ← HTTP server: POST /infer
        ├── trash_model_data.cc/h            ← model nhúng dạng C array (DROM)
        ├── app_camera_esp.c/h               ← cấu hình GPIO OV2640
        └── esp_cli.c/h                      ← CLI test ảnh tĩnh qua UART
```

---

## Môi trường ESP-IDF

| Thành phần | Đường dẫn                                                                |
| ----------- | -------------------------------------------------------------------------- |
| ESP-IDF     | `/home/danz/esp/v5.5.1/esp-idf`                                           |
| Python venv | `~/.espressif/python_env/idf5.5_py3.13_env`                               |
| Toolchain   | `~/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20241119/xtensa-esp-elf/bin`|

```bash
# Load môi trường (chạy mỗi khi mở terminal mới)
cd /home/danz/Downloads/trash/trash_classifier
source idf_env.sh
# Output: ✓ ESP-IDF v5.5.1 loaded (ESP32-S3 N16R8). Use 'idf' command.
```

---

## Bước 0: Tạo model full INT8 (bắt buộc)

Model gốc `quantized_and_pruned_model.tflite` là **hybrid quantized** và
**không tương thích** với esp_nn kernels của TFLite Micro.
Phải tạo lại model `trash_model_int8.tflite` trước khi build firmware.

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

# Cài dependencies (một lần)
pip install tensorflow pillow numpy --break-system-packages

# Rebuild model + convert sang full INT8
# (fine-tune 15 epoch head + 5 epoch full → ~15–20 phút)
python3 retrain_int8.py
```

Kết quả thành công:

```
TensorFlow 2.21.0 loaded
Dataset : training/dataset-resized/dataset-resized
  cardboard   : 403 images
  glass       : 501 images
  metal       : 410 images
  paper       : 594 images
  plastic     : 482 images
  trash       : 137 images

Phase 1: Training head only (15 epochs)...
✓ Head training done. Best val accuracy: 0.9124

Phase 2: Fine-tuning all layers (5 epochs)...
✓ Fine-tune done. Best val accuracy: 0.9312

Converting to full INT8 (calibrating activations)...
✓ Saved: trash_model_int8.tflite  (3516 KB)

Verification:
  Input  : [1, 224, 224, 3]  int8   scale=0.007874  zp=-1
  Output : [1, 6]            int8   scale=0.003906  zp=-128
✓ SUCCESS: Full INT8 – compatible with TFLite Micro esp_nn!
```

Sau đó nhúng model vào firmware:

```bash
bash generate_model_cc.sh trash_model_int8.tflite
# → tự động ghi ra: ../main/trash_model_data.cc
```

---

## Bước 1: Build firmware

```bash
cd /home/danz/Downloads/trash/trash_classifier
source idf_env.sh

# Xoá sdkconfig cũ để áp dụng defaults mới (nếu chưa làm)
rm -f sdkconfig

# Build (~5–10 phút lần đầu, ~30s các lần sau)
idf build
```

Kết quả build thành công:

```
trash_classifier.bin binary size 0x496eb0 bytes.
Smallest app partition is 0x4c0000 bytes. 0x29150 bytes (3%) free.
Project build complete.
```

> ⚠️ **Partition gần đầy (3% còn lại).** Khi model INT8 được nhúng (nhỏ hơn
> ~3.5MB so với hybrid), firmware sẽ nhỏ lại đáng kể và có nhiều room hơn.

---

## Bước 2: Flash lên ESP32-S3 N16R8

Flash **4 files** theo địa chỉ chính xác:

```bash
# Cách 1: idf flash (tự động đọc flash_args)
idf flash -p /dev/ttyUSB0

# Cách 2: esptool thủ công (dùng khi flash qua web hoặc tool ngoài)
python -m esptool --chip esp32s3 -b 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0      build/bootloader/bootloader.bin \
    0x8000   build/partition_table/partition-table.bin \
    0x13000  build/ota_data_initial.bin \
    0x20000  build/trash_classifier.bin
```

| File                        | Địa chỉ  | Vai trò                              |
| ---------------------------- | --------- | ------------------------------------- |
| `bootloader.bin`            | `0x0`    | Bootloader (chọn OTA slot)            |
| `partition-table.bin`       | `0x8000` | Bảng phân vùng                       |
| `ota_data_initial.bin`      | `0x13000`| Trỏ boot vào `ota_0` (lần đầu)      |
| `trash_classifier.bin`      | `0x20000`| Firmware chính                        |

> **N16R8 – nếu flash thất bại:** Giữ **BOOT** → nhấn **RESET** → thả **BOOT** → flash lại.

---

## Bước 3: Xem log serial

Console được cấu hình **UART0** (cùng cổng với bootloader):

```bash
# Linux/Mac
idf monitor -p /dev/ttyUSB0
# hoặc
screen /dev/ttyUSB0 115200

# Windows: mở Serial Monitor tại COM8 (hoặc cổng UART của board), baud 115200
```

Log khởi động thành công:

```
I (xxx) main_functions: ==============================================
I (xxx) main_functions:  TrashNet Classifier  –  ESP32-S3 + OV2640
I (xxx) main_functions: ==============================================
I (xxx) main_functions: Categories  : 6
I (xxx) main_functions: [0] Cardboard  [1] Glass  [2] Metal
I (xxx) main_functions: [3] Paper      [4] Plastic [5] Trash
I (xxx) main_functions: Model loaded from flash (3824888 bytes)
I (xxx) main_functions: Tensor arena: 1280 KB allocated from PSRAM
I (xxx) main_functions: Input tensor: [1,224,224,3] INT8 scale=0.008 zp=-1
I (xxx) main_functions: Setup complete – HTTP inference server ready
I (xxx) wifi_ap: WiFi AP started  SSID: TrashNet-ESP32  IP: 192.168.4.1
I (xxx) http_infer: HTTP server started on port 80
```

---

## Bước 4: Sử dụng HTTP inference

### Kết nối WiFi

1. Trên điện thoại / máy tính, kết nối WiFi: **`TrashNet-ESP32`** (không có mật khẩu)
2. IP gateway: `192.168.4.1`

### Upload ảnh và nhận kết quả

```bash
# Dùng curl (Linux/Mac/Windows với curl)
curl -X POST http://192.168.4.1/infer \
     -H "Content-Type: image/jpeg" \
     --data-binary @photo.jpg

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

### Giao diện web đơn giản

Truy cập `http://192.168.4.1` trên trình duyệt → giao diện upload ảnh.

---

## Thông số bộ nhớ (full INT8 model)

| Vùng nhớ       | Kích thước | Vị trí | Mục đích                          |
| --------------- | ----------- | ------- | ---------------------------------- |
| Model weights   | ~3.5 MB     | Flash   | `g_trash_model_data[]` in DROM    |
| Tensor Arena    | 1280 KB     | PSRAM   | Activation + scratch buffers      |
| WiFi / stack    | ~100 KB     | SRAM    | FreeRTOS + LwIP + WiFi driver     |
| HTTP server     | ~20 KB      | SRAM    | esp_http_server                   |
| JPEG decode buf | ~920 KB     | PSRAM   | Decode JPEG ảnh upload            |

> **Tại sao arena từ PSRAM?**
> Full INT8 model với Flatten(): SHAPE+STRIDED_SLICE+PACK tạo nhiều scratch tensor.
> Tổng đo thực tế: 1,206,848 bytes (1,179 KB). Dùng 1280 KB để có ~8% headroom.
> PSRAM 8 MB của N16R8 đủ thừa. (Dùng GlobalAveragePooling2D thay Flatten → giảm xuống ~200 KB)

---

## Classes (TrashNet 6 nhãn)

| Index | Label    | Tiếng Việt              | Mẫu train |
| ----- | --------- | ------------------------ | ---------- |
| 0     | Cardboard | Bìa các-tông / giấy cứng | 403        |
| 1     | Glass     | Thủy tinh                | 501        |
| 2     | Metal     | Kim loại                 | 410        |
| 3     | Paper     | Giấy                     | 594        |
| 4     | Plastic   | Nhựa                     | 482        |
| 5     | Trash     | Rác hỗn hợp              | 137        |

---

## Troubleshooting

**`Hybrid models are not supported on TFLite Micro`**
→ Model trong `trash_model_data.cc` là hybrid quantized, không tương thích với esp_nn.
→ Chạy: `python3 models/retrain_int8.py` → `bash models/generate_model_cc.sh trash_model_int8.tflite`

**`AllocateTensors() FAILED` + `Didn't find op for builtin opcode 'FULLY_CONNECTED'`**
→ Op chưa được đăng ký trong `MicroMutableOpResolver`.
→ Kiểm tra `main_functions.cc`: phải có `micro_op_resolver.AddFullyConnected()`.

**`FATAL: Input tensor type is NOT INT8`**
→ Model vẫn là FLOAT32 I/O (hybrid). Tạo lại model với `retrain_int8.py`.

**Không thấy log sau dòng `entry 0x403c8938`**
→ Firmware đang log ra USB Serial/JTAG thay vì UART.
→ Kiểm tra `sdkconfig.defaults.esp32s3` phải có `CONFIG_ESP_CONSOLE_UART_DEFAULT=y`.
→ Xem log đúng trên cổng UART (COM8 trên Windows, /dev/ttyUSB0 trên Linux).

**Camera init failed**
→ Camera đang bị comment (HTTP inference mode không cần camera).
→ Để bật: bỏ comment `[CAMERA MODE]` trong `main.cc` và `main_functions.cc`.

**Inference trả về 0% hoặc kết quả sai**
→ Kiểm tra log: model có load thành công không? (`Model loaded from flash`)
→ Kiểm tra ảnh upload có đúng định dạng JPEG không.
→ Kiểm tra kích thước ảnh: `http_infer.cc` resize về 224×224 trước khi inference.

**Partition table overflow khi build**
→ Kiểm tra `ls -lh build/trash_classifier.bin`
→ Partition OTA hiện tại: 4.75 MB (`0x4C0000`). Tăng thêm trong `partitions.csv` nếu cần.

---

## Training

### Dataset TrashNet

```
models/training/
├── pruning-and-quantization-in-keras.ipynb  ← notebook gốc
├── dataset-resized.zip                      ← gitignored (43MB)
└── dataset-resized/dataset-resized/
    ├── cardboard/   (403 ảnh, 512×384 JPEG)
    ├── glass/       (501 ảnh)
    ├── metal/       (410 ảnh)
    ├── paper/       (594 ảnh)
    ├── plastic/     (482 ảnh)
    └── trash/       (137 ảnh)
```

```bash
# Giải nén dataset
cd models/training && unzip dataset-resized.zip
```

### Workflow tạo model (full INT8)

```
Dataset → retrain_int8.py → trash_model_int8.tflite
                                    ↓
                         python3 models/check_ops.py    ← verify ops trước khi build!
                                    ↓
                         generate_model_cc.sh
                                    ↓
                         main/trash_model_data.cc  (C array, alignas(8))
                                    ↓
                               idf build + flash
```

**Tại sao dùng `retrain_int8.py` thay notebook gốc?**
- Notebook lưu model vào `tempfile.mkstemp()` → file bị xoá khi đóng session
- `retrain_int8.py` dùng **`preprocess_input`** (MobileNet chuẩn: [0,255]→[-1,1])
  vs notebook gốc dùng `rescale=1/255` ([0,1]) → accuracy thấp hơn ~20-25%
- Convert thẳng sang full INT8 với `inference_input_type=tf.int8`

**Model hiện tại (sau retrain):**
- Input: `[1,224,224,3]` INT8  `scale=0.007843  zp=-1`  (range ≈ [-1,1])
- Output: `[1,6]` INT8  `scale=0.003906  zp=-128`
- Ops: `CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, SHAPE, STRIDED_SLICE, PACK, RESHAPE, SOFTMAX`
- Array: `alignas(8) const unsigned char g_trash_model_data[]` (khớp person_detect_model_data.cc)

**ESP32 preprocessing (không cần thay đổi):**
```cpp
// uint8 [0,255] → int8 [-128,127] via XOR 0x80 (pixel - 128)
// Với scale=0.00784, zp=-1: maps pixel=0→-1.0, pixel=128→0.0, pixel=255→+1.0
s_input->data.int8[i] = (int8_t)(raw_pixels[i] ^ 0x80);  // ✅ đúng
```

---

## Bật Camera Mode (tuỳ chọn)

Firmware hiện tại dùng HTTP inference (không cần camera). Để bật live camera:

**1. Bỏ comment trong `main/main.cc`:**
```cpp
// while (true) {
//     loop();          ← bỏ comment 2 dòng này
// }
```

**2. Bỏ comment trong `main/main_functions.cc` (block `[CAMERA MODE]`):**
```cpp
// #ifndef CLI_ONLY_INFERENCE
//     TfLiteStatus cam_status = InitCamera();   ← bỏ comment block này
//     ...
// #endif
```

**3. Build lại.**

---

## Hướng dẫn mở rộng

### A – Thêm/bớt ảnh test CLI

Hệ thống ảnh test gồm **3 thành phần phải đồng bộ**:

| File                             | Vai trò                                            |
| --------------------------------- | --------------------------------------------------- |
| `static_images/CMakeLists.txt`  | Khai báo `EMBED_FILES` → nhúng vào firmware        |
| `main/esp_cli.c`                | `IMAGE_COUNT`, extern symbols, `image_database[]`  |
| `static_images/sample_images/`  | File raw RGB888, 224×224×3 = 150,528 bytes mỗi file|

> **Số ảnh tối ưu = bội số 6** (6, 12, 18…) để mỗi class có số mẫu bằng nhau.
> Hiện tại: 12 ảnh (2 ảnh/class).

#### Ví dụ: tạo 12 ảnh cân bằng (2 ảnh/class)

```bash
cd /home/danz/Downloads/trash/trash_classifier

python3 << 'EOF'
from PIL import Image
import os

DATASET = "models/training/dataset-resized/dataset-resized"
OUT     = "static_images/sample_images"
os.makedirs(OUT, exist_ok=True)

plan = [
    ("cardboard", 0, "image0"),  ("glass",    0, "image1"),
    ("metal",     0, "image2"),  ("paper",    0, "image3"),
    ("plastic",   0, "image4"),  ("trash",    0, "image5"),
    ("cardboard", 1, "image6"),  ("glass",    1, "image7"),
    ("metal",     1, "image8"),  ("paper",    1, "image9"),
    ("plastic",   1, "image10"), ("trash",    1, "image11"),
]

for cls, idx, outname in plan:
    files = sorted(f for f in os.listdir(os.path.join(DATASET, cls))
                   if not f.startswith('.'))
    img = Image.open(os.path.join(DATASET, cls, files[idx])).convert("RGB").resize((224,224))
    with open(os.path.join(OUT, outname), "wb") as f:
        f.write(img.tobytes())
    print(f"✓ {outname} ← {cls}/{files[idx]}")
EOF
```

Cập nhật `static_images/CMakeLists.txt`:
```cmake
EMBED_FILES "sample_images/image0"  "sample_images/image1"
            "sample_images/image2"  "sample_images/image3"
            "sample_images/image4"  "sample_images/image5"
            "sample_images/image6"  "sample_images/image7"
            "sample_images/image8"  "sample_images/image9"
            "sample_images/image10" "sample_images/image11"
```

Cập nhật `main/esp_cli.c`: `#define IMAGE_COUNT 12`

### B – Thêm class mới

1. **Dataset:** thêm thư mục class mới vào `models/training/dataset-resized/dataset-resized/`
2. **Retrain:** chỉnh `CLASSES` trong `retrain_int8.py`, chạy lại
3. **Model settings:** sửa `kCategoryCount` và `kCategoryLabels[]` trong `model_settings.h/cc`
4. **Nhúng model:** `bash models/generate_model_cc.sh trash_model_int8.tflite`
5. **Build:** `rm -f sdkconfig && idf build`

---

## Quy tắc tổng quát

| Muốn làm gì                           | Files cần sửa                                                        |
| -------------------------------------- | --------------------------------------------------------------------- |
| Thêm/bớt ảnh test CLI                 | `static_images/CMakeLists.txt` + `main/esp_cli.c` + file raw        |
| Đổi ảnh test (cùng số lượng)          | Chỉ thay file raw, build lại                                         |
| Thêm class mới                         | `model_settings.h/cc` + retrain + nhúng model + thêm ảnh test       |
| Thay model (cùng class, cùng size)    | `generate_model_cc.sh` + build lại                                   |
| Đổi resolution input                   | `model_settings.h` (kNumCols/kNumRows) + retrain + convert ảnh       |
| Đổi sang full camera mode             | Bỏ comment `[CAMERA MODE]` trong `main.cc` + `main_functions.cc`    |
