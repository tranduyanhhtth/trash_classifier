# TrashNet 6-Class Classifier – ESP32-S3 N16R8 + OV2640

Bộ phân loại rác thải 6 nhóm chạy trên vi điều khiển **ESP32-S3 N16R8**
(16MB Flash · 8MB Octal PSRAM) với camera **OV2640**,
sử dụng mô hình **MobileNetV1 1.0** và framework **TFLite Micro** của Espressif.

| Thông số chip | Giá trị                                  |
| --------------- | ------------------------------------------ |
| SoC             | ESP32-S3 (Xtensa LX7 dual-core 240 MHz)    |
| Flash           | 16 MB – QIO 80 MHz                        |
| PSRAM           | 8 MB Octal SPI (OPI) 80 MHz                |
| Camera          | OV2640 (JPEG, tối đa 2MP)                |
| Framework       | ESP-IDF v5.5.1 + TFLite Micro              |
| Model input     | 224 × 224 × 3 RGB                        |
| Inference time  | ~150–300 ms (float) / ~100–200 ms (INT8) |

---

## Cấu trúc dự án

```
trash/
├── esp-tflite-micro/                        ← thư viện TFLite Micro (đã clone)
└── trash_classifier/                        ← DỰ ÁN CHÍNH
    ├── partitions.csv                       ← partition table 16MB (4.5MB/OTA slot)
    ├── sdkconfig.defaults                   ← cấu hình N16R8 (PSRAM, 240MHz, ...)
    ├── sdkconfig.defaults.esp32s3           ← override cho target esp32s3
    ├── models/
    │   ├── quantized_and_pruned_model.tflite  ← model gốc (hybrid-quantized)
    │   ├── convert_to_int8.py               ← script chuyển sang full-INT8
    │   └── generate_model_cc.sh             ← script xxd → C array
    └── main/
        ├── main.cc                          ← FreeRTOS entry point
        ├── main_functions.cc                ← CORE: TFLite inference engine
        ├── model_settings.h/cc              ← nhãn 6 categories + ngưỡng
        ├── image_provider.cc/h              ← chụp ảnh, JPEG→RGB→crop→resize
        ├── classification_responder.cc/h    ← in kết quả ra serial + LED
        ├── trash_model_data.cc/h            ← model nhúng dưới dạng C array
        ├── app_camera_esp.c/h               ← cấu hình GPIO OV2640
        ├── esp_cli.c/h                      ← CLI qua UART
        ├── Kconfig.projbuild                ← menuconfig
        └── idf_component.yml                ← dependencies (esp-tflite-micro, esp32-camera)
```

---

## Môi trường ESP-IDF (đã cài sẵn trên máy này)

| Thành phần | Đường dẫn                                                                |
| ------------ | ---------------------------------------------------------------------------- |
| ESP-IDF      | `/home/danz/esp/v5.5.1/esp-idf`                                            |
| Python venv  | `~/.espressif/python_env/idf5.5_py3.13_env`                                |
| Toolchain    | `~/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20241119/xtensa-esp-elf/bin` |
| ninja        | `~/.espressif/tools/ninja/1.12.1`                                          |
| cmake        | hệ thống (`/usr/bin/cmake` v3.28, cài qua `apt`)                      |

### Load môi trường ESP-IDF

File `idf_env.sh` nằm sẵn trong thư mục dự án:

```bash
cd /home/danz/Downloads/trash/trash_classifier

# Load môi trường (chạy mỗi khi mở terminal mới)
source idf_env.sh

# Hoặc dùng đường dẫn tuyệt đối từ bất kỳ đâu:
source /home/danz/Downloads/trash/trash_classifier/idf_env.sh
```

Output:

```
✓ ESP-IDF v5.5.1 loaded (ESP32-S3 N16R8). Use 'idf' command.
  IDF_PATH : /home/danz/esp/v5.5.1/esp-idf
  Toolchain: .../xtensa-esp-elf/esp-14.2.0_20241119/...
```

> **Tip:** Tự động load mỗi lần mở terminal:
>
> ```bash
> echo "source /home/danz/Downloads/trash/trash_classifier/idf_env.sh" >> ~/.bashrc
> ```

---

## Về model: hai chế độ hoạt động

| Chế độ                      | Model                                 | Tốc độ    | Kích thước firmware         |
| ------------------------------ | ------------------------------------- | ------------ | ------------------------------ |
| **Float (hiện tại)**   | `quantized_and_pruned_model.tflite` | ~250–300 ms | **4.1 MB** ✓ đã build |
| **Full INT8 (tối ưu)** | `trash_model_int8.tflite`           | ~100–200 ms | ~0.9 MB                        |

Model gốc `quantized_and_pruned_model.tflite` là **hybrid-quantized** (weights INT8, I/O FLOAT32).
TFLite Micro hỗ trợ chế độ này — firmware đã build thành công với model này.
Để tăng tốc, có thể chuyển sang full-INT8 (xem Phần B bên dưới).

---

## Phần A – Dùng model hiện có (đã nhúng sẵn)

`trash_model_data.cc` đã được tạo xong với model `quantized_and_pruned_model.tflite`.
Bỏ qua bước chuyển đổi, chạy thẳng từ **Bước 2: Build**.

---

## Phần B – Chuyển đổi model (tuỳ chọn, để tăng tốc)

### B1. Kiểm tra model gốc

```bash
# Xem thông tin model
python3 - << 'EOF'
import struct, sys

path = "/home/danz/Downloads/trash/trash_classifier/models/quantized_and_pruned_model.tflite"
with open(path, "rb") as f:
    data = f.read()

print(f"File size : {len(data):,} bytes ({len(data)/1024/1024:.2f} MB)")
print(f"FlatBuffer magic: {data[4:8]}")   # should be b'TFL3'
print("Dùng Netron (https://netron.app) để xem chi tiết I/O dtype")
EOF
```

Kết quả model gốc:

```
File size : 3,685,560 bytes (3.51 MB)
Input  dtype : FLOAT32  ← hybrid quantization (weights INT8, I/O float)
Output dtype : FLOAT32
```

### B2. Chuyển sang full-INT8 (cần dataset để calibrate)

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

# Cài TensorFlow nếu chưa có
pip install tensorflow pillow numpy

# Option A: dùng ảnh thật từ TrashNet dataset (accuracy cao nhất)
python3 convert_to_int8.py \
    --model quantized_and_pruned_model.tflite \
    --images /đường/dẫn/đến/thư/mục/ảnh \
    --output trash_model_int8.tflite \
    --num-samples 200

# Option B: dùng random data (nhanh, ~2% accuracy thấp hơn)
python3 convert_to_int8.py \
    --model quantized_and_pruned_model.tflite \
    --random \
    --output trash_model_int8.tflite
```

Kết quả mong đợi:

```
✓ Saved: trash_model_int8.tflite
  Input  : shape=[1, 224, 224, 3], dtype=int8   ✓
  Output : shape=[1, 6], dtype=int8              ✓
✓ SUCCESS: Model is fully INT8-quantized!
```

### B3. Nhúng model vào firmware bằng xxd

Đây là quá trình thực tế đã làm với model gốc (áp dụng tương tự cho INT8):

```bash
cd /home/danz/Downloads/trash/trash_classifier/models

# ── Bước 3a: Kiểm tra kích thước model ──────────────────────────────────────
ls -lh trash_model_int8.tflite
# VD: 3.6M quantized_and_pruned_model.tflite  (đã nhúng)
#     ~900K trash_model_int8.tflite           (INT8 nhỏ hơn nhiều)

# ── Bước 3b: Chuyển .tflite → C array bằng xxd ──────────────────────────────
# Cách 1: dùng script tự động (khuyến nghị)
./generate_model_cc.sh trash_model_int8.tflite
# → tự động ghi ra: ../main/trash_model_data.cc

# Cách 2: thủ công (hiểu rõ từng bước)
# Bước i – tạo header file trước
cat > ../main/trash_model_data.cc << 'HEADER'
/*
 * trash_model_data.cc – AUTO-GENERATED
 * Nguồn: trash_model_int8.tflite
 * Tái tạo: cd models && ./generate_model_cc.sh trash_model_int8.tflite
 */
#include "trash_model_data.h"
HEADER

# Bước ii – chạy xxd để tạo hex array, đổi tên symbol cho đúng
xxd -i trash_model_int8.tflite \
  | sed 's/^unsigned char .*\[\]/const unsigned char g_trash_model_data[]/' \
  | sed 's/^unsigned int .*=/const unsigned int g_trash_model_data_len =/' \
  >> ../main/trash_model_data.cc

# Bước iii – kiểm tra kết quả
echo "Số dòng :"
wc -l ../main/trash_model_data.cc

echo "Dòng đầu (header):"
head -6 ../main/trash_model_data.cc

echo "Dòng cuối (độ dài array):"
tail -3 ../main/trash_model_data.cc
```

Kết quả mong đợi sau bước 3b:

```
Số dòng : 307149   (cho model 3.6MB; ~75K dòng cho INT8 ~900KB)
Dòng đầu:
  #include "trash_model_data.h"
  const unsigned char g_trash_model_data[] = {
    0x1c, 0x00, 0x00, ...
Dòng cuối:
  };
  const unsigned int g_trash_model_data_len = 3685560;
```

> ⚠️ **Lưu ý partition table:** Kích thước firmware = kích thước model + ~500KB overhead.
> Model 3.6MB → firmware 4.1MB → cần OTA slot ≥ 4.5MB (đã cấu hình trong `partitions.csv`).
> Model INT8 ~900KB → firmware ~1.4MB → slot 3MB là đủ (có thể giảm partition để tiết kiệm).

---

## Bước 1: Chuẩn bị model (chọn A hoặc B ở trên)

Model mặc định (`quantized_and_pruned_model.tflite`) **đã được nhúng sẵn** vào
`main/trash_model_data.cc`. Bạn có thể build và flash ngay mà không cần làm gì thêm.

---

## Bước 2: Set target và build

```bash
# ── Load môi trường ESP-IDF ──────────────────────────────────────────────────
source ~/idf_env.sh
# Output: ✓ ESP-IDF v5.5.1 loaded. Use 'idf' command.

# ── Vào thư mục dự án ────────────────────────────────────────────────────────
cd /home/danz/Downloads/trash/trash_classifier

# ── Lần đầu tiên: đặt target chip ────────────────────────────────────────────
idf set-target esp32s3
# Sẽ tạo file sdkconfig từ sdkconfig.defaults + sdkconfig.defaults.esp32s3
# Output: Set Target to: esp32s3, new sdkconfig will be created.

# ── (Tuỳ chọn) Xem và chỉnh cấu hình thông qua menu ─────────────────────────
idf menuconfig
# → Camera Module Selection: chọn đúng board (mặc định: ESP32-S3-EYE)

# ── Build (lần đầu: 5–10 phút; các lần sau: ~30 giây) ────────────────────────
idf build
```

Kết quả build thành công (thực tế):

```
[1342/1342] Project build complete. To flash, run:
  idf.py flash
trash_classifier.bin binary size 0x402540 bytes.
Smallest app partition is 0x480000 bytes. 0x7dac0 bytes (11%) free.
```

---

## Bước 3: Flash lên ESP32-S3 N16R8

```bash
# ── Tìm cổng của board ───────────────────────────────────────────────────────
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
# ESP32-S3 native USB → /dev/ttyACM0
# Board dùng CH340/CP2102 → /dev/ttyUSB0

# ── Flash qua USB-JTAG native (ESP32-S3 hỗ trợ sẵn, không cần mạch ngoài) ───
idf flash -p /dev/ttyACM0

# ── Flash qua UART (CH340/CP2102), tốc độ cao hơn ────────────────────────────
idf flash -p /dev/ttyUSB0 --baud 921600

# ── Flash và mở serial monitor luôn sau khi xong ────────────────────────────
idf flash monitor -p /dev/ttyACM0

# ── Hoặc dùng lệnh esptool trực tiếp (in sẵn khi build xong) ────────────────
python -m esptool --chip esp32s3 -b 460800 \
    --before default_reset --after hard_reset \
    write_flash \
    --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0      build/bootloader/bootloader.bin \
    0x8000   build/partition_table/partition-table.bin \
    0x13000  build/ota_data_initial.bin \
    0x20000  build/trash_classifier.bin

# ── Chỉ mở monitor (board đã có firmware) ────────────────────────────────────
idf monitor -p /dev/ttyACM0
# Thoát monitor: Ctrl+]
```

> **N16R8 – nếu flash thất bại:**
> Giữ nút **BOOT** → nhấn **RESET** một lần → thả **BOOT** → chạy lại `idf flash`.
> Sau khi flash xong, nhấn **RESET** để khởi động firmware mới.

---

## Kết quả trên Serial Monitor

```
I (1234) main_functions: ─────────────────────────────────────────
I (1234) main_functions:  TrashNet Inference  (142 ms)
I (1234) main_functions: ─────────────────────────────────────────
I (1234) main_functions:     Cardboard   2.1% [####................]
I (1234) main_functions:     Glass       1.8% [###.................]
I (1234) main_functions:     Metal       3.4% [######..............]
I (1234) main_functions: >>  Paper      87.6% [#################...] <<
I (1234) main_functions:     Plastic     4.1% [########............]
I (1234) main_functions:     Trash       1.0% [##..................]
I (1234) main_functions: ─────────────────────────────────────────
I (1234) main_functions:  RESULT: [PAPER]  87.6%  Paper
I (1234) main_functions: ─────────────────────────────────────────
```

---

## Thông số bộ nhớ (N16R8, model float nhúng vào flash)

| Vùng nhớ      | Kích thước | Vị trí | Mục đích                      |
| --------------- | ------------- | -------- | -------------------------------- |
| Model weights   | 3.5 MB        | Flash    | `g_trash_model_data[]` in DROM |
| Tensor Arena    | 350 KB        | SRAM*    | Activation buffers khi inference |
| Camera FB (×2) | ~230 KB       | PSRAM    | JPEG frame buffer                |
| RGB decode buf  | 230 KB        | PSRAM    | Decoded RGB888 (320×240×3)     |
| Resize scratch  | 150 KB        | PSRAM    | Buffer trung gian 224×224×3    |

*Tensor Arena: ưu tiên SRAM (nhanh hơn 3×), tự động fallback sang PSRAM nếu không đủ.

---

## Classes (TrashNet 6 nhãn)

| Index | Nhãn     | Tiếng Việt                  |
| ----- | --------- | ----------------------------- |
| 0     | Cardboard | Bìa các-tông / giấy cứng |
| 1     | Glass     | Thủy tinh                    |
| 2     | Metal     | Kim loại                     |
| 3     | Paper     | Giấy                         |
| 4     | Plastic   | Nhựa                         |
| 5     | Trash     | Rác hỗn hợp                |

---

## Troubleshooting

**`AllocateTensors()` FAILED**
→ Tăng `kTensorArenaSize` trong `main_functions.cc`. Thử 450KB trước.

**Camera init failed**
→ Kiểm tra GPIO. Chạy `idf menuconfig` → Camera Module Selection.

**Inference rất chậm (>500ms)**
→ Kiểm tra `CONFIG_NN_OPTIMIZED=y` trong `sdkconfig`.
→ Đảm bảo CPU ở 240MHz: `idf menuconfig` → `ESP System Settings → CPU frequency`.

**Input tensor type không khớp**→ Mở `main_functions.cc` xem `s_input->type`:

- `kTfLiteFloat32` = model float (hiện tại), input dùng `s_input->data.f`
- `kTfLiteInt8` = model INT8, input dùng `s_input->data.int8`

**Low confidence liên tục (<50%)**
→ Giảm `kConfidenceThreshold` trong `model_settings.h`.
→ Đảm bảo ánh sáng đủ, vật thể chiếm >60% khung hình.

**Partition table overflow khi build**
→ Kiểm tra kích thước firmware: `ls -lh build/trash_classifier.bin`
→ Tăng kích thước OTA slot trong `partitions.csv` (hiện tại: 4.5MB = `0x480000`)
