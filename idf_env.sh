#!/usr/bin/env bash
# ESP-IDF v5.5.1 environment for ESP32-S3 N16R8
# Usage: source ~/idf_env.sh

export IDF_PATH=/home/danz/esp/v5.5.1/esp-idf
export IDF_PYTHON_ENV_PATH=/home/danz/.espressif/python_env/idf5.5_py3.13_env

XTENSA=/home/danz/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20241119/xtensa-esp-elf/bin
RISCV=/home/danz/.espressif/tools/riscv32-esp-elf/esp-14.2.0_20241119/riscv32-esp-elf/bin
NINJA=/home/danz/.espressif/tools/ninja/1.12.1
OPENOCD=/home/danz/.espressif/tools/openocd-esp32/v0.12.0-esp32-20250707/openocd-esp32/bin
ULP=/home/danz/.espressif/tools/esp32ulp-elf/2.38_20240113/esp32ulp-elf/bin

export PATH="$XTENSA:$RISCV:$NINJA:$OPENOCD:$ULP:$IDF_PATH/tools:$PATH"

# Function thay vì alias (hoạt động khi source trong mọi shell)
idf() {
    "$IDF_PYTHON_ENV_PATH/bin/python" "$IDF_PATH/tools/idf.py" "$@"
}
export -f idf

echo "✓ ESP-IDF v5.5.1 loaded (ESP32-S3 N16R8). Use 'idf' command."
echo "  IDF_PATH : $IDF_PATH"
echo "  Toolchain: $XTENSA"
