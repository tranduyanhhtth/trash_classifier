#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Khởi động WiFi Access Point.
 *
 * ESP32-S3 tạo hotspot "TrashNet-ESP32" (password: trashnet123).
 * Sau khi gọi xong, truy cập http://192.168.4.1 từ thiết bị đã kết nối.
 */
esp_err_t wifi_ap_start(void);

#ifdef __cplusplus
}
#endif
