#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Khởi động HTTP Inference Server trên cổng 80.
 *
 * Endpoints:
 *   GET  /        → Trang web upload ảnh (HTML)
 *   POST /infer   → Nhận JPEG, chạy inference, trả JSON kết quả
 *   GET  /health  → {"status":"ok"}
 */
esp_err_t http_infer_server_start(void);

#ifdef __cplusplus
}
#endif
