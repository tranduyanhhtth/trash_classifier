/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * image_provider.cc  –  Camera capture & pre-processing for 224×224 RGB.
 *
 * Pipeline overview:
 *   OV2640 (JPEG, QVGA 320×240)
 *       │
 *       ▼ esp_camera_fb_get()   → raw JPEG bytes in PSRAM
 *       │
 *       ▼ fmt2rgb888()          → RGB888 planar buffer (from esp32-camera)
 *       │
 *       ▼ Bilinear crop+resize  → 224×224 RGB888
 *       │
 *       ▼ Quantize              → int8_t  (pixel_uint8 - 128)
 *       │
 *       ▼ → TfLiteTensor input buffer
 *
 * Memory layout:
 *   - JPEG frame buffer  : allocated in PSRAM via esp-camera (fb_count=2)
 *   - RGB decode buffer  : ~230 KB, heap_caps_malloc(PSRAM)
 *   - Resize scratch     : 224×224×3 = ~150 KB, reuses model tensor memory
 *
 * Design notes:
 *   1. We capture at FRAMESIZE_QVGA (320×240) then centre-crop the largest
 *      square (240×240) and scale down to 224×224.  This avoids the OV2640
 *      native 224×224 mode which has non-square pixels on some modules.
 *   2. All heavy buffers live in PSRAM.  The tensor arena lives in SRAM for
 *      maximum inference throughput (SRAM is ~3× faster than PSRAM on S3).
 *   3. We use nearest-neighbour resize for speed; bilinear can be enabled
 *      by defining USE_BILINEAR_RESIZE.
 */

#include "image_provider.h"
#include "model_settings.h"
#include "app_camera_esp.h"
#include "esp_main.h"

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cstring>
#include <cstdlib>

static const char* TAG = "image_provider";

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
// Decoded RGB888 buffer for the raw camera frame (320×240×3 = 230,400 bytes)
// Allocated on PSRAM once in InitCamera().
static uint8_t* s_rgb_buf      = nullptr;
static size_t   s_rgb_buf_size = 0;

// Optional display buffer (RGB565, 224×224×2 = 100,352 bytes) – PSRAM
static uint16_t* s_display_buf = nullptr;

// ---------------------------------------------------------------------------
// Public: InitCamera
// ---------------------------------------------------------------------------
TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
    ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled – skipping camera init");
    return kTfLiteOk;
#endif

#if DISPLAY_SUPPORT
    if (!s_display_buf) {
        s_display_buf = (uint16_t*)heap_caps_malloc(
            kNumCols * kNumRows * sizeof(uint16_t),
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_display_buf) {
            ESP_LOGE(TAG, "Failed to allocate display buffer (%u bytes)",
                     (unsigned)(kNumCols * kNumRows * sizeof(uint16_t)));
            return kTfLiteError;
        }
    }
#endif

    // Pre-allocate RGB decode buffer in PSRAM.
    // Size = largest possible decoded frame (320×240×3)
    s_rgb_buf_size = 320 * 240 * 3;
    if (!s_rgb_buf) {
        s_rgb_buf = (uint8_t*)heap_caps_malloc(
            s_rgb_buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_rgb_buf) {
            ESP_LOGE(TAG, "Failed to allocate RGB decode buffer (%u bytes)",
                     (unsigned)s_rgb_buf_size);
            return kTfLiteError;
        }
    }
    ESP_LOGI(TAG, "RGB decode buffer: %u bytes in PSRAM", (unsigned)s_rgb_buf_size);

#if ESP_CAMERA_SUPPORTED
    int ret = app_camera_init();
    if (ret != 0) {
        ESP_LOGE(TAG, "app_camera_init() failed (ret=%d)", ret);
        return kTfLiteError;
    }
    ESP_LOGI(TAG, "OV2640 camera initialised successfully");
#else
    ESP_LOGE(TAG, "Camera not supported for this build target");
    return kTfLiteError;
#endif

    return kTfLiteOk;
}

// ---------------------------------------------------------------------------
// Helper: display buffer accessor
// ---------------------------------------------------------------------------
void* image_provider_get_display_buf() {
    return (void*)s_display_buf;
}

// ---------------------------------------------------------------------------
// Helper: nearest-neighbour resize + crop
// ---------------------------------------------------------------------------
/**
 * @brief  Centre-crop src_w×src_h to a square then scale to dst_w×dst_h.
 *         Both src and dst are RGB888 (3 bytes/pixel, interleaved).
 *
 * @param src        Source buffer  (src_w × src_h × 3)
 * @param src_w      Source width
 * @param src_h      Source height
 * @param dst        Destination buffer (dst_w × dst_h × 3, caller-allocated)
 * @param dst_w      Destination width  (224)
 * @param dst_h      Destination height (224)
 */
static void crop_and_resize_rgb888(const uint8_t* src, int src_w, int src_h,
                                   uint8_t* dst,  int dst_w, int dst_h) {
    // Determine the largest square crop centred in the source frame
    int crop_size  = (src_w < src_h) ? src_w : src_h;
    int crop_x0    = (src_w - crop_size) / 2;  // left offset
    int crop_y0    = (src_h - crop_size) / 2;  // top  offset

    // Scale factors (fixed-point ×16 for speed)
    int scale_x_fp = (crop_size << 16) / dst_w;  // src pixels per dst pixel × 2^16
    int scale_y_fp = (crop_size << 16) / dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        int sy = crop_y0 + (int)(((long long)dy * scale_y_fp) >> 16);
        if (sy >= src_h) sy = src_h - 1;

        const uint8_t* src_row = src + sy * src_w * 3;
        uint8_t*       dst_row = dst + dy * dst_w * 3;

        for (int dx = 0; dx < dst_w; dx++) {
            int sx = crop_x0 + (int)(((long long)dx * scale_x_fp) >> 16);
            if (sx >= src_w) sx = src_w - 1;

            const uint8_t* sp = src_row + sx * 3;
            uint8_t*       dp = dst_row + dx * 3;
            dp[0] = sp[0];  // R
            dp[1] = sp[1];  // G
            dp[2] = sp[2];  // B
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: convert RGB888 224×224 to RGB565 for display
// ---------------------------------------------------------------------------
#if DISPLAY_SUPPORT
static void rgb888_to_rgb565(const uint8_t* src_rgb888, uint16_t* dst_rgb565,
                              int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint8_t r = src_rgb888[i * 3 + 0];
        uint8_t g = src_rgb888[i * 3 + 1];
        uint8_t b = src_rgb888[i * 3 + 2];
        dst_rgb565[i] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    }
}
#endif

// ---------------------------------------------------------------------------
// Public: GetImage
// ---------------------------------------------------------------------------
TfLiteStatus GetImage(int image_width, int image_height, int channels,
                      int8_t* image_data) {
#if !ESP_CAMERA_SUPPORTED
    (void)image_width; (void)image_height; (void)channels; (void)image_data;
    ESP_LOGE(TAG, "Camera not supported");
    return kTfLiteError;
#else

    // ── 1. Capture JPEG frame ──────────────────────────────────────────────
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "esp_camera_fb_get() failed");
        return kTfLiteError;
    }

    bool ok = false;

    // ── 2. Decode JPEG → RGB888 into s_rgb_buf ────────────────────────────
    // esp32-camera provides fmt2rgb888() for JPEG → RGB888 conversion.
    // src buffer is in PSRAM; decode target (s_rgb_buf) also PSRAM – acceptable.
    size_t out_len = 0;
    bool decode_ok = false;

    if (fb->format == PIXFORMAT_JPEG) {
        // Decode JPEG into our pre-allocated PSRAM buffer
        decode_ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, s_rgb_buf);
        out_len = (size_t)fb->width * fb->height * 3;
    } else if (fb->format == PIXFORMAT_RGB565) {
        // Camera already decoded to RGB565 – convert to RGB888 ourselves
        size_t npix = (size_t)fb->width * fb->height;
        const uint16_t* p565 = (const uint16_t*)fb->buf;
        for (size_t i = 0; i < npix; i++) {
            uint16_t pixel = p565[i];
            s_rgb_buf[i * 3 + 0] = (pixel >> 8) & 0xF8;  // R
            s_rgb_buf[i * 3 + 1] = (pixel >> 3) & 0xFC;  // G
            s_rgb_buf[i * 3 + 2] = (pixel << 3) & 0xF8;  // B
        }
        out_len = npix * 3;
        decode_ok = true;
    } else if (fb->format == PIXFORMAT_RGB888) {
        memcpy(s_rgb_buf, fb->buf, (size_t)fb->width * fb->height * 3);
        out_len = (size_t)fb->width * fb->height * 3;
        decode_ok = true;
    } else {
        ESP_LOGE(TAG, "Unsupported camera pixel format: %d", fb->format);
    }

    int cam_w = fb->width;
    int cam_h = fb->height;
    esp_camera_fb_return(fb);  // return ASAP to free frame buffer slot

    if (!decode_ok) {
        ESP_LOGE(TAG, "Frame decode failed");
        return kTfLiteError;
    }

    // ── 3. Allocate temporary 224×224×3 resize destination ────────────────
    // We use SRAM-friendly stack allocation would be too big (150 KB),
    // so we reuse image_data directly as a temp RGB buffer, then overwrite
    // with int8 in-place (channel-by-channel, same size).
    // Instead, allocate a small scratch from PSRAM once (lazy).
    static uint8_t* s_resize_buf = nullptr;
    if (!s_resize_buf) {
        s_resize_buf = (uint8_t*)heap_caps_malloc(
            kNumCols * kNumRows * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_resize_buf) {
            ESP_LOGE(TAG, "Failed to allocate resize scratch buffer");
            return kTfLiteError;
        }
    }

    // ── 4. Centre-crop + nearest-neighbour resize to 224×224 ──────────────
    crop_and_resize_rgb888(s_rgb_buf, cam_w, cam_h,
                           s_resize_buf, image_width, image_height);

    // ── 5. Optionally fill display buffer (RGB565) ─────────────────────────
#if DISPLAY_SUPPORT
    if (s_display_buf) {
        rgb888_to_rgb565(s_resize_buf, s_display_buf, image_width, image_height);
    }
#endif

    // ── 6. Quantize RGB888 → int8_t and write to TFLite input tensor ───────
    //
    // Full-integer-quantized MobileNet INT8 models expect inputs in range
    // [-128, 127].  The standard conversion is:
    //
    //   int8_val = (uint8_pixel - 128)    (equivalent to XOR with 0x80)
    //
    // This assumes the quantization parameters of the input tensor are:
    //   scale = 1/128.0,  zero_point = -128
    // which is the standard for full-INT8 MobileNet models.
    //
    // IMPORTANT: if your model uses a different scale/zero_point (inspect with
    //   netron.app or python: interpreter.get_input_details()[0]['quantization'])
    // adjust the formula below accordingly:
    //   int8_val = round(float_pixel / 255.0 / scale) + zero_point
    //
    const uint8_t* rgb_pixel = s_resize_buf;
    const int total_pixels   = image_width * image_height * channels;

    for (int i = 0; i < total_pixels; i++) {
        // XOR 0x80  ==  subtract 128  (unsigned → signed conversion)
        image_data[i] = (int8_t)(rgb_pixel[i] ^ 0x80);
    }

    return kTfLiteOk;
#endif  // ESP_CAMERA_SUPPORTED
}
