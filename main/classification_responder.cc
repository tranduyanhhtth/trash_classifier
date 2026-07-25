/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * classification_responder.cc  –  Output handler implementation.
 *
 * Outputs a formatted table to the serial monitor after each inference:
 *
 *   ┌─────────────────────────────────────────┐
 *   │ TrashNet Inference  (42 ms)             │
 *   ├──────────────┬────────┬─────────────────┤
 *   │ Class        │  Score │ Bar             │
 *   ├──────────────┼────────┼─────────────────┤
 *   │ Cardboard    │   2.1% │ ▒               │
 *   │ Glass        │   1.8% │ ▒               │
 *   │ Metal        │   3.4% │ ▒               │
 *   │ Paper        │  87.6% │ ███████████████ │ ← WINNER
 *   │ Plastic      │   4.1% │ ▒               │
 *   │ Trash        │   1.0% │ ▒               │
 *   └──────────────┴────────┴─────────────────┘
 *
 * Extend this file to add GPIO/LED, UART, or MQTT responses.
 */

#include "classification_responder.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "esp_log.h"
#include "driver/gpio.h"

static const char* TAG = "classifier";

// ---------------------------------------------------------------------------
// Optional LED GPIO output
// ---------------------------------------------------------------------------
// Uncomment and configure a GPIO to blink on successful classification.
// #define LED_GPIO  GPIO_NUM_2

// ---------------------------------------------------------------------------
// RespondToClassification  –  main entry point
// ---------------------------------------------------------------------------
void RespondToClassification(const float scores[kCategoryCount],
                              int         top_index,
                              float       top_score,
                              int64_t     inference_ms) {
    // ── Header ────────────────────────────────────────────────────────────
    ESP_LOGI(TAG, "─────────────────────────────────────────");
    ESP_LOGI(TAG, " TrashNet Inference  (%" PRId64 " ms)", inference_ms);
    ESP_LOGI(TAG, "─────────────────────────────────────────");

    // ── Per-class score bar chart ──────────────────────────────────────────
    const int   BAR_MAX    = 20;    // max bar width in chars

    for (int i = 0; i < kCategoryCount; i++) {
        float pct     = scores[i] * 100.0f;
        int   bar_len = (int)(scores[i] * BAR_MAX + 0.5f);
        if (bar_len > BAR_MAX) bar_len = BAR_MAX;

        // Build bar string
        char bar[BAR_MAX + 2] = {0};
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        for (int b = bar_len; b < BAR_MAX; b++) bar[b] = '.';

        bool is_winner = (i == top_index);
        if (is_winner) {
            ESP_LOGI(TAG, " >> %-10s %5.1f%% [%s] <<",
                     kCategoryLabels[i], pct, bar);
        } else {
            ESP_LOGI(TAG, "    %-10s %5.1f%% [%s]",
                     kCategoryLabels[i], pct, bar);
        }
    }

    ESP_LOGI(TAG, "─────────────────────────────────────────");

    // ── Final verdict ──────────────────────────────────────────────────────
    if (top_score >= kConfidenceThreshold) {
        ESP_LOGI(TAG, " RESULT: %s  %.1f%%  %s",
                 kCategoryEmoji[top_index],
                 top_score * 100.0f,
                 kCategoryLabels[top_index]);
    } else {
        ESP_LOGW(TAG, " RESULT: LOW CONFIDENCE (%.1f%% < threshold %.0f%%)",
                 top_score * 100.0f,
                 kConfidenceThreshold * 100.0f);
    }

    ESP_LOGI(TAG, "─────────────────────────────────────────\n");

    // ── Optional GPIO / LED ────────────────────────────────────────────────
#ifdef LED_GPIO
    if (top_score >= kConfidenceThreshold) {
        gpio_set_level(LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(LED_GPIO, 0);
    }
#endif
}
