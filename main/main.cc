/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * main.cc  –  FreeRTOS entry point.
 *
 * Creates a dedicated high-priority task for TFLite inference.
 * Stack size: 8 KB is sufficient for our inference loop (all heavy memory
 * lives in the tensor arena and PSRAM heap, not on the stack).
 */

#include "main_functions.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_main.h"
#include "esp_cli.h"
#include "wifi_ap.h"
#include "http_infer.h"

static const char* TAG = "main";

void tf_main(void) {
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, " TrashNet Classifier  –  ESP32-S3 + OV2640  ");
    ESP_LOGI(TAG, " github.com/espressif/esp-tflite-micro       ");
    ESP_LOGI(TAG, "==============================================");

    setup();          // TFLite init (model + camera)
    wifi_ap_start();              // Create WiFi hotspot
    http_infer_server_start();    // Start HTTP server on :80
    esp_cli_start();  // start CLI for optional serial-triggered inference

#if CLI_ONLY_INFERENCE
    // In CLI mode, wait for host to send image data over UART
    esp_cli_register_inference_command();
    vTaskDelay(portMAX_DELAY);
#else
    // Camera live-capture mode: run inference as fast as possible
    while (true) {
        loop();
    }
#endif
}

extern "C" void app_main() {
    // Spawn the TFLite task.
    // Priority 8 – higher than default (5) to ensure timely inference.
    // Stack 8 KB – increase to 16 KB if you observe stack overflow panics.
    xTaskCreate(
        (TaskFunction_t)&tf_main,  // function
        "tf_main",                  // task name
        8 * 1024,                   // stack depth in bytes (8 KB)
        NULL,                       // parameter
        8,                          // priority (0=lowest, 24=highest on ESP-IDF)
        NULL                        // task handle (not needed)
    );
    vTaskDelete(NULL);  // delete the app_main bootstrap task
}
