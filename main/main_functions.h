/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * main_functions.h  –  Public interface for setup() and loop().
 */

#pragma once


// One-time initialisation: load model, init camera, allocate tensor arena.
void setup();

// Continuous inference loop: capture frame, invoke model, report result.
void loop();

// CLI-triggered single inference on a provided raw pixel buffer.
// Declared with C linkage to match esp_main.h extern "C" declaration.
#ifdef __cplusplus
extern "C" {
#endif
void run_inference(void* ptr);
#ifdef __cplusplus
}
#endif

