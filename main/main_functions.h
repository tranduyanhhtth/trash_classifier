/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * main_functions.h  –  Public interface for setup() and loop().
 */

#pragma once

#include "model_settings.h"
#include <stdint.h>
#include <stdbool.h>

// ─── Inference result (populated by run_inference) ───────────────────────────
// Accessible from C (http_infer.c) and C++ (main_functions.cc).
typedef struct {
    float   scores[6];   /* softmax probability per class [0,1] */
    int     top_index;   /* argmax index */
    float   top_score;   /* scores[top_index] */
    int64_t inference_ms;
    bool    valid;       /* true once run_inference() has completed at least once */
} InferenceResult_t;

#ifdef __cplusplus
extern "C" {
#endif

extern InferenceResult_t g_last_result;  /* defined in main_functions.cc */

// One-time initialisation: load model, init camera, allocate tensor arena.
void setup();

// Continuous inference loop: capture frame, invoke model, report result.
void loop();

// CLI/HTTP-triggered single inference on a provided raw pixel buffer.
void run_inference(void* ptr);

#ifdef __cplusplus
}
#endif
