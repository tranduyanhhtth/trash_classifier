/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * classification_responder.h  –  Output handler declaration.
 *
 * After each inference the scores array is passed here for:
 *   - Serial monitor formatted output
 *   - LED / GPIO signalling (optional)
 *   - Future: MQTT / HTTP reporting
 */

#pragma once

#include <cstdint>
#include "model_settings.h"


// ---------------------------------------------------------------------------
// Main responder called after every inference
// ---------------------------------------------------------------------------
/**
 * @brief  Process softmax output scores and report the classification result.
 *
 * @param scores      Float array of length kCategoryCount (6).
 *                    Each element is the dequantized probability [0.0, 1.0].
 * @param top_index   Index of the highest-scoring class  [0..5].
 * @param top_score   Probability of the top class        [0.0, 1.0].
 * @param inference_ms  Time taken by interpreter->Invoke() in milliseconds.
 */
void RespondToClassification(const float scores[kCategoryCount],
                              int         top_index,
                              float       top_score,
                              int64_t     inference_ms);
