/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * Model settings header – defines all compile-time constants that
 * must match the trained TFLite model exactly.
 *
 * Model: quantized_and_pruned_model.tflite (MobileNetV1 1.0)
 *   Input  : [1, 224, 224, 3]  INT8  (full-integer quantized)
 *   Output : [1, 6]            INT8  (softmax logits, dequantize in code)
 *
 * Classes (TrashNet order):
 *   0 = cardboard   1 = glass   2 = metal
 *   3 = paper       4 = plastic 5 = trash
 */

#pragma once

// ---------------------------------------------------------------------------
// Image dimensions – MUST match the model's input tensor shape
// ---------------------------------------------------------------------------
constexpr int kNumCols     = 224;   // width  in pixels
constexpr int kNumRows     = 224;   // height in pixels
constexpr int kNumChannels = 3;     // RGB

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;  // 150,528

// ---------------------------------------------------------------------------
// Classification labels
// ---------------------------------------------------------------------------
constexpr int kCategoryCount = 6;

// Index constants for each category
constexpr int kCardboardIndex = 0;
constexpr int kGlassIndex     = 1;
constexpr int kMetalIndex     = 2;
constexpr int kPaperIndex     = 3;
constexpr int kPlasticIndex   = 4;
constexpr int kTrashIndex     = 5;

// Human-readable label array (declared in model_settings.cc)
extern const char* kCategoryLabels[kCategoryCount];

// Emoji/icon tags for serial monitor output
extern const char* kCategoryEmoji[kCategoryCount];

// ---------------------------------------------------------------------------
// Inference confidence threshold
// ---------------------------------------------------------------------------
// Only report a detection when the winning class probability exceeds this.
// 0.60 = 60% – adjust based on real-world testing on your device.
constexpr float kConfidenceThreshold = 0.60f;
