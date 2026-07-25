/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * image_provider.h  –  Camera interface declaration.
 *
 * Provides:
 *   InitCamera()  – configure and start the OV2640
 *   GetImage()    – capture a frame and write pre-processed int8_t pixels
 *                   into the TFLite input buffer
 */

#pragma once

#include "tensorflow/lite/c/common.h"

// ---------------------------------------------------------------------------
// Camera initialisation
// ---------------------------------------------------------------------------
/**
 * @brief  Initialise OV2640 camera using pin configuration from app_camera_esp.h.
 *         Must be called once from setup() before GetImage().
 * @return kTfLiteOk on success, kTfLiteError on failure.
 */
TfLiteStatus InitCamera();

// ---------------------------------------------------------------------------
// Frame capture + pre-processing
// ---------------------------------------------------------------------------
/**
 * @brief  Capture one frame from OV2640, resize/crop to (image_width x image_height),
 *         convert colour channels and write quantized int8_t pixels into image_data.
 *
 * @param image_width   Target width  in pixels (must equal kNumCols  = 224).
 * @param image_height  Target height in pixels (must equal kNumRows  = 224).
 * @param channels      Number of colour channels (must equal kNumChannels = 3).
 * @param image_data    Output buffer – must be pre-allocated with
 *                      (image_width * image_height * channels) bytes.
 *                      Values are written as int8_t (quantized: pixel - 128).
 * @return kTfLiteOk on success, kTfLiteError on failure.
 */
TfLiteStatus GetImage(int image_width, int image_height, int channels,
                      int8_t* image_data);

// ---------------------------------------------------------------------------
// Display helper (only when DISPLAY_SUPPORT is enabled)
// ---------------------------------------------------------------------------
/**
 * @brief  Returns pointer to internal RGB565 display buffer.
 *         Buffer is valid after the first successful GetImage() call.
 */
void* image_provider_get_display_buf();
