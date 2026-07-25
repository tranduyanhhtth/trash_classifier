/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * main_functions.cc  –  TFLite Micro inference engine core.
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  Architecture Overview                                               │
 * │                                                                      │
 * │  SRAM (512 KB internal)                                             │
 * │  ┌──────────────────────────┐                                        │
 * │  │   Tensor Arena           │  ← kTensorArenaSize                   │
 * │  │   (static, internal RAM) │    ~350 KB for 224×224 MobileNetV1    │
 * │  └──────────────────────────┘                                        │
 * │                                                                      │
 * │  PSRAM (2/8 MB external SPI RAM)                                    │
 * │  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐               │
 * │  │ Camera JPEG  │ │ RGB888 dec   │ │ Resize scratch│               │
 * │  │ frame buffer │ │ 320×240×3    │ │ 224×224×3     │               │
 * │  │ (fb_count=2) │ │ ~230 KB      │ │ ~150 KB       │               │
 * │  └──────────────┘ └──────────────┘ └───────────────┘               │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  Inference Pipeline (each loop() iteration)                          │
 * │                                                                      │
 * │  1. GetImage()        JPEG→RGB888→crop→resize→int8_t  (~15-30 ms)  │
 * │  2. interpreter->Invoke()  forward pass                (~80-200 ms) │
 * │  3. dequantize output      int8→float                  (<1 ms)      │
 * │  4. argmax + softmax       pick winner class           (<1 ms)      │
 * │  5. RespondToClassification()  log to UART             (<1 ms)      │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * Key TFLite Micro API notes:
 *   - MicroMutableOpResolver must list EVERY op used by your model.
 *     Check ops in Netron (https://netron.app) or with the Python API:
 *       for op in interpreter._get_ops_details(): print(op)
 *   - tensor_arena must remain valid for the lifetime of the interpreter.
 *   - kTensorArenaSize: start high, then call interpreter->arena_used_bytes()
 *     after AllocateTensors() to find the minimum required size.
 */

#include "main_functions.h"
#include "model_settings.h"
#include "trash_model_data.h"
#include "image_provider.h"
#include "classification_responder.h"

// TFLite Micro core
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// FreeRTOS / ESP-IDF
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_main.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

static const char* TAG = "main_functions";

// ---------------------------------------------------------------------------
// TFLite Micro globals (static – lifetime of the application)
// ---------------------------------------------------------------------------
namespace {

// Parsed model reference (points into g_trash_model_data flash array)
const tflite::Model* s_model = nullptr;

// TFLite Micro interpreter
tflite::MicroInterpreter* s_interpreter = nullptr;

// Pointer to the input tensor (convenience alias into the arena)
TfLiteTensor* s_input = nullptr;

// ---------------------------------------------------------------------------
// Tensor Arena size selection:
//
//  INT8 model  (full quantized):   input 147KB + activations ~200KB ≈ 350KB
//  FLOAT32 model (hybrid quant):   input 588KB + activations ~200KB ≈ 788KB
//
// We size for FLOAT32 worst-case (768 KB) and allocate from PSRAM.
// The ESP32-S3 N16R8 has 8 MB PSRAM so this is fine.
// Inference is ~3× slower from PSRAM vs SRAM, but accuracy is unaffected.
//
// If you convert the model to full-INT8, you can drop this to 400 KB and
// use SRAM for faster inference.
// ---------------------------------------------------------------------------

#if CONFIG_NN_OPTIMIZED
constexpr int kScratchBufSize = 60 * 1024;  // 60 KB
#else
constexpr int kScratchBufSize = 0;
#endif

// 768 KB covers FLOAT32 I/O MobileNetV1 224×224 (input alone = 588 KB)
constexpr int kTensorArenaSize = 768 * 1024 + kScratchBufSize;

static uint8_t* s_tensor_arena = nullptr;

}  // anonymous namespace

// ─── Shared inference result (read by http_infer.c) ──────────────────────────
// Must be defined here (C++ TU) but declared extern "C" in main_functions.h.
extern "C" InferenceResult_t g_last_result = {};

// ---------------------------------------------------------------------------
// setup()  –  called once from tf_main() before the inference loop
// ---------------------------------------------------------------------------
void setup() {
    ESP_LOGI(TAG, "TrashNet Classifier initialising...");
    ESP_LOGI(TAG, "  Model input : %d×%d×%d (INT8)", kNumCols, kNumRows, kNumChannels);
    ESP_LOGI(TAG, "  Categories  : %d", kCategoryCount);
    for (int i = 0; i < kCategoryCount; i++) {
        ESP_LOGI(TAG, "    [%d] %s", i, kCategoryLabels[i]);
    }

    // ── 1. Map the TFLite model from flash ─────────────────────────────────
    // GetModel() does NOT copy data – it points directly into the const array
    // stored in DROM.  Very fast and zero-copy.
    s_model = tflite::GetModel(g_trash_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch: model=%d, supported=%d",
                    s_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "Model loaded from flash (%u bytes)", g_trash_model_data_len);

    // ── 2. Allocate Tensor Arena ───────────────────────────────────────────
    // PSRAM first: FLOAT32 I/O model needs 588KB for input alone, which exceeds
    // the typical ~300KB of free internal SRAM after FreeRTOS + WiFi stack.
    if (!s_tensor_arena) {
        s_tensor_arena = (uint8_t*)heap_caps_malloc(
            kTensorArenaSize,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    // Fallback: internal SRAM (only works for INT8 models with smaller arena)
    if (!s_tensor_arena) {
        ESP_LOGW(TAG, "PSRAM arena failed, retrying with internal SRAM");
        s_tensor_arena = (uint8_t*)heap_caps_malloc(
            kTensorArenaSize,
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (!s_tensor_arena) {
        ESP_LOGE(TAG, "FATAL: Cannot allocate tensor arena (%d bytes)",
                 kTensorArenaSize);
        return;
    }
    ESP_LOGI(TAG, "Tensor arena: %d KB allocated from %s",
             kTensorArenaSize / 1024,
             heap_caps_get_allocated_size(s_tensor_arena) > 0 ? "PSRAM" : "SRAM");

    // ── 3. Register TFLite Micro operations ───────────────────────────────
    //
    // MobileNetV1 architecture requires these ops (verify in Netron):
    //   Conv2D             → first conv + 1×1 pointwise convs
    //   DepthwiseConv2D    → depthwise separable convs (the key MobileNet op)
    //   FullyConnected     → final classifier head (Dense layer)
    //   AveragePool2D      → global average pooling before classifier head
    //   Reshape            → flatten after pooling
    //   Softmax            → final classification output
    //   Add                → residual connections (MobileNetV2 only; safe to include)
    //   Quantize           → dequantize input layer (full-INT8 models)
    //   Dequantize         → quantize output layer  (full-INT8 models)
    //
    static tflite::MicroMutableOpResolver<11> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddFullyConnected();   // ← classifier head (Dense layer)
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddPad();

    // ── 4. Build the interpreter ──────────────────────────────────────────
    static tflite::MicroInterpreter static_interpreter(
        s_model, micro_op_resolver, s_tensor_arena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    // ── 5. Allocate tensors ───────────────────────────────────────────────
    // This partitions the tensor arena into sub-regions for each tensor.
    // If it fails, increase kTensorArenaSize.
    TfLiteStatus status = s_interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() FAILED.");
        ESP_LOGE(TAG, "  Increase kTensorArenaSize (currently %d KB)",
                 kTensorArenaSize / 1024);
        return;
    }

    // Report actual arena usage – helps tune kTensorArenaSize
    size_t used = s_interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena used: %u / %d bytes (%.0f%%)",
             (unsigned)used, kTensorArenaSize,
             100.0f * (float)used / (float)kTensorArenaSize);

    // ── 6. Inspect input tensor ───────────────────────────────────────────
    s_input = s_interpreter->input(0);
    ESP_LOGI(TAG, "Input tensor: dims=%d [%d,%d,%d,%d]  type=%d",
             s_input->dims->size,
             s_input->dims->data[0],  // batch
             s_input->dims->data[1],  // height
             s_input->dims->data[2],  // width
             s_input->dims->data[3],  // channels
             s_input->type);
    ESP_LOGI(TAG, "Input quantization: scale=%.6f  zero_point=%d",
             s_input->params.scale,
             s_input->params.zero_point);

    // Sanity check dimensions match compile-time constants
    if (s_input->dims->data[1] != kNumRows ||
        s_input->dims->data[2] != kNumCols ||
        s_input->dims->data[3] != kNumChannels) {
        ESP_LOGE(TAG, "FATAL: Input tensor shape mismatch!");
        ESP_LOGE(TAG, "  Expected : [1,%d,%d,%d]", kNumRows, kNumCols, kNumChannels);
        ESP_LOGE(TAG, "  Got      : [%d,%d,%d,%d]",
                 s_input->dims->data[0], s_input->dims->data[1],
                 s_input->dims->data[2], s_input->dims->data[3]);
        return;
    }
    if (s_input->type != kTfLiteInt8) {
        ESP_LOGW(TAG, "Input tensor type is NOT INT8 (got %d) – FLOAT32 path will be used.",
                 s_input->type);
        ESP_LOGW(TAG, "  For better performance, re-export with full INT8 quantization.");
        // Do NOT return – run_inference() handles FLOAT32 input correctly.
    }

    ESP_LOGI(TAG, "Setup complete – HTTP inference server ready");

    // ── 7. [CAMERA MODE] Camera initialisation ────────────────────────────
    // #ifndef CLI_ONLY_INFERENCE
    //     TfLiteStatus cam_status = InitCamera();
    //     if (cam_status != kTfLiteOk) {
    //         ESP_LOGE(TAG, "InitCamera() failed");
    //         return;
    //     }
    // #endif
}

// ---------------------------------------------------------------------------
// Helper: dequantize int8 output tensor → float probabilities
// ---------------------------------------------------------------------------
/**
 * @brief  Convert raw INT8 logits from the output tensor into float
 *         probabilities [0.0, 1.0] using the tensor's quantization parameters.
 *
 * Formula:
 *   float_val = (int8_val - zero_point) × scale
 *
 * NOTE: For full-INT8 models the output IS already softmax (the op is fused
 *       into the model).  We just dequantize and we get probabilities.
 *       The values will sum to approximately 1.0.
 *
 * @param output_tensor  Pointer to the interpreter output tensor.
 * @param out_scores     Float array of length kCategoryCount to fill.
 */
static void dequantize_output(TfLiteTensor* output_tensor,
                               float out_scores[kCategoryCount]) {
    const float scale      = output_tensor->params.scale;
    const int   zero_point = output_tensor->params.zero_point;

    if (output_tensor->type == kTfLiteInt8) {
        const int8_t* raw = output_tensor->data.int8;
        float sum = 0.0f;
        for (int i = 0; i < kCategoryCount; i++) {
            out_scores[i] = (raw[i] - zero_point) * scale;
            // Clamp to [0, 1] to guard against degenerate quantization params
            if (out_scores[i] < 0.0f) out_scores[i] = 0.0f;
            if (out_scores[i] > 1.0f) out_scores[i] = 1.0f;
            sum += out_scores[i];
        }
        // Renormalize if numerical drift causes sum ≠ 1.0
        if (sum > 0.0f) {
            for (int i = 0; i < kCategoryCount; i++) out_scores[i] /= sum;
        }
    } else if (output_tensor->type == kTfLiteFloat32) {
        // Hybrid quantized model – output is already float
        const float* raw = output_tensor->data.f;
        for (int i = 0; i < kCategoryCount; i++) {
            out_scores[i] = raw[i];
        }
    } else {
        ESP_LOGE("dequant", "Unsupported output type: %d", output_tensor->type);
        memset(out_scores, 0, sizeof(float) * kCategoryCount);
    }
}

// ---------------------------------------------------------------------------
// Helper: argmax – find the winning class
// ---------------------------------------------------------------------------
static int argmax(const float scores[], int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (scores[i] > scores[best]) best = i;
    }
    return best;
}

// ---------------------------------------------------------------------------
// loop()  –  called continuously from tf_main() after setup()
// ---------------------------------------------------------------------------
#ifndef CLI_ONLY_INFERENCE
void loop() {
    if (!s_interpreter || !s_input) {
        ESP_LOGE(TAG, "Interpreter not initialised – skipping");
        vTaskDelay(pdMS_TO_TICKS(1000));
        return;
    }

    // ── Step 1: Capture image ──────────────────────────────────────────────
    // GetImage() fills s_input->data.int8 directly with pre-processed pixels.
    TfLiteStatus img_status = GetImage(kNumCols, kNumRows, kNumChannels,
                                        s_input->data.int8);
    if (img_status != kTfLiteOk) {
        ESP_LOGE(TAG, "GetImage() failed – retrying");
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    // ── Step 2: Run inference ──────────────────────────────────────────────
    int64_t t0 = esp_timer_get_time();                   // µs timestamp before

    TfLiteStatus invoke_status = s_interpreter->Invoke();

    int64_t t1 = esp_timer_get_time();                   // µs timestamp after
    int64_t inference_ms = (t1 - t0) / 1000LL;          // µs → ms

    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    // ── Step 3: Dequantize output tensor ──────────────────────────────────
    TfLiteTensor* output = s_interpreter->output(0);
    float scores[kCategoryCount] = {};
    dequantize_output(output, scores);

    // ── Step 4: Find winning class ────────────────────────────────────────
    int   top_index = argmax(scores, kCategoryCount);
    float top_score = scores[top_index];

    // ── Step 5: Report result ─────────────────────────────────────────────
    RespondToClassification(scores, top_index, top_score, inference_ms);

    // ── Step 6: Yield to IDLE task (prevents watchdog timeout) ───────────
    vTaskDelay(pdMS_TO_TICKS(1));
}
#endif  // CLI_ONLY_INFERENCE

// ---------------------------------------------------------------------------
// run_inference()  –  CLI-triggered single inference (raw uint8 pixel buffer)
// ---------------------------------------------------------------------------
/**
 * CLI workflow (used with CONFIG_CLI_ONLY_INFERENCE):
 *   1. Host sends raw pixel data over UART/USB at (kNumCols×kNumRows×kNumChannels) bytes.
 *   2. esp_cli.c calls run_inference(ptr) with a pointer to that buffer.
 *   3. Pixels are converted from uint8 to int8, inference runs, result logged.
 *
 * @param ptr  Pointer to uint8_t buffer of kNumCols×kNumRows×kNumChannels bytes (RGB).
 */
void run_inference(void* ptr) {
    if (!s_interpreter || !s_input) {
        ESP_LOGE(TAG, "Interpreter not ready – setup() may have failed");
        // Populate error result so HTTP returns meaningful JSON
        memset(g_last_result.scores, 0, sizeof(g_last_result.scores));
        g_last_result.top_index = 0;
        g_last_result.top_score = 0.0f;
        g_last_result.inference_ms = -1;
        g_last_result.valid = false;
        return;
    }

    const uint8_t* raw_pixels = (const uint8_t*)ptr;
    const int      total_bytes = kNumCols * kNumRows * kNumChannels;

    // ── Fill input tensor (handles both FLOAT32 and INT8 models) ─────────
    if (s_input->type == kTfLiteFloat32) {
        // Hybrid quantized model: weights INT8, I/O FLOAT32
        // Normalize uint8 [0,255] → float [0.0, 1.0]
        float* dst = s_input->data.f;
        for (int i = 0; i < total_bytes; i++) {
            dst[i] = raw_pixels[i] / 255.0f;
        }
        ESP_LOGD(TAG, "Input type: FLOAT32, normalized [0,1]");
    } else if (s_input->type == kTfLiteInt8) {
        // Full INT8 model: uint8 [0,255] → int8 [-128,127] via XOR 0x80
        // Valid only if quantization: scale=1/128, zero_point=-128
        for (int i = 0; i < total_bytes; i++) {
            s_input->data.int8[i] = (int8_t)(raw_pixels[i] ^ 0x80);
        }
        ESP_LOGD(TAG, "Input type: INT8, scale=%.4f zp=%d",
                 s_input->params.scale, s_input->params.zero_point);
    } else {
        ESP_LOGE(TAG, "Unsupported input tensor type: %d", s_input->type);
        return;
    }

    // ── Invoke ────────────────────────────────────────────────────────────
    int64_t t0 = esp_timer_get_time();

    if (kTfLiteOk != s_interpreter->Invoke()) {
        ESP_LOGE(TAG, "Invoke() failed in run_inference()");
        return;
    }

    int64_t inference_ms = (esp_timer_get_time() - t0) / 1000LL;

    // ── Decode output ─────────────────────────────────────────────────────
    TfLiteTensor* output = s_interpreter->output(0);
    float scores[kCategoryCount] = {};
    dequantize_output(output, scores);

    int   top_index = argmax(scores, kCategoryCount);
    float top_score = scores[top_index];

    ESP_LOGI(TAG, "Inference: %s (%.1f%%)  %lld ms",
             kCategoryLabels[top_index], top_score * 100.0f, (long long)inference_ms);

    RespondToClassification(scores, top_index, top_score, inference_ms);

    // ── Persist result for HTTP server ────────────────────────────────────
    for (int i = 0; i < kCategoryCount; i++) g_last_result.scores[i] = scores[i];
    g_last_result.top_index    = top_index;
    g_last_result.top_score    = top_score;
    g_last_result.inference_ms = inference_ms;
    g_last_result.valid        = true;
}
