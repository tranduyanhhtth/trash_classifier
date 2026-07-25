/*
 * TrashNet 6-Class Classifier for ESP32-S3 + OV2640
 * ===================================================
 * Category label strings and emoji tags.
 */

#include "model_settings.h"

// TrashNet 6 labels – index order must match model output logit order
const char* kCategoryLabels[kCategoryCount] = {
    "Cardboard",  // 0
    "Glass",      // 1
    "Metal",      // 2
    "Paper",      // 3
    "Plastic",    // 4
    "Trash",      // 5
};

// Optional emoji / ASCII art for richer serial-monitor output
const char* kCategoryEmoji[kCategoryCount] = {
    "[CARDBOARD]",  // 0
    "[GLASS]",      // 1
    "[METAL]",      // 2
    "[PAPER]",      // 3
    "[PLASTIC]",    // 4
    "[TRASH]",      // 5
};
