cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS static_images)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Helper function to set required environment variables
function(add_bsp SDKCONFIG BSP)
    string(REGEX MATCH "CONFIG_${BSP}=y" REGEX_RESULT ${SDKCONFIG})
    if (REGEX_RESULT)
        set(ENV{${BSP}} 1)
    endif()
endfunction()

# Define all BSP variables
set(ENV{TFLITE_USE_BSP_S3_EYE}  0)
set(ENV{TFLITE_USE_BSP_KORVO_2} 0)
set(ENV{TFLITE_USE_BSP_KALUGA}  0)

# Set correct var to 1 based on sdkconfig
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/sdkconfig)
    file(READ ${CMAKE_CURRENT_LIST_DIR}/sdkconfig SDKCONFIG_RULE)
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_S3_EYE")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KORVO_2")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KALUGA")
endif()

project(trash_classifier)
