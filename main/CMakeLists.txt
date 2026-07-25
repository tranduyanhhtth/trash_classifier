
#
# Main component of TF Micro project 'trash_classifier'.
#

idf_component_register(
    SRCS
        "classification_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "trash_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"

    PRIV_REQUIRES console static_images spi_flash esp_timer driver freertos
    INCLUDE_DIRS "")

