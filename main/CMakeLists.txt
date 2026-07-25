
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
        "wifi_ap.cc"
        "http_infer.cc"

    PRIV_REQUIRES
        console static_images spi_flash esp_timer driver freertos
        esp_wifi esp_http_server nvs_flash esp_netif
        espressif__esp_jpeg
    INCLUDE_DIRS "")

