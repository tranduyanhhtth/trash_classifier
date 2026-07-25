/*
 * wifi_ap.c – ESP32-S3 WiFi Access Point (SoftAP)
 *
 * Tạo hotspot để user kết nối và truy cập HTTP inference server tại
 * http://192.168.4.1 mà không cần router.
 */

#include "wifi_ap.h"

extern "C" {
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
}
#include "sdkconfig.h"

static const char *TAG = "wifi_ap";

esp_err_t wifi_ap_start(void)
{
    /* NVS init (needed by WiFi driver) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t ap_config = {
        .ap = {
            .ssid           = CONFIG_INFER_AP_SSID,
            .password       = CONFIG_INFER_AP_PASSWORD,
            .ssid_len       = 0,
            .channel        = 1,
            .authmode       = WIFI_AUTH_WPA2_PSK,
            .max_connection = 4,
            .pmf_cfg        = { .required = false },
        },
    };

    /* Open network if password is empty */
    if (strlen(CONFIG_INFER_AP_PASSWORD) == 0) {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, " WiFi AP started");
    ESP_LOGI(TAG, " SSID    : %s", CONFIG_INFER_AP_SSID);
    ESP_LOGI(TAG, " Password: %s", CONFIG_INFER_AP_PASSWORD);
    ESP_LOGI(TAG, " URL     : http://192.168.4.1");
    ESP_LOGI(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    return ESP_OK;
}
