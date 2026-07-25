/*
 * http_infer.c – HTTP Inference Server for TrashNet ESP32-S3
 *
 * Pipeline:
 *   Browser  ──POST /infer──►  Receive JPEG
 *                              → esp_jpeg_decode()   (RGB888)
 *                              → nn_resize()         (224×224)
 *                              → run_inference()     (TFLite Micro)
 *                              → JSON response       {"class":...}
 */

#include "http_infer.h"

extern "C" {
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "jpeg_decoder.h"       /* espressif__esp_jpeg managed component */
}

#include "model_settings.h"
#include "main_functions.h"

#include <string.h>
#include <stdio.h>

static const char *TAG = "http_infer";

/* Maximum uploaded JPEG size (512 KB – enough for any reasonable photo) */
#define MAX_JPEG_BYTES  (512 * 1024)

/* ─────────────────────────────────────────────────────────────────────────
 * Nearest-neighbour resize: src (sw×sh RGB888) → dst (dw×dh RGB888)
 * Fast, no floating-point, good enough for inference pre-processing.
 * ───────────────────────────────────────────────────────────────────────── */
static void nn_resize(const uint8_t *src, int sw, int sh,
                            uint8_t *dst, int dw, int dh)
{
    for (int dy = 0; dy < dh; dy++) {
        int sy = (dy * sh) / dh;
        for (int dx = 0; dx < dw; dx++) {
            int sx = (dx * sw) / dw;
            const uint8_t *s = src + (sy * sw + sx) * 3;
                  uint8_t *d = dst + (dy * dw + dx) * 3;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * HTML page – embedded directly (no SPIFFS needed)
 * ───────────────────────────────────────────────────────────────────────── */
static const char HTML[] =
"<!DOCTYPE html><html lang='vi'><head>"
"<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>TrashNet Classifier</title><style>"
"*{box-sizing:border-box;margin:0;padding:0}"
"body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;"
"     display:flex;align-items:center;justify-content:center;min-height:100vh}"
".card{background:#161b22;border:1px solid #30363d;border-radius:16px;"
"      padding:28px 24px;max-width:460px;width:95%;box-shadow:0 8px 32px #0008}"
"h1{text-align:center;color:#f78166;font-size:1.35em;margin-bottom:4px}"
".sub{text-align:center;color:#8b949e;font-size:.8em;margin-bottom:22px}"
".drop{display:flex;flex-direction:column;align-items:center;border:2px dashed #30363d;"
"      border-radius:12px;padding:28px 16px;cursor:pointer;transition:.2s;user-select:none}"
".drop:hover,.drop.over{border-color:#f78166;background:#1f242b}"
".drop .ico{font-size:2.4em;margin-bottom:8px}"
".drop .txt{color:#8b949e;font-size:.9em}"
"#file{display:none}"
"#preview{width:100%;max-height:220px;object-fit:contain;border-radius:8px;"
"         margin:14px 0;display:none}"
"button{width:100%;padding:13px;background:#238636;color:#fff;border:none;"
"       border-radius:8px;font-size:1em;cursor:pointer;margin-top:10px;transition:.2s}"
"button:hover:not(:disabled){background:#2ea043}"
"button:disabled{background:#21262d;color:#484f58;cursor:not-allowed}"
"#status{text-align:center;font-size:.85em;color:#8b949e;margin-top:8px;min-height:20px}"
"#result{margin-top:18px;padding:16px;background:#0d1117;border:1px solid #30363d;"
"        border-radius:10px;display:none}"
".top-cls{font-size:1.3em;font-weight:700;color:#79c0ff}"
".top-conf{font-size:.85em;color:#8b949e;margin:3px 0 14px}"
".row{display:flex;align-items:center;margin:5px 0;font-size:.82em}"
".rname{width:76px;color:#c9d1d9}"
".rbg{flex:1;background:#21262d;border-radius:4px;height:10px;overflow:hidden}"
".rfill{height:100%;border-radius:4px;transition:width .4s ease}"
".rpct{width:46px;text-align:right;color:#8b949e;padding-left:6px}"
"#err{color:#f85149;text-align:center;margin-top:10px;font-size:.88em;display:none}"
"</style></head><body><div class='card'>"
"<h1>&#x1F5D1; TrashNet Classifier</h1>"
"<p class='sub'>ESP32-S3 · TFLite Micro · 6 classes</p>"
"<div class='drop' id='drop' onclick='document.getElementById(\"file\").click()'>"
"  <div class='ico'>&#x1F4F7;</div>"
"  <div class='txt'>Click or drag &amp; drop an image</div>"
"</div>"
"<input type='file' id='file' accept='image/*'>"
"<img id='preview'>"
"<button id='btn' disabled onclick='runInfer()'>&#x26A1; Run Inference</button>"
"<div id='status'></div>"
"<div id='err'></div>"
"<div id='result'>"
"  <div class='top-cls' id='tcls'></div>"
"  <div class='top-conf' id='tconf'></div>"
"  <div id='bars'></div>"
"</div></div>"
"<script>"
"const COLORS={'cardboard':'#a0522d','glass':'#87ceeb','metal':'#c0c0c0',"
"              'paper':'#f5f5dc','plastic':'#50c878','trash':'#cd853f'};"
"const drop=document.getElementById('drop');"
"['dragover','dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{"
"  e.preventDefault();"
"  if(ev==='dragover')drop.classList.add('over');"
"  else{drop.classList.remove('over');"
"    if(ev==='drop'&&e.dataTransfer.files[0])setFile(e.dataTransfer.files[0]);}"
"}));"
"document.getElementById('file').onchange=e=>setFile(e.target.files[0]);"
"function setFile(f){"
"  document.getElementById('btn').disabled=false;"
"  document.getElementById('result').style.display='none';"
"  document.getElementById('err').style.display='none';"
"  document.getElementById('status').textContent='';"
"  const r=new FileReader();"
"  r.onload=e=>{"
"    const p=document.getElementById('preview');"
"    p.src=e.target.result;p.style.display='block';};"
"  r.readAsDataURL(f);"
"  window._file=f;}"
"async function runInfer(){"
"  const f=window._file;if(!f)return;"
"  const btn=document.getElementById('btn');"
"  btn.disabled=true;"
"  document.getElementById('status').textContent='Uploading & running inference...';"
"  document.getElementById('err').style.display='none';"
"  document.getElementById('result').style.display='none';"
"  try{"
"    const resp=await fetch('/infer',{method:'POST',"
"      headers:{'Content-Type':'application/octet-stream'},"
"      body:f});"
"    const j=await resp.json();"
"    if(j.error)throw new Error(j.error);"
"    document.getElementById('status').textContent='';"
"    document.getElementById('tcls').textContent=j.class+' '+getEmoji(j.class);"
"    document.getElementById('tconf').textContent="
"      'Confidence: '+(j.confidence*100).toFixed(1)+'%  \u00b7  '+j.inference_ms+' ms';"
"    const bars=document.getElementById('bars');"
"    bars.innerHTML='';"
"    Object.entries(j.scores)"
"      .sort((a,b)=>b[1]-a[1])"
"      .forEach(([k,v])=>{"
"        const pct=(v*100).toFixed(1);"
"        const col=COLORS[k.toLowerCase()]||'#58a6ff';"
"        bars.innerHTML+="
"          '<div class=\"row\"><div class=\"rname\">'+k+'</div>'"
"          +'<div class=\"rbg\"><div class=\"rfill\" style=\"width:'+pct+'%;background:'+col+'\"></div></div>'"
"          +'<div class=\"rpct\">'+pct+'%</div></div>';});"
"    document.getElementById('result').style.display='block';"
"  }catch(e){"
"    const el=document.getElementById('err');"
"    el.textContent='Error: '+e.message;el.style.display='block';"
"    document.getElementById('status').textContent='';}"
"  btn.disabled=false;}"
"function getEmoji(c){"
"  return{'Cardboard':'\U0001F4E6','Glass':'\U0001F37E','Metal':'\U0001F527',"
"         'Paper':'\U0001F4C4','Plastic':'\U0001F9F4','Trash':'\U0001F5D1'}[c]||'';}"
"</script></body></html>";

/* ─────────────────────────────────────────────────────────────────────────
 * GET /
 * ───────────────────────────────────────────────────────────────────────── */
static esp_err_t handle_root(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_send(req, HTML, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/* ─────────────────────────────────────────────────────────────────────────
 * GET /health
 * ───────────────────────────────────────────────────────────────────────── */
static esp_err_t handle_health(httpd_req_t *req)
{
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, "{\"status\":\"ok\",\"model\":\"TrashNet\",\"classes\":6}",
                    HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/* ─────────────────────────────────────────────────────────────────────────
 * POST /infer
 * Body: raw JPEG bytes (Content-Type: application/octet-stream)
 * ───────────────────────────────────────────────────────────────────────── */
static esp_err_t handle_infer(httpd_req_t *req)
{
    uint8_t *jpeg_buf = NULL;
    uint8_t *rgb_buf  = NULL;
    uint8_t *img224   = NULL;
    char     resp[600];

    /* ── Validate content length ─────────────────────────────────────── */
    int jpeg_len = req->content_len;
    if (jpeg_len <= 0 || jpeg_len > MAX_JPEG_BYTES) {
        snprintf(resp, sizeof(resp),
                 "{\"error\":\"bad content-length %d (max %d)\"}",
                 jpeg_len, MAX_JPEG_BYTES);
        goto send_json;
    }

    /* ── Receive JPEG body (PSRAM) ───────────────────────────────────── */
    jpeg_buf = (uint8_t *) heap_caps_malloc(jpeg_len,
                            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!jpeg_buf) {
        snprintf(resp, sizeof(resp), "{\"error\":\"OOM: jpeg buffer\"}");
        goto send_json;
    }

    {
        int received = 0, remaining = jpeg_len;
        while (remaining > 0) {
            int r = httpd_req_recv(req, (char *)(jpeg_buf + received), remaining);
            if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
            if (r <= 0) {
                snprintf(resp, sizeof(resp), "{\"error\":\"recv failed\"}");
                goto send_json;
            }
            received  += r;
            remaining -= r;
        }
        ESP_LOGI(TAG, "Received %d bytes JPEG", received);
    }

    /* ── Decode JPEG → RGB888 ────────────────────────────────────────── */
    {
        esp_jpeg_image_cfg_t cfg = {
            .indata      = jpeg_buf,
            .indata_size = (uint32_t)jpeg_len,
            .outbuf      = NULL,
            .outbuf_size = 0,
            .out_format  = JPEG_IMAGE_FORMAT_RGB888,
            .out_scale   = JPEG_IMAGE_SCALE_0,
        };
        esp_jpeg_image_output_t info;

        /* First pass: get dimensions & required output buffer size */
        if (esp_jpeg_get_image_info(&cfg, &info) != ESP_OK) {
            snprintf(resp, sizeof(resp), "{\"error\":\"JPEG header invalid\"}");
            goto send_json;
        }
        ESP_LOGI(TAG, "Image: %u×%u, decoded size: %zu bytes",
                 info.width, info.height, info.output_len);

        rgb_buf = (uint8_t *) heap_caps_malloc(info.output_len,
                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!rgb_buf) {
            snprintf(resp, sizeof(resp),
                     "{\"error\":\"OOM: rgb buffer (%zu bytes)\"}", info.output_len);
            goto send_json;
        }

        /* Second pass: decode */
        cfg.outbuf      = rgb_buf;
        cfg.outbuf_size = info.output_len;
        if (esp_jpeg_decode(&cfg, &info) != ESP_OK) {
            snprintf(resp, sizeof(resp), "{\"error\":\"JPEG decode failed\"}");
            goto send_json;
        }

        /* Done with JPEG bytes */
        heap_caps_free(jpeg_buf); jpeg_buf = NULL;

        /* ── Resize to 224×224 (nearest-neighbour) ─────────────────── */
        img224 = (uint8_t *) heap_caps_malloc(
                    kNumCols * kNumRows * kNumChannels,
                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!img224) {
            snprintf(resp, sizeof(resp), "{\"error\":\"OOM: resize buffer\"}");
            goto send_json;
        }

        nn_resize(rgb_buf, (int)info.width, (int)info.height,
                  img224, kNumCols, kNumRows);

        heap_caps_free(rgb_buf); rgb_buf = NULL;
    }

    /* ── Run TFLite inference ────────────────────────────────────────── */
    ESP_LOGI(TAG, "Calling run_inference()...");
    run_inference((void *)img224);
    heap_caps_free(img224); img224 = NULL;

    /* Check if inference actually ran */
    if (!g_last_result.valid) {
        snprintf(resp, sizeof(resp),
                 "{\"error\":\"Inference failed: interpreter not ready. "
                 "Check serial log via USB Serial/JTAG.\"}");
        goto send_json;
    }

    /* ── Build JSON response ─────────────────────────────────────────── */
    {
        const InferenceResult_t *R = &g_last_result;

        /* scores object */
        char scores[280];
        int  p = snprintf(scores, sizeof(scores), "{");
        for (int i = 0; i < kCategoryCount; i++) {
            p += snprintf(scores + p, sizeof(scores) - p,
                          "\"%s\":%.4f%s",
                          kCategoryLabels[i], R->scores[i],
                          i < kCategoryCount - 1 ? "," : "");
        }
        snprintf(scores + p, sizeof(scores) - p, "}");

        snprintf(resp, sizeof(resp),
                 "{\"class\":\"%s\",\"confidence\":%.4f,"
                 "\"inference_ms\":%lld,\"scores\":%s}",
                 kCategoryLabels[R->top_index],
                 R->top_score,
                 (long long)R->inference_ms,
                 scores);

        ESP_LOGI(TAG, "Result: %s (%.1f%%, %lld ms)",
                 kCategoryLabels[R->top_index],
                 R->top_score * 100.0f,
                 (long long)R->inference_ms);
    }

send_json:
    if (jpeg_buf) heap_caps_free(jpeg_buf);
    if (rgb_buf)  heap_caps_free(rgb_buf);
    if (img224)   heap_caps_free(img224);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Start HTTP server
 * ───────────────────────────────────────────────────────────────────────── */
esp_err_t http_infer_server_start(void)
{
    httpd_config_t config        = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers      = 8;
    config.recv_wait_timeout     = 30;   /* seconds */
    config.send_wait_timeout     = 30;
    config.stack_size            = 8192;

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server");
        return ESP_FAIL;
    }

    static const httpd_uri_t routes[] = {
        { .uri = "/",       .method = HTTP_GET,  .handler = handle_root   },
        { .uri = "/health", .method = HTTP_GET,  .handler = handle_health },
        { .uri = "/infer",  .method = HTTP_POST, .handler = handle_infer  },
    };

    for (int i = 0; i < 3; i++) {
        httpd_register_uri_handler(server, &routes[i]);
    }

    ESP_LOGI(TAG, "HTTP server started → http://192.168.4.1");
    return ESP_OK;
}
