#include "esp_camera.h"
#include <WiFi.h>
#include <Arduino.h>
#include "Chirale_TensorFlowLite.h"
#include "dermax_model.h"
#include "board_config.h"
#include <TJpg_Decoder.h>   // For JPEG decoding

// ===========================
// WiFi credentials
// ===========================
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";

// ===========================
// TensorFlow Lite setup
// ===========================
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

Chirale::MicroInterpreter* interpreter;

#define INPUT_W 224
#define INPUT_H 224
#define INPUT_C 3
#define INPUT_SIZE (INPUT_W * INPUT_H * INPUT_C)
#define NUM_CLASSES 7

// ===========================
// Function prototypes
// ===========================
void startCameraServer();
void setupLedFlash();
void runDermaxInference(camera_fb_t* fb);
void processCameraFrame();

// ===========================
// Setup
// ===========================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // --- TensorFlow setup ---
  interpreter = new Chirale::MicroInterpreter(dermax_model, tensor_arena, kTensorArenaSize);
  if (!interpreter->AllocateTensors()) {
    Serial.println("Tensor allocation failed!");
    while (true);
  }
  Serial.println("TensorFlow Lite Micro ready.");

  // --- Camera configuration ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;   // smaller to save RAM
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (config.pixel_format == PIXFORMAT_JPEG && psramFound()) {
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);

#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  // --- WiFi connection ---
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");

  // --- Start camera server ---
  startCameraServer();
  Serial.print("Camera Ready! Connect to: http://");
  Serial.println(WiFi.localIP());
}

// ===========================
// JPEG decode helper (TJpgDec output callback)
// ===========================
static uint16_t* rgbBuffer;
static int rgbIndex;
bool tjpgCallback(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t* bitmap) {
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      uint16_t pixel = bitmap[j * w + i];
      uint8_t r = (pixel >> 11) & 0x1F;
      uint8_t g = (pixel >> 5) & 0x3F;
      uint8_t b = pixel & 0x1F;
      r = (r * 255) / 31;
      g = (g * 255) / 63;
      b = (b * 255) / 31;

      rgbBuffer[rgbIndex++] = (r << 16) | (g << 8) | b;
    }
  }
  return true;
}

// ===========================
// DERMAX inference
// ===========================
void runDermaxInference(camera_fb_t* fb) {
  // --- Decode JPEG into RGB buffer ---
  rgbIndex = 0;
  TJpgDec.setJpgScale(1);  // full size
  TJpgDec.setCallback(tjpgCallback);
  rgbBuffer = (uint16_t*)malloc(fb->width * fb->height * sizeof(uint16_t));
  TJpgDec.drawJpg(0, 0, fb->buf, fb->len);

  // --- Resize + normalize into input tensor ---
  float* input = interpreter->input(0)->data.f;
  int idx = 0;
  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      int srcX = x * fb->width / INPUT_W;
      int srcY = y * fb->height / INPUT_H;
      uint32_t pixel = rgbBuffer[srcY * fb->width + srcX];
      uint8_t r = (pixel >> 16) & 0xFF;
      uint8_t g = (pixel >> 8) & 0xFF;
      uint8_t b = pixel & 0xFF;
      input[idx++] = r / 255.0f;
      input[idx++] = g / 255.0f;
      input[idx++] = b / 255.0f;
    }
  }
  free(rgbBuffer);

  // --- Run inference ---
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  // --- Read output ---
  float* output = interpreter->output(0)->data.f;
  int predicted_class = 0;
  float max_val = output[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output[i] > max_val) {
      max_val = output[i];
      predicted_class = i;
    }
  }
  Serial.print("Predicted class: ");
  Serial.print(predicted_class);
  Serial.print(" (confidence: ");
  Serial.print(max_val, 4);
  Serial.println(")");
}

// ===========================
// Capture frame and run inference
// ===========================
void processCameraFrame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  runDermaxInference(fb);
  esp_camera_fb_return(fb);
}

// ===========================
// Main loop
// ===========================
void loop() {
  processCameraFrame();
  delay(2000);
}
