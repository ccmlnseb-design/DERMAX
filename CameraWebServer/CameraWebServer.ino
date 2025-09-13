#include "esp_camera.h"
#include <WiFi.h>
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>   // ✅ must be first for ESP32 build
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Your headers
#include "dermax_model.h"   // exported model as byte array
#include "board_config.h"   // ESP32-CAM pin mapping

// ===========================
// WiFi setup
// ===========================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===========================
// TensorFlow Lite globals
// ===========================
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 60 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}

// Model input/output
#define INPUT_W 224
#define INPUT_H 224
#define INPUT_C 3
#define INPUT_SIZE (INPUT_W * INPUT_H * INPUT_C)
#define NUM_CLASSES 7

// ===========================
// Camera setup
// ===========================
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size   = FRAMESIZE_QVGA;   // 320x240
  config.pixel_format = PIXFORMAT_RGB565; // easier than JPEG for inference
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Camera init failed!");
    while (true);
  }

  Serial.println("✅ Camera ready.");
}

// ===========================
// Run inference on camera frame
// ===========================
void runInference(camera_fb_t* fb) {
  if (!fb) {
    Serial.println("❌ Frame capture failed!");
    return;
  }

  // Pointer to model input
  float* input_data = input->data.f;

  // Resize + normalize: take center crop from 320x240 → 224x224
  int x_offset = (fb->width - INPUT_W) / 2;
  int y_offset = (fb->height - INPUT_H) / 2;

  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      int src_x = x + x_offset;
      int src_y = y + y_offset;

      uint16_t pixel = ((uint16_t*)fb->buf)[src_y * fb->width + src_x];
      uint8_t r = (pixel >> 11) & 0x1F;
      uint8_t g = (pixel >> 5) & 0x3F;
      uint8_t b = pixel & 0x1F;
      r = (r * 255) / 31;
      g = (g * 255) / 63;
      b = (b * 255) / 31;

      *input_data++ = r / 255.0f;
      *input_data++ = g / 255.0f;
      *input_data++ = b / 255.0f;
    }
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    return;
  }

  // Process output
  output = interpreter->output(0);
  int predicted_class = 0;
  float max_val = output->data.f[0];

  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > max_val) {
      max_val = output->data.f[i];
      predicted_class = i;
    }
  }

  Serial.printf("✅ Prediction: class %d (confidence %.4f)\n",
                predicted_class, max_val);
}

// ===========================
// Arduino setup
// ===========================
void setup() {
  Serial.begin(115200);

  // WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected");

  // Camera
  setupCamera();

  // TensorFlow Lite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(dermax_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema mismatch!");
    while (true);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed!");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ TensorFlow Lite Micro ready.");
}

// ===========================
// Arduino loop
// ===========================
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) {
    runInference(fb);
    esp_camera_fb_return(fb);
  }
  delay(2000);
}
