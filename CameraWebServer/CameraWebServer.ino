#include "esp_camera.h"
#include <WiFi.h>
#include <TensorFlowLite_ESP32.h>
#include "dermax_model.h"   // must define `const unsigned char dermax_model[]` and `int dermax_model_len`
#include "board_config.h"   // your ESP32-CAM pin config

// Arena for TFLM
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TFLM globals
tflite::MicroInterpreter* interpreter;
tflite::ErrorReporter* error_reporter;
tflite::MicroMutableOpResolver<10> resolver;  // reserve space for 10 ops
const tflite::Model* model;
TfLiteTensor* input;
TfLiteTensor* output;

// ==================== WiFi ====================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // -------- Load Model --------
  model = tflite::GetModel(dermax_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema version mismatch!");
    while (1);
  }

  // -------- Add Required Ops --------
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

  // -------- Error Reporter --------
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // -------- Interpreter --------
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ TensorFlow Lite Micro ready.");

  // -------- Camera Config --------
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
  config.frame_size = FRAMESIZE_QVGA;   // 320x240
  config.pixel_format = PIXFORMAT_RGB565; // easier for TFLM
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Camera init failed");
    while (1);
  }

  Serial.println("✅ Camera ready.");
}

// ==================== Run Inference ====================
void runInference(camera_fb_t* fb) {
  if (!fb) {
    Serial.println("❌ Frame capture failed");
    return;
  }

  // Resize + normalize to fit input tensor
  int input_w = input->dims->data[1];
  int input_h = input->dims->data[2];
  int input_c = input->dims->data[3];

  uint8_t* in = fb->buf;
  float* input_data = input->data.f;

  int idx = 0;
  for (int y = 0; y < input_h; y++) {
    for (int x = 0; x < input_w; x++) {
      int src_x = x * fb->width / input_w;
      int src_y = y * fb->height / input_h;
      int src_index = (src_y * fb->width + src_x) * 3;  // assuming RGB888

      input_data[idx++] = in[src_index + 0] / 255.0f;  // R
      input_data[idx++] = in[src_index + 1] / 255.0f;  // G
      input_data[idx++] = in[src_index + 2] / 255.0f;  // B
    }
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed");
    return;
  }

  // Read results
  int num_classes = output->dims->data[1];
  float max_score = -1;
  int predicted = -1;

  for (int i = 0; i < num_classes; i++) {
    float score = output->data.f[i];
    Serial.printf("Class %d: %.4f\n", i, score);
    if (score > max_score) {
      max_score = score;
      predicted = i;
    }
  }

  Serial.printf("✅ Predicted class: %d (%.4f)\n", predicted, max_score);
}

// ==================== Loop ====================
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  runInference(fb);
  esp_camera_fb_return(fb);
  delay(2000);
}
