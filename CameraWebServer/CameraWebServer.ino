#include "esp_camera.h"
#include <WiFi.h>
#include <Arduino.h>
#include "Chirale_TensorFlowLite.h"
#include "dermax_model.h"
#include "board_config.h"

// ===========================
// WiFi credentials
// ===========================
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";

// ===========================
// TensorFlow Lite setup
// ===========================
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
MicroInterpreter* interpreter;  // <- removed Chirale::

// Input and output definitions
#define INPUT_SIZE 224*224*3  // 224x224 RGB input
#define NUM_CLASSES 7         // 7 types of skin cancer

// ===========================
// Function prototypes
// ===========================
void startCameraServer();
void setupLedFlash();
void runDermaxInference(camera_fb_t* fb);
void processCameraFrame();

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    // --- Initialize TFLite Micro interpreter ---
    interpreter = new MicroInterpreter(dermax_model, tensor_arena, kTensorArenaSize);
    if (!interpreter->allocateTensors()) {
        Serial.println("Failed to allocate tensors!");
    } else {
        Serial.println("TFLite Micro interpreter ready");
    }

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
    config.frame_size = FRAMESIZE_UXGA;
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
    s->set_framesize(s, FRAMESIZE_QVGA); // lower resolution for higher FPS

#if defined(LED_GPIO_NUM)
    setupLedFlash();
#endif

    // --- Connect to WiFi ---
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
// DERMAX inference
// ===========================
void runDermaxInference(camera_fb_t* fb) {
    // TODO: Convert JPEG to float array for model input
    float input_data[INPUT_SIZE] = {0};  // placeholder

    float* input = interpreter->input(0)->data.f;
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = input_data[i] / 255.0f;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

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
    Serial.println(predicted_class);
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
    delay(2000); // wait 2 seconds before next frame
}
