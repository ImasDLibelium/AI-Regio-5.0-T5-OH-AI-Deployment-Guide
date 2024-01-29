
// From https://randomnerdtutorials.com/esp32-mqtt-publish-subscribe-arduino-ide

#include <WiFi.h>
#include "PubSubClient.h"
#include <ArduinoJson.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "o3_predictor.h"

// WIFI CONF
//const char* ssid = "Wokwi-GUEST"; //uncomment and edit to your wifi
//const char* password = "";
WiFiClient espClient;

// MQTT CONF
const char* mqttServer = "broker.emqx.io";
const char* mqttTopic = "topicName/aq";
int port = 1883;
String stMac;
char mac[50];
char clientId[50];
PubSubClient client(espClient);

// TENSORFLOW LITE
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

StaticJsonDocument<200> doc;

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(0));

  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  wifiConnect();

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println(WiFi.macAddress());
  stMac = WiFi.macAddress();
  stMac.replace(":", "_");
  Serial.println(stMac);

  client.setServer(mqttServer, port);
  client.setCallback(callback);
  //pinMode(ledPin, OUTPUT);
  //TFLITE
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(O3_predictor_tflite);
  Serial.println("Model Loaded");
  Serial.print("TFLITE SCHEMA VERSION ");
  Serial.println(TFLITE_SCHEMA_VERSION);
  Serial.print("MODEL VERSION ");
  Serial.println(model->version());
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  Serial.println("Creating Op Resolver");
  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<1> resolver;

  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  Serial.println("Creating Interpreter");
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  Serial.println("Interpreter created");
  Serial.println("Allocating Tensors");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }
  Serial.println("Tensors initialized");

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void wifiConnect() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
}

void mqttReconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    long r = random(1000);
    sprintf(clientId, "clientId-%ld", r);
    if (client.connect(clientId)) {
      Serial.print(clientId);
      Serial.println(" connected");
      client.subscribe(mqttTopic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void callback(char* topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  deserializeJson(doc, (const byte*)message, length);
  String stMessage;
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    stMessage += (char)message[i];
  }
  Serial.println();
  if (String(topic) == mqttTopic) {
    // message received to this topic
    float values[] = {doc["PM10"], doc["PM2.5"], doc["NO2"], doc["SO2"], doc["O3"], doc["CO"]};
    float pm10 = doc["PM10"];
    float pm25 = doc["PM2.5"];
    float no2 = doc["NO2"];
    float so2 = doc["SO2"];
    float o3 = doc["O3"];
    float co = doc["CO"];

    // Place our calculated value in the model's input tensor
    Serial.println("Setting input data");

    //input->data.f[0] = values[0];
    input->data.f[0] = pm10;
    Serial.printf("PM10:%.2f, ", pm10);
    input->data.f[1] = values[1];
    Serial.printf("PM2.5:%.2f, ", values[1]);
    input->data.f[2] = values[2];
    Serial.printf("NO2:%.2f, ", values[2]);
    input->data.f[3] = values[3];
    Serial.printf("SO2:%.2f, ", values[3]);
    input->data.f[4] = values[4];
    Serial.printf("O3:%.2f, ", values[4]);
    input->data.f[5] = values[5];
    Serial.printf("CO:%.2f", values[5]);
    Serial.println();

    // Run inference, and report any error
    Serial.println("Invoking Tensorflow Lite");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on values: %f\n",
                             static_cast<float*>(values));

      return;
    }
    
    Serial.print("Raw output type:");
    Serial.println(output->type);

    //Serial.print("Data output as floats: ");
    //for (int i = 0; i < 6; i++){
    //  Serial.print(output->data.f[i]);
    //  Serial.println();
    //}
    
    Serial.print("Data output as float: ");
    Serial.println(output->data.f[0]);
    char buffer[10];
    sprintf(buffer, "{'o3':%.2f}",output->data.f[0]);
    Serial.println(buffer);
    Serial.println("[APP] Free memory: " + String(esp_get_free_heap_size()) + " bytes");
    
    //client.publish(topic, (char*) buffer);  // Not working?
    
    // Output the results. A custom HandleOutput function can be implemented
    // for each supported hardware target.
    //HandleOutput(error_reporter, x_val, y_val);

    // Increment the inference_counter, and reset it if we have reached
    // the total number per cycle
    inference_count += 1;
    //if (inference_count >= kInferencesPerCycle) inference_count = 0;
  }
}

void loop() {
  delay(10);
  if (!client.connected()) {
    mqttReconnect();
  }
  client.loop();
}
