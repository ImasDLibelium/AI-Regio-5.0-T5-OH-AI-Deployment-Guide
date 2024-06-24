#include <WiFiManager.h> // https://github.com/tzapu/WiFiManager
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Inicializa el cliente WiFi, servidor web y cliente MQTT
WiFiManager wifiManager;
WiFiClient espClient;
AsyncWebServer server(8080);
PubSubClient client(espClient);

// Estructura para los parámetros
struct Parameter {
  String type;
  String field;
};

// Variables de configuración dinámica
String mqtt_server;
int mqtt_port;
String mqtt_input_topic;
String mqtt_output_topic;
std::vector<Parameter> inputParameters;
std::vector<Parameter> outputParameters;

bool isConfigured = false;  // Variable para verificar si la configuración ha sido recibida

// TensorFlow Lite variables
tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
constexpr int tensor_arena_size = 16 * 1024;
uint8_t tensor_arena[tensor_arena_size];
bool modelLoaded = false;
std::vector<uint8_t> model_buffer;

void setup() {
  Serial.begin(115200);

  // Start WiFiManager for captive portal, with a timeout of 300 seconds (5 minutes)
  wifiManager.setConfigPortalTimeout(300); // Set timeout to 300 seconds (5 minutes)

  // Use autoConnect with a unique AP name and password to avoid conflicts
  if (!wifiManager.autoConnect("AI_REDGIO", "AIREDGIO.2024")) {
    Serial.println("Failed to connect and hit timeout");
    // Optionally, reset the ESP32 if timeout occurs
    ESP.restart();
  }

  Serial.println("Connected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  wifiManager.stopConfigPortal();

  // Configurar servidor web
  server.on("/uploadModel", HTTP_POST, [](AsyncWebServerRequest *request){
    request->send(200, "text/plain", "Upload started");
  }, handleFileUpload);

  server.onRequestBody([](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
    if (request->contentType() == "application/json") {
      String body = String((char*)data).substring(0, len);
      Serial.println("Received configuration:");
      Serial.println(body);  // Imprimir el cuerpo recibido para depuración
      if (configure_from_json(body)) {
        request->send(200, "application/json", "{\"status\":\"ok\"}");
        isConfigured = true;  // Indicar que la configuración ha sido recibida
      } else {
        request->send(400, "application/json", "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
      }
    } else {
      request->send(400, "application/json", "{\"status\":\"error\",\"message\":\"Content-Type must be application/json\"}");
    }
  });

  server.begin();

  // Configurar cliente MQTT
  client.setCallback(callback);
}

void loop() {
  if (!client.connected() && isConfigured) {
    reconnect();
  }
  client.loop();
}

void handleFileUpload(AsyncWebServerRequest *request, const String& filename, size_t index, uint8_t *data, size_t len, bool final) {
  if (!index) {
    Serial.printf("Upload Start: %s\n", filename.c_str());
    model_buffer.clear();
  }
  model_buffer.insert(model_buffer.end(), data, data + len);
  if (final) {
    Serial.printf("Upload End: %s, %u B\n", filename.c_str(), index + len);
    model = tflite::GetModel(model_buffer.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model schema version does not match TFLite version");
      request->send(500, "text/plain", "Model schema version does not match TFLite version");
      return;
    }
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    static tflite::AllOpsResolver local_resolver;
    interpreter = new tflite::MicroInterpreter(model, local_resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      Serial.println("Failed to allocate tensors!");
      request->send(500, "text/plain", "Failed to allocate tensors!");
      return;
    }
    modelLoaded = true;
    Serial.println("Model loaded and interpreter initialized!");
    request->send(200, "text/plain", "Model uploaded successfully.");
  }
}

bool configure_from_json(const String& jsonString) {
  StaticJsonDocument<1024> doc;
  DeserializationError error = deserializeJson(doc, jsonString);

  if (error) {
    Serial.println("Error al deserializar JSON de configuración");
    return false;
  }

  // Configurar broker y tópicos
  String broker = doc["broker"];
  int brokerDelimiterIndex = broker.indexOf("://");
  String serverInfo = broker.substring(brokerDelimiterIndex + 3);
  int portDelimiterIndex = serverInfo.indexOf(":");
  mqtt_server = serverInfo.substring(0, portDelimiterIndex);
  mqtt_port = serverInfo.substring(portDelimiterIndex + 1).toInt();
  mqtt_input_topic = doc["topic-recv"].as<String>();
  mqtt_output_topic = doc["topic-pub"].as<String>();

  // Imprimir configuración recibida
  Serial.println("Configuración recibida:");
  Serial.print("mqtt_server: ");
  Serial.println(mqtt_server);
  Serial.print("mqtt_port: ");
  Serial.println(mqtt_port);
  Serial.print("mqtt_input_topic: ");
  Serial.println(mqtt_input_topic);
  Serial.print("mqtt_output_topic: ");
  Serial.println(mqtt_output_topic);

  // Configurar parámetros de entrada
  inputParameters.clear();
  JsonArray inputArray = doc["inputParameters"];
  for (JsonObject param : inputArray) {
    Parameter p;
    p.type = param["type"].as<String>();
    p.field = param["field"].as<String>();
    inputParameters.push_back(p);
    Serial.print("Input Parameter - Type: ");
    Serial.print(p.type);
    Serial.print(", Field: ");
    Serial.println(p.field);
  }

  // Configurar parámetros de salida
  outputParameters.clear();
  JsonArray outputArray = doc["outputParameters"];
  for (JsonObject param : outputArray) {
    Parameter p;
    p.type = param["type"].as<String>();
    p.field = param["field"].as<String>();
    outputParameters.push_back(p);
    Serial.print("Output Parameter - Type: ");
    Serial.print(p.type);
    Serial.print(", Field: ");
    Serial.println(p.field);
  }

  // Configurar y conectar al broker MQTT
  client.setServer(mqtt_server.c_str(), mqtt_port);
  reconnect();

  return true;
}

void callback(char* topic, byte* payload, unsigned int length) {
  if (!modelLoaded) {
    Serial.println("Model not loaded.");
    return;
  }

  StaticJsonDocument<512> doc;
  deserializeJson(doc, payload, length);

  // Procesar parámetros de entrada
  TfLiteTensor* input_tensor = interpreter->input(0);
  int param_num = 0;
  for (Parameter p : inputParameters) {
    if (p.type == "f") {
      float value = doc[p.field].as<float>();
      input_tensor->data.f[param_num] = value;
      Serial.print("Received float: ");
      Serial.print(p.field);
      Serial.print(" = ");
      Serial.println(value);
    } else if (p.type == "i32") {
      int32_t value = doc[p.field].as<int32_t>();
      input_tensor->data.i32[param_num] = value;
      Serial.print("Received int32: ");
      Serial.print(p.field);
      Serial.print(" = ");
      Serial.println(value);
    } else {
      Serial.println("Tipo no soportado");
    }
    param_num++;
  }

  // Invocar el modelo para predecir
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke the model");
    return;
  }

  // Preparar mensaje de salida
  StaticJsonDocument<200> mensaje_salida;
  param_num = 0;
  for (Parameter p : outputParameters) {
    if (p.type == "f") {
      mensaje_salida[p.field] = interpreter->output(0)->data.f[param_num];
    } else if (p.type == "i32") {
      mensaje_salida[p.field] = interpreter->output(0)->data.i32[param_num];
    } else {
      Serial.println("Tipo no soportado");
    }
    param_num++;
  }

  // Serializar y publicar el mensaje de salida
  char outputBuffer[512];
  serializeJson(mensaje_salida, outputBuffer);
  client.publish(mqtt_output_topic.c_str(), outputBuffer);

  // Imprimir mensaje de salida en el Serial
  Serial.println("Published MQTT message:");
  serializeJson(mensaje_salida, Serial);
}

void reconnect() {
  if (!isConfigured) {
    return;  // No intentar conectar si no se ha recibido la configuración
  }

  // Bucle hasta que se realice la conexión
  while (!client.connected()) {
    Serial.print("Intentando conexión MQTT a ");
    Serial.print(mqtt_server);
    Serial.print(":");
    Serial.print(mqtt_port);
    Serial.println("...");

    // Intentamos conectar
    if (client.connect("AIREDGIO")) {
      Serial.println("Conectado");
      client.subscribe(mqtt_input_topic.c_str());
      Serial.print("Suscrito al tópico: ");
      Serial.println(mqtt_input_topic);
    } else {
      Serial.print("falló, rc=");
      Serial.print(client.state());
      Serial.println(" intentando de nuevo en 5 segundos");
      delay(5000);
    }
  }
}