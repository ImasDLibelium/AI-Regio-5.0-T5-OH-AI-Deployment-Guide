# Project Documentation for Prediction on ESP32

## Introduction
This project demonstrates how to configure an ESP32 to load a TensorFlow Lite model and use it to perform predictions based on data received via MQTT. Configuration and input data are sent to the ESP32 using HTTP and MQTT, respectively.

*Everything is embedded in the ESP32, thus behaving like an Edge Node.*

**Technologies Used**
- ESP32: A low-cost microcontroller with WiFi and Bluetooth connectivity.
- TensorFlow Lite: A machine learning library for resource-constrained devices.
- ArduinoJson: A library for handling JSON data on Arduino devices.
- PubSubClient: A library for handling MQTT connectivity.
- ESPAsyncWebServer: A library for handling asynchronous web servers on the ESP32.

**System Requirements**

**Hardware:**
- ESP32
- WiFi connection

**Software:**
- Arduino IDE
- Arduino Libraries:
	- WiFiManager
	- ESPAsyncWebServer
	- AsyncTCP
	- TensorFlowLite_ESP32
	- ArduinoJson
	- PubSubClient

**Library Installation**
- To install the necessary libraries, follow these steps:
*TensorFlowLite_ESP32, ArduinoJson ESPAsyncWebServer, AsyncTCP and PubSubClient:*
- Open the Arduino IDE.
- Go to Sketch -> Include Library -> Manage Libraries.
- Search for and install the mentioned libraries.

**Flexible WiFi Configuration:**
- This code allows you to configure the ESP32's WiFi in a flexible manner using a captive portal provided by WiFiManager. The first time it is started, the ESP32 creates an access point (AP) that you can connect to in order to configure the desired WiFi network.

	- AP Name: “AI_REDGIO” - Temporary WiFi network name for configuration.
	- AP Password: “AIREDGIO.2024” - Password to connect to this network.
	- Web Portal: Connect to 192.168.4.1 to access the wifi configuration web portal.

## IMPORTANT DETAIL
*The Wifi and IP of the device must match the computer or client from which the HTTP request is sent.*

**Model Upload API Requirements**
To upload a TensorFlow Lite model to the ESP32, use an HTTP POST request. Here are the details:

- Endpoint: /uploadModel
- HTTP Method: POST
- Port: 8080
- Headers:
	- Content-Type: multipart/form-data
-Body:
	- Attach the TensorFlow Lite model file.
	
Example of using cURL to upload the model:

	curl -X POST http://<ESP32_IP>:8080/uploadModel -F "model=@path/to/your/model.tflite"  or use applications such as Thunder Client		(VSCode) or Postman.
 
**Configuration API Requirements**
To configure the ESP32 with the MQTT broker details and input/output parameters, send an HTTP POST request with the following JSON format:

- Endpoint: /configure
- HTTP Method: POST
- Port: 8080
- Headers:
	- Content-Type: application/json
- Body:

	{
	  "broker": "YOUR_BROKER",
	  "topic-recv": "YOUR_TOPIC_RECV",
	  "inputParameters": [
	    {
	      "type": "TYPE1",
	      "field": "FIELD1"
	    },
	    {
	      "type": "TYPE2",
	      "field": "FIELD2"
	    }
	    // Add more parameters as needed
	  ],
	  "topic-pub": "YOUR_TOPIC_PUB",
	  "outputParameters": [
	    {
	      "type": "TYPE1",
	      "field": "FIELD1"
	    },
	    {
	      "type": "TYPE2",
	      "field": "FIELD2"
	    }
	    // Add more parameters as needed
	  ],
	  "iterations": "NUMBER_OF_ITERATIONS"
	}
	
Example of using cURL to send the configuration:

	curl -X POST http://<ESP32_IP>:8080/configure -H "Content-Type: application/json" -d @config.json

**MQTT Input Data Requirements**
To send input data to the ESP32 via MQTT, follow these details:

- Broker: Specified in the configuration.
- Receive Topic: Specified in the configuration (topic-recv).
- Data Format:

	{
	  "field1": "value1",
	  "field2": "value2"
	  // Add more fields as needed
	}

## Code Details
**WiFi Connection:**
- The ESP32 connects to the WiFi network using the provided credentials.

**Web Server Configuration:**
- The web server handles HTTP requests for uploading the model and receiving the JSON configuration.

**TensorFlow Lite Model Upload:**
- The model is uploaded via the /uploadModel endpoint.
- The model is stored in a buffer and initialized for use in predictions.

**MQTT Configuration:**
- Configuration is received via the /configure endpoint.
- The MQTT broker details, input and output topics, and input and output parameters are configured.

**MQTT Broker Connection:**
- The ESP32 connects to the MQTT broker using the provided details.
- Subscribes to the specified input topic.

**MQTT Message Handling:**
- Incoming messages on the input topic are processed to extract input parameters.
- The model is invoked to perform the prediction.
- Prediction results are published to the output topic.


## Key Points and Limitations
**Maximum Memory:**
- The ESP32 has a memory limit. The available memory for models depends on the model's complexity and other libraries' usage. Typically, models should not exceed 300KB.

**Error Handling:**
- The code handles errors such as model loading failure, invalid configuration, and memory issues.

**Scalability:**
- The system is designed to be flexible and scalable, allowing different models and parameters to be configured via JSON.
- This project provides a solid foundation for using the ESP32 as a prediction device based on machine learning, suitable for applications in the Internet of Things (IoT).

