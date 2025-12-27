/*
 * ESP32 MQTT to UART Bridge
 * Nhận lệnh từ MQTT và forward sang Arduino Nano qua UART
 */

#include <WiFi.h>
#include <PubSubClient.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT config
const char* mqtt_server = "mqtt.fuvitech.vn";
const int mqtt_port = 2883;
const char* mqtt_topic = "DATH/DKXE";

// UART config (pins 16, 17)
#define RXD2 16
#define TXD2 17

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);
  
  // Map MQTT commands to Nano commands
  int command = 0;
  if (message == "FORWARD") {
    command = 1;
  } else if (message == "BACKWARD") {
    command = 2;
  } else if (message == "TURN_LEFT_SLOW") {
    command = 3;  // Quay trái chậm (arc)
  } else if (message == "TURN_RIGHT_SLOW") {
    command = 4;  // Quay phải chậm (arc)
  } else if (message == "TURN_LEFT_FAST") {
    command = 5;  // Quay trái nhanh (spin)
  } else if (message == "TURN_RIGHT_FAST") {
    command = 6;  // Quay phải nhanh (spin)
  } else if (message == "STOP") {
    command = 0;
  }
  
  // Gửi command qua UART cho Nano
  Serial2.print(command);
  Serial2.print('\n');
  
  Serial.print("Sent to Nano: ");
  Serial.println(command);
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(mqtt_topic);
      Serial.print("Subscribed to: ");
      Serial.println(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  // Serial monitor
  Serial.begin(115200);
  
  // UART to Arduino Nano (pins 16, 17)
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);
  
  Serial.println("ESP32 MQTT to UART Bridge");
  Serial.println("Starting...");
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  
  Serial.println("Setup complete!");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
