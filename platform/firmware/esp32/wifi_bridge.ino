/*
 * ESP32 Wi-Fi bridge for the RHS streaming platform.
 *
 * Wiring: ESP32 GPIO16 (RX2) <- STM32 TX, GPIO17 (TX2) -> STM32 RX, GND common.
 * Frames are forwarded verbatim; the gateway decodes them on the host.
 *
 * Library: PubSubClient (Arduino Library Manager).
 */

#include <WiFi.h>
#include <PubSubClient.h>

static const char *WIFI_SSID = "CHANGE_ME";
static const char *WIFI_PASS = "CHANGE_ME";
static const char *MQTT_HOST = "192.168.1.10";
static const int MQTT_PORT = 1883;
static const char *TOPIC_IN = "rhs/rx";
static const char *TOPIC_OUT = "rhs/tx";

WiFiClient net;
PubSubClient mqtt(net);
HardwareSerial STM32(2);

void onMqtt(char *topic, byte *payload, unsigned int length) {
  (void)topic;
  STM32.write(payload, length);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String clientId = "rhs-esp32-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    if (!mqtt.connect(clientId.c_str())) {
      delay(1000);
      continue;
    }
    mqtt.subscribe(TOPIC_IN);
  }
}

void setup() {
  Serial.begin(115200);
  STM32.begin(115200, SERIAL_8N1, 16, 17);
  connectWifi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqtt);
  mqtt.setBufferSize(2048);
  connectMqtt();
}

void loop() {
  if (!mqtt.connected()) {
    connectMqtt();
  }
  mqtt.loop();

  uint8_t buffer[512];
  size_t available = STM32.available();
  if (available > 0) {
    size_t count = STM32.readBytes(buffer, min(available, sizeof(buffer)));
    mqtt.publish(TOPIC_OUT, buffer, count);
  }
}
