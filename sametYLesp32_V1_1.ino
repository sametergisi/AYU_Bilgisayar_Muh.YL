#include <WiFi.h>
#include <DHT.h>

const char* WIFI_SSID = "SUPERONLINE_WiFi_93AF";
const char* WIFI_PASS = "CLPCVL77NAJ9";

const char* SERVER_IP = "ip here";
const uint16_t SERVER_PORT = 5000;

const uint8_t MIC_PIN = 32;
const uint8_t LDR_PIN = 35;
const uint8_t MQ4_PIN = 34;
const uint8_t DHT_PIN = 16;

const uint8_t DHT_TYPE = DHT11;
const uint16_t SEND_PERIOD_MS = 1000;
const uint8_t ADC_SAMPLE_COUNT = 8;

WiFiClient tcpClient;
DHT dht(DHT_PIN, DHT_TYPE);

uint32_t lastSendMs = 0;

uint16_t readAnalogAverage(const uint8_t pin)
{
    uint32_t sum = 0;

    for (uint8_t i = 0; i < ADC_SAMPLE_COUNT; i++) {
        sum += (uint32_t)analogRead(pin);
        delay(2);
    }

    return (uint16_t)(sum / ADC_SAMPLE_COUNT);
}

bool ensureWifiAndTcp()
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi baglaniyor...");
        WiFi.begin(WIFI_SSID, WIFI_PASS);

        uint32_t startMs = millis();
        while ((WiFi.status() != WL_CONNECTED) && ((millis() - startMs) < 10000UL)) {
            delay(500);
            Serial.print(".");
        }
        Serial.println();

        if (WiFi.status() == WL_CONNECTED) {
            Serial.print("WiFi OK. IP: ");
            Serial.println(WiFi.localIP());
        } else {
            Serial.println("WiFi baglanamadi.");
            return false;
        }
    }

    if (!tcpClient.connected()) {
        Serial.println("TCP baglaniyor...");
        if (!tcpClient.connect(SERVER_IP, SERVER_PORT)) {
            Serial.println("TCP baglanamadi.");
            return false;
        }
        Serial.println("TCP baglandi.");
    }

    return true;
}

void setup()
{
    Serial.begin(115200);
    delay(1000);

    dht.begin();

    analogReadResolution(12);
    analogSetPinAttenuation(MIC_PIN, ADC_11db);
    analogSetPinAttenuation(LDR_PIN, ADC_11db);
    analogSetPinAttenuation(MQ4_PIN, ADC_11db);

    WiFi.mode(WIFI_STA);

    Serial.println("ESP32 basladi.");
}

void loop()
{
    const uint32_t nowMs = millis();

    if ((nowMs - lastSendMs) < SEND_PERIOD_MS) {
        return;
    }

    lastSendMs = nowMs;

    const uint16_t micValue = readAnalogAverage(MIC_PIN);
    const uint16_t ldrValue = readAnalogAverage(LDR_PIN);
    const uint16_t mq4Value = readAnalogAverage(MQ4_PIN);

    const float hum = dht.readHumidity();
    const float temp = dht.readTemperature();

    if (isnan(hum) || isnan(temp)) {
        Serial.println("DHT11 okunamadi.");
        return;
    }

    const uint16_t hum10 = (uint16_t)(hum * 10.0f);
    const uint16_t temp10 = (uint16_t)(temp * 10.0f);

    char packet[96];
    snprintf(
        packet,
        sizeof(packet),
        "AE%u:%u:%u:%u:%uAYU",
        (unsigned)micValue,
        (unsigned)ldrValue,
        (unsigned)mq4Value,
        (unsigned)hum10,
        (unsigned)temp10
    );

    Serial.println("Gonderilecek paket:");
    Serial.println(packet);

    if (!ensureWifiAndTcp()) {
        return;
    }

    const size_t packetLen = strlen(packet);
    const size_t written = tcpClient.write((const uint8_t*)packet, packetLen);

    if (written == packetLen) {
        Serial.println("TCP gonderildi.");
    } else {
        Serial.println("TCP gonderme hatasi.");
        tcpClient.stop();
    }
}