#include <Arduino.h>

#define PIR_PIN 13
#define BUZZER_PIN 26
#define LIGHT_PIN 27

const bool LIGHT_RELAY_ACTIVE_LOW = false;
const unsigned long PRESENCE_TIMEOUT_MS = 60000UL;
const bool AUTO_OFF_WHEN_EMPTY = false;

bool present = false;
bool lightState = false;
bool buzzerState = false;
unsigned long lastMotionTime = 0;

void writeLight(bool state)
{
    if (LIGHT_RELAY_ACTIVE_LOW)
    {
        digitalWrite(LIGHT_PIN, state ? LOW : HIGH);
    }
    else
    {
        digitalWrite(LIGHT_PIN, state ? HIGH : LOW);
    }
}

void setLight(bool state)
{
    lightState = state;
    writeLight(state);
}

void setBuzzer(bool state)
{
    buzzerState = state;
    digitalWrite(BUZZER_PIN, state ? HIGH : LOW);
}

void activatePresence()
{
    if (!present)
    {
        present = true;
        Serial.println("[PRESENCE] ACTIVE");
    }

    lastMotionTime = millis();
}

void deactivatePresence()
{
    if (!present)
    {
        return;
    }

    present = false;
    Serial.println("[PRESENCE] INACTIVE");

    if (AUTO_OFF_WHEN_EMPTY)
    {
        setLight(false);
        setBuzzer(false);
        Serial.println("[AUTO] LIGHT OFF");
        Serial.println("[AUTO] BUZZER OFF");
    }
}

void updatePresence()
{
    unsigned long now = millis();

    if (digitalRead(PIR_PIN) == HIGH)
    {
        activatePresence();
    }

    if (present && now - lastMotionTime >= PRESENCE_TIMEOUT_MS)
    {
        deactivatePresence();
    }
}

void printStatus()
{
    Serial.println("[STATE]");
    Serial.print("[STATE] room=");
    Serial.println(present ? "OCCUPIED" : "EMPTY");
    Serial.print("[STATE] LIGHT=");
    Serial.println(lightState ? "ON" : "OFF");
    Serial.print("[STATE] BUZZER=");
    Serial.println(buzzerState ? "ON" : "OFF");
}

void executeCommand(char command)
{
    switch (command)
    {
        case '1':
            if (!present)
            {
                Serial.println("[CMD] IGNORED - LIGHT ON - ROOM EMPTY");
                return;
            }
            setLight(true);
            Serial.println("[CMD] OK - LIGHT ON");
            return;

        case '2':
            if (!present)
            {
                Serial.println("[CMD] IGNORED - LIGHT OFF - ROOM EMPTY");
                return;
            }
            setLight(false);
            Serial.println("[CMD] OK - LIGHT OFF");
            return;

        case '3':
            if (!present)
            {
                Serial.println("[CMD] IGNORED - BUZZER ON - ROOM EMPTY");
                return;
            }
            setBuzzer(true);
            Serial.println("[CMD] OK - BUZZER ON");
            return;

        case '4':
            if (!present)
            {
                Serial.println("[CMD] IGNORED - BUZZER OFF - ROOM EMPTY");
                return;
            }
            setBuzzer(false);
            Serial.println("[CMD] OK - BUZZER OFF");
            return;

        case '?':
            printStatus();
            return;

        case '\n':
        case '\r':
        case ' ':
            return;

        default:
            Serial.print("[CMD] UNKNOWN: ");
            Serial.println(command);
            return;
    }
}

void readSerialCommands()
{
    while (Serial.available() > 0)
    {
        executeCommand((char)Serial.read());
    }
}

void setup()
{
    Serial.begin(115200);

    pinMode(PIR_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LIGHT_PIN, OUTPUT);

    setLight(false);
    setBuzzer(false);

    Serial.println();
    Serial.println("================================");
    Serial.println("      ESP32 VOICE CONTROL");
    Serial.println("================================");
    Serial.println("[SYSTEM] PIR enabled on GPIO 13");
    Serial.println("[SYSTEM] Buzzer on GPIO 26");
    Serial.println("[SYSTEM] Light on GPIO 27");
    Serial.println("[SYSTEM] Presence timeout: 60 seconds");
    Serial.println("[COMMANDS] 1=LIGHT ON, 2=LIGHT OFF");
    Serial.println("[COMMANDS] 3=BUZZER ON, 4=BUZZER OFF");
    Serial.println("[COMMANDS] ?=STATUS");
    Serial.println("[SYSTEM] Waiting for presence...");
    printStatus();
}

void loop()
{
    updatePresence();
    readSerialCommands();
    delay(5);
}