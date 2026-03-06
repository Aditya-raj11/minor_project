/*
  Smart Attendance System - Arduino LED Controller
  ------------------------------------------------
  Listens over USB Serial for commands from the Python Facial Recognition Engine.
  
  COM PROTOCOL:
  '1' = Face Recognized (Green LED pulses for 2 seconds)
  '0' = Unknown Face (Red LED pulses for 2 seconds)
*/

const int GREEN_PIN = 8;
const int RED_PIN = 9;

int currentLed = -1;
unsigned long ledTurnedOnAt = 0;
const unsigned long LED_DURATION = 2000; // 2 seconds

void setup() {
  Serial.begin(9600);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  
  // Flash both to indicate bootup ready
  digitalWrite(GREEN_PIN, HIGH);
  digitalWrite(RED_PIN, HIGH);
  delay(500);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(RED_PIN, LOW);
  
  // Ready light pulse
  Serial.println("ARDUINO_READY");
}

void loop() {
  // Check if Python sent a command
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == '1') {
      // Known Face -> Green
      digitalWrite(GREEN_PIN, HIGH);
      digitalWrite(RED_PIN, LOW);
      currentLed = GREEN_PIN;
      ledTurnedOnAt = millis();
    } 
    else if (command == '0') {
      // Unknown Face -> Red
      digitalWrite(RED_PIN, HIGH);
      digitalWrite(GREEN_PIN, LOW);
      currentLed = RED_PIN;
      ledTurnedOnAt = millis();
    }
  }

  // Turn off LED after duration expires
  if (currentLed != -1 && (millis() - ledTurnedOnAt >= LED_DURATION)) {
    digitalWrite(currentLed, LOW);
    currentLed = -1;
  }
}
