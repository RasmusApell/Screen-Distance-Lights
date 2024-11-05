#include <Adafruit_NeoPixel.h>

#define PIN_NEOPIXEL 13 
#define NUMPIXELS 6      

Adafruit_NeoPixel pixels(NUMPIXELS, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);

unsigned long lastSerialData = 0;
const unsigned long TIMEOUT = 1000; //(ms)
String inputBuffer = "";

// Distance thresholds in (m)
const float DISTANCE_NEAR = 0.5;      
const float DISTANCE_MEDIUM = 0.75;  
const float DISTANCE_FAR = 1;    

void setup() {
  Serial.begin(9600);
  
  pixels.begin();
  pixels.setBrightness(50); 
  pixels.show();
}

void loop() {
  bool dataRecieved = false;
  while (Serial.available() > 0) {
    lastSerialData = millis();
    char c = Serial.read();
    if (c == '\n') {
      float distance = inputBuffer.toFloat();
      updateLights(distance);
      inputBuffer = "";
      dataRecieved = true;
    } else {
      inputBuffer += c;
    }
  }
  if (!dataRecieved && (millis() - lastSerialData) > TIMEOUT) {
    pixels.clear();
    pixels.setPixelColor(0, pixels.Color(255, 0, 0));
    pixels.show();
  }
}

void updateLights(float distance) {
  uint32_t color;
  int numLit;
  
  if (distance < 0) { // No face detected
    float t = millis() / 1000.0;
    int brightness = (sin(t * 3.14159 * 2) + 1) * 127.5;
    color = pixels.Color(0, 0, brightness);
    numLit = 1;
  }
  else if (distance <= DISTANCE_NEAR) {
    // Red 
    color = pixels.Color(255, 0, 0);
    numLit = 2;
  }
  else if (distance <= DISTANCE_MEDIUM) {
    // Orange 
    color = pixels.Color(255, 69, 0);
    numLit = 3;
  }
  else if (distance <= DISTANCE_FAR) {
    // Yellow 
    color = pixels.Color(255, 255, 0);
    numLit = 4;
  }
  else {
    // Green
    color = pixels.Color(0, 255, 0);
    numLit = 5;
  }

  pixels.clear();
  pixels.setPixelColor(0, pixels.Color(0, 0, 255));
  pixels.setPixelColor(numLit, color);
  pixels.show();
}