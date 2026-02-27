const int latchPin = 10;  // Pin connected to the latch of the shift register
const int clockPin = 11;  // Pin connected to the clock of the shift register
const int dataPin = 12;   // Pin connected to the data of the shift register

// Define button pins
const int modeButtonPin = 9;
const int speedButtonPin = 8;

// Define debounce time
const unsigned long debounceDelay = 50;  // 50 milliseconds

// Variables to store the current and last state of each button
int modeState = LOW;
int speedState = LOW;
int lastModeState = LOW;
int lastSpeedState = LOW;

// Variables to store the last time the button state changed
unsigned long lastDebounceTime1 = 0;
unsigned long lastDebounceTime2 = 0;

// Define modeNumbers and speedNumbers arrays
int modeNumbers[3] = { B01100000, B11011010, B11110010 };
int speedNumbers[4] = { B01100000, B11011010, B11110010, B01100110 };

// Current display modes
int modeDisp = 0;   // Adjusted to zero index for array access
int speedDisp = 0;  // Adjusted to zero index for array access

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(9600);

  // Set up shift register pins
  pinMode(latchPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin, OUTPUT);

  // Set button pins as inputs with pull-up resistors
  pinMode(modeButtonPin, INPUT_PULLUP);
  pinMode(speedButtonPin, INPUT_PULLUP);
}

void loop() {
  // Read the current state of both buttons
  int modeButtonReading = digitalRead(modeButtonPin);
  int speedButtonReading = digitalRead(speedButtonPin);

  // Debounce button 1
  if (modeButtonReading != lastModeState) {
    lastDebounceTime1 = millis();  // Reset the debounce timer for button 1
  }
  if ((millis() - lastDebounceTime1) > debounceDelay) {
    // Update modeState if the reading has been stable
    if (modeButtonReading != modeState) {
      modeState = modeButtonReading;
      // Take action if button 1 is pressed
      if (modeState == LOW) {  // Button is active LOW
        modeDisp++;
        if (modeDisp > 2) {  // Ensure it stays within bounds for modeNumbers
          modeDisp = 0;      // Loop back to the first mode
        }
        Serial.print("modeDisp: ");
        Serial.println(modeDisp);
        updateShiftRegister();  // Update the shift register after button press
      }
    }
  }
  lastModeState = modeButtonReading;  // Update last button state

  // Debounce button 2
  if (speedButtonReading != lastSpeedState) {
    lastDebounceTime2 = millis();  // Reset the debounce timer for button 2
  }
  if ((millis() - lastDebounceTime2) > debounceDelay) {
    // Update speedState if the reading has been stable
    if (speedButtonReading != speedState) {
      speedState = speedButtonReading;
      // Take action if button 2 is pressed
      if (speedState == LOW) {  // Button is active LOW
        speedDisp++;
        if (speedDisp > 3) {  // Ensure it stays within bounds for speedNumbers
          speedDisp = 0;      // Loop back to the first speed
        }
        Serial.print("speedDisp: ");
        Serial.println(speedDisp);
        updateShiftRegister();  // Update the shift register after button press
      }
    }
  }
  lastSpeedState = speedButtonReading;  // Update last button state
}

void updateShiftRegister() {
  // Combine the current mode and speed values with mode first
  int combinedValue = (modeNumbers[modeDisp] << 8) | speedNumbers[speedDisp];

  // Set the latch pin to low to start shifting out
  digitalWrite(latchPin, LOW);

  // Shift out the bits to the shift register in LSBFIRST order
  for (int i = 0; i < 16; i++) {  // Send 16 bits, LSB first
    // Get the current bit (from LSB to MSB)
    int bit = (combinedValue >> i) & 1;

    // Send the bit to the data pin
    digitalWrite(dataPin, bit);
    // Pulse the clock pin to shift the bit into the register
    digitalWrite(clockPin, HIGH);
    digitalWrite(clockPin, LOW);
  }

  // Set the latch pin high to update the output of the shift register
  digitalWrite(latchPin, HIGH);
  delay(200);  // Add a short delay to avoid flooding the register
}
