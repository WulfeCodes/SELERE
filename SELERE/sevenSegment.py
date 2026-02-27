import RPi.GPIO as GPIO
import time

# Define shift register pins (using BCM numbering)
latchPin = 25  # GPIO25 (physical pin 22)
clockPin = 24  # GPIO24 (physical pin 18)
dataPin = 23   # GPIO23 (physical pin 16)

# Define button pins
modeButtonPin = 16  # GPIO16 (physical pin 36)
speedButtonPin = 6  # GPIO6 (physical pin 31)

# Define debounce time
debounceDelay = 0.05  # 50 milliseconds in seconds

# Variables to store the current and last state of each button
modeState = GPIO.LOW
speedState = GPIO.LOW
lastModeState = GPIO.LOW
lastSpeedState = GPIO.LOW

# Variables to store the last time the button state changed
lastDebounceTime1 = 0
lastDebounceTime2 = 0

# Define modeNumbers and speedNumbers arrays
modeNumbers = [0b01100000, 0b11011010, 0b11110010]
speedNumbers = [0b01100000, 0b11011010, 0b11110010, 0b01100110]

# Current display modes
modeDisp = 0   # Adjusted to zero index for array access
speedDisp = 0  # Adjusted to zero index for array access

def setup():
    global latchPin, clockPin, dataPin
    global modeButtonPin, speedButtonPin

    # Set up GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Configure shift register pins as outputs
    GPIO.setup(latchPin, GPIO.OUT)
    GPIO.setup(clockPin, GPIO.OUT)
    GPIO.setup(dataPin, GPIO.OUT)

    # Configure button pins as inputs with pull-up resistors
    GPIO.setup(modeButtonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(speedButtonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def loop():
    global modeState, speedState, lastModeState, lastSpeedState
    global lastDebounceTime1, lastDebounceTime2
    global modeDisp, speedDisp
    global debounceDelay

    # Read the current state of both buttons
    modeButtonReading = GPIO.input(modeButtonPin)
    speedButtonReading = GPIO.input(speedButtonPin)

    currentTime = time.time()

    # Debounce mode button
    if modeButtonReading != lastModeState:
        lastDebounceTime1 = currentTime  # Reset the debounce timer for mode button

    if (currentTime - lastDebounceTime1) > debounceDelay:
        # Update modeState if the reading has been stable
        if modeButtonReading != modeState:
            modeState = modeButtonReading
            # Take action if mode button is pressed
            if modeState == GPIO.LOW:  # Button is active LOW
                modeDisp += 1
                if modeDisp > 2:  # Ensure it stays within bounds for modeNumbers
                    modeDisp = 0  # Loop back to the first mode
                speedDisp = 0  # Reset speed to 1 (index 0) whenever mode is changed
                print("modeDisp:", modeDisp + 1, flush=True)
                print("speedDisp:", speedDisp + 1, flush=True)
                updateShiftRegister()  # Update the shift register after button press

    lastModeState = modeButtonReading  # Update last button state

    # Debounce speed button
    if speedButtonReading != lastSpeedState:
        lastDebounceTime2 = currentTime  # Reset the debounce timer for speed button

    if (currentTime - lastDebounceTime2) > debounceDelay:
        # Update speedState if the reading has been stable
        if speedButtonReading != speedState:
            speedState = speedButtonReading
            # Take action if speed button is pressed
            if speedState == GPIO.LOW:  # Button is active LOW
                speedDisp += 1
                if speedDisp > 3:  # Ensure it stays within bounds for speedNumbers
                    speedDisp = 0  # Loop back to the first speed
                print("modeDisp:", modeDisp + 1, flush=True)
                print("speedDisp:", speedDisp + 1, flush=True)
                updateShiftRegister()  # Update the shift register after button press

    lastSpeedState = speedButtonReading  # Update last button state

def updateShiftRegister():
    global modeDisp, speedDisp
    global latchPin, clockPin, dataPin
    global modeNumbers, speedNumbers

    # Combine the current mode and speed values with mode first
    combinedValue = (modeNumbers[modeDisp] << 8) | speedNumbers[speedDisp]

    # Set the latch pin to low to start shifting out
    GPIO.output(latchPin, GPIO.LOW)

    # Shift out the bits to the shift register in LSBFIRST order
    for i in range(16):  # Send 16 bits, LSB first
        # Get the current bit (from LSB to MSB)
        bit = (combinedValue >> i) & 1

        # Send the bit to the data pin
        GPIO.output(dataPin, bit)
        # Pulse the clock pin to shift the bit into the register
        GPIO.output(clockPin, GPIO.HIGH)
        GPIO.output(clockPin, GPIO.LOW)

    # Set the latch pin high to update the output of the shift register
    GPIO.output(latchPin, GPIO.HIGH)
    time.sleep(0.2)  # Add a short delay to avoid flooding the register

if __name__ == "__main__":
    try:
        setup()
        while True:
            loop()
    except KeyboardInterrupt:
        GPIO.cleanup()
