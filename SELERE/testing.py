import threading
import time

# Class for Knee Motor
class KneeMotor:
    def __init__(self):
        self.position = 0
        self.speed = 0
        self.acceleration = 0
        self.torque = 0
        self.rangeOfMotionTop = 100
        self.rangeOfMotionBottom = 0

    def extend(self, rangeOfMotionTop, rangeOfMotionBottom, speed, acceleration):
        self.rangeOfMotionTop = rangeOfMotionTop
        self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = speed
        self.acceleration = acceleration
        print("Extending Knee: Range [{}-{}], Speed {}, Acceleration {}".format(
            rangeOfMotionBottom, rangeOfMotionTop, speed, acceleration))
        time.sleep(1)

    def retract(self, rangeOfMotionTop, rangeOfMotionBottom, speed, acceleration):
        self.rangeOfMotionTop = rangeOfMotionTop
        self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = speed
        self.acceleration = acceleration
        print("Retracting Knee: Range [{}-{}], Speed {}, Acceleration {}".format(
            rangeOfMotionBottom, rangeOfMotionTop, speed, acceleration))
        time.sleep(1)

    def assist(self, torque):
        self.torque = torque
        print("Assisting Knee with Torque:", torque)

    def resist(self, torque):
        self.torque = torque
        print("Resisting Knee with Torque:", torque)


# Class for Ankle Motor
class AnkleMotor:
    def __init__(self):
        self.position = 0
        self.speed = 0
        self.acceleration = 0
        self.torque = 0
        self.rangeOfMotionTop = 100
        self.rangeOfMotionBottom = 0

    def extend(self, rangeOfMotionTop, rangeOfMotionBottom, speed, acceleration):
        self.rangeOfMotionTop = rangeOfMotionTop
        self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = speed
        self.acceleration = acceleration
        print("Extending Ankle: Range [{}-{}], Speed {}, Acceleration {}".format(
            rangeOfMotionBottom, rangeOfMotionTop, speed, acceleration))
        time.sleep(1)

    def retract(self, rangeOfMotionTop, rangeOfMotionBottom, speed, acceleration):
        self.rangeOfMotionTop = rangeOfMotionTop
        self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = speed
        self.acceleration = acceleration
        print("Retracting Ankle: Range [{}-{}], Speed {}, Acceleration {}".format(
            rangeOfMotionBottom, rangeOfMotionTop, speed, acceleration))
        time.sleep(1)

    def assist(self, torque):
        self.torque = torque
        print("Assisting Ankle with Torque:", torque)

    def resist(self, torque):
        self.torque = torque
        print("Resisting Ankle with Torque:", torque)


# Class for User Interface
class UserInterface:
    def __init__(self):
        self.modeButton = False
        self.button1 = False
        self.button2 = False
        self.button3 = False
        self.button3_state = 0
        self.button3_states = [0, 1, 2, 3]  # Example states for button3

    def press_mode_button(self):
        self.modeButton = True

    def release_mode_button(self):
        self.modeButton = False

    def press_button1(self):
        self.button1 = True

    def release_button1(self):
        self.button1 = False

    def press_button2(self):
        self.button2 = True

    def release_button2(self):
        self.button2 = False

    def press_button3(self):
        self.button3 = True
        self.button3_state = (self.button3_state + 1) % len(self.button3_states)
        print("Button3 State Changed to:", self.button3_state)

    def release_button3(self):
        self.button3 = False


# Exoskeleton Class containing modes and motors
class Exoskeleton:
    def __init__(self):
        self.modes = ["Mode 1", "Mode 2", "Mode 3"]
        self.currentMode = self.modes[0]
        self.kneeMotor = KneeMotor()
        self.ankleMotor = AnkleMotor()
        self.userInterface = UserInterface()

    def nextMode(self):
        current_index = self.modes.index(self.currentMode)
        self.currentMode = self.modes[(current_index + 1) % len(self.modes)]
        print("Switched to", self.currentMode)

    def handle_knee_motor(self):
        self.kneeMotor.extend(100, 0, self.userInterface.button3_state, 5)
        self.kneeMotor.retract(100, 0, self.userInterface.button3_state, 5)

    def handle_ankle_motor(self):
        self.ankleMotor.extend(100, 0, self.userInterface.button3_state, 5)
        self.ankleMotor.retract(100, 0, self.userInterface.button3_state, 5)


# Function to start the Exoskeleton and its loop in a separate thread
def start_exoskeleton():
    exo = Exoskeleton()

    while True:
        command = input("Enter command (mode, button1, button2, button3, quit): ").strip().lower()

        if command == "mode":
            exo.userInterface.press_mode_button()
            exo.nextMode()
            exo.userInterface.release_mode_button()
        elif command == "button1":
            exo.userInterface.press_button1()
            exo.handle_knee_motor()  # Button1 controls Knee Motor
            exo.userInterface.release_button1()
        elif command == "button2":
            exo.userInterface.press_button2()
            exo.handle_ankle_motor()  # Button2 controls Ankle Motor
            exo.userInterface.release_button2()
        elif command == "button3":
            exo.userInterface.press_button3()
            exo.userInterface.release_button3()
        elif command == "quit":
            print("Exiting...")
            break
        else:
            print("Invalid command.")

# Run the program
start_exoskeleton()
