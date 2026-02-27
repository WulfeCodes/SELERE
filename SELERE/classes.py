import time
import kneeMotor.motorCAN
import kneeMotor.motorControl

# Class for Knee Motor
class KneeMotor:
    def __init__(self, id, name, direction, maxHeight, minHeight):
        self.position = 0
        self.speed = 0
        self.current = 0
        self.temp = 0
        self.errorCode=0
        self.torque = 0
        self.maxHeight = maxHeight
        self.minHeight = minHeight
        self.rangeOfMotion = abs(maxHeight - minHeight)
        self.canbus = 0
        self.initialDirection = direction
        self.currentDirection = direction
        self.desSpd = 0
        self.desHeight = 0
        self.maxSpd = 1250
        self.minSpd = 0
        self.maxCurrent = 5000
        self.maxAssist = self.maxCurrent
        self.minAssist = 0
        self.maxResist = self.maxCurrent
        self.minResist = 0
        self.desCurrent = 0
        self.id = id
        self.name = name

    def extend(self, rangeOfMotionTop, desiredPosition, desiredSpeed, desiredAcceleration):
        #self.rangeOfMotionTop = rangeOfMotionTop
        #self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = desiredSpeed
        self.acceleration = desiredAcceleration
        print("Extending Knee: Range [{}-{}], desiredSpeed {}, desiredAcceleration {}".format(
            desiredPosition, rangeOfMotionTop, desiredSpeed, desiredAcceleration))
        kneeMotor.motorControl.position_speed_acceleration(self.canbus, self.rangeOfMotionTop-desiredPosition, desiredSpeed, desiredAcceleration)
        self.position = desiredPosition
        time.sleep(1)

    def retract(self, desiredPosition, rangeOfMotionBottom, desiredSpeed, desiredAcceleration):
        #self.rangeOfMotionTop = rangeOfMotionTop
        #self.rangeOfMotionBottom = rangeOfMotionBottom
        self.speed = desiredSpeed
        self.acceleration = desiredAcceleration
        kneeMotor.motorCAN.write_log("Retracting Knee: Range [{}-{}], desiredSpeed {}, desiredAcceleration {}".format(
             desiredPosition, rangeOfMotionBottom, desiredSpeed, desiredAcceleration), log_dir="logs")
        
        kneeMotor.motorControl.position_speed_acceleration(self.canbus, rangeOfMotionBottom-desiredPosition, desiredSpeed, desiredAcceleration)
        self.position=desiredPosition
        time.sleep(1)

    def assist(self, torque):
        self.torque = torque
        kneeMotor.motorControl.current(self.canbus, torque)
        kneeMotor.motorCAN.write_log("Assisting Knee with Torque:" + torque, log_dir="logs")
     
    def resist(self, torque):
        self.torque = torque
        kneeMotor.motorControl.current(self.canbus, torque)
        kneeMotor.motorCAN.write_log("Resisting Knee with Torque:", torque)

    def getPosition(self):
        return self.position 
    
    def getSpeed(self):
        return self.speed

    def getDesSpeed(self):
        return self.desSpd

    def getDesCurrent(self):
        return self.desCurrent


# Class for Ankle Motor
class AnkleMotor:
    def __init__(self, index, name, direction, maxHeight, minHeight):
        self.position = 0
        self.speed = 0
        self.current = 0
        self.temp = 0
        self.errorCode=0
        self.torque = 0
        self.maxHeight = maxHeight
        self.minHeight = minHeight
        self.rangeOfMotion = abs(maxHeight - minHeight)
        self.initialDirection = direction
        self.currentDirection = direction
        self.desSpd = 0
        self.desHeight = 0
        self.maxSpd = 1
        self.minSpd = 0
        self.maxCurrent = 2
        self.maxAssist = self.maxCurrent
        self.minAssist = 0
        self.maxResist = self.maxCurrent
        self.minResist = 0
        self.desCurrent = 0
        self.id = 0
        self.candle = 0
        self.index = index
        self.name = name

    def getPosition(self):
        return self.candle.md80s[self.index].getPosition()

    def getSpeed(self):
        return self.speed

    def getDesSpeed(self):
        return self.desSpd

    def getDesCurrent(self):
        return self.desCurrent
    
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
   
        

# Exoskeleton Class containing modes and motors
class Exoskeleton:

    def __init__(self):
        self.modeFA = Mode("Full", 1)
        self.modePA = Mode("Partial", 2)
        self.modePR = Mode("Resistance", 3)
        self.modes = (self.modeFA, self.modePA, self.modePR)
        self.currentMode = self.modes[0]
        self.leftKnee = KneeMotor(0, "Left Knee", -1, 1000, 1) #id, name, initial direction, max height, min height
        self.leftAnkle = AnkleMotor(0, "Left Ankle", 1, 3/4, 0.01)
        self.leftAnkle.id=100
        self.rightKnee = KneeMotor(1, "Right Knee", 1, 1000, 1)
        self.rightAnkle = AnkleMotor(1, "Right Ankle", -1, 3/4, 0.01)
        self.joints = (self.leftKnee, self.rightKnee, self.leftAnkle, self.rightAnkle)
        self.currentJoint = self.joints[0]
        self.states = ("stopped", "started")
        self.currentState = self.states[0]

class Mode:
    def __init__(self, name, number):
        self.name = name
        self.number = number
        self.height = (0,1) #format (minHeight, maxHeight)

 

