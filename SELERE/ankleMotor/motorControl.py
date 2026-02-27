import pyCandle
import sys
import time
import math
import os

# Create CANdle object and ping FDCAN bus in search of drives.
# Any found drives will be printed out by the ping() method.

def setupCandle():
    global candle 

    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)

    global MOTORS
    
    motors = candle.ping()

    MOTORS = motors

    for motor in motors:
        candle.addMd80(motor)
        if motor == 100: 
            print('left motor found')
            print('motor ID:',motor)
        elif motor==200: 
            print('right motor found')
            print('motor ID:',motor)


    if len(motors)==0: print('No ankle motors found')

    for motor in motors:
        candle.md80s[motors.index(motor)].setPositionControllerParams(0.05, 0.5, 0.0, 1.5) #Ki, Kp Kd, iWindup
        candle.md80s[motors.index(motor)].setVelocityControllerParams(0.05, 0.5, 0.0, 1.5) #Ki, Kp Kd, iWindup
        print(f'motor {motor} at current position {candle.md80s[motors.index(motor)].getPosition()}')
        setPosition(motor)

    candle.begin()
    return (candle, motors) 

def position(id, position):
    startTime=time.time()
    
    motor = next((m for m in candle.md80s if m.getId() == id), None)

    if motor:

        currRadian=motor.getPosition()
        relativeRadianChange=math.radians(position)
        newRadian = currRadian + relativeRadianChange

        motor.setTargetPosition(newRadian)
        while not motor.isTargetPositionReached():
# Force a check of the bus health
            print(f"Bus Frequency: {candle.getActualCommunicationFrequency()} Hz")
            print('original position',currRadian)
            print('motor at',motor.getPosition())
            print('moving to: ',newRadian)
            # time.sleep(0.01)
        endTime = time.time()
        print('TOTAL ACTUATION TIME: ', endTime-startTime)
    else:
        print(f"Motor ID {id} not found, unable to actuate")

def calibrate(id):
    candle.setupMd80Calibration(id)
    candle.configMd80Save(id)    
    print('motor:',id, 'calibrated')
    motor = candle.getMd80FromList(id)
    print(motor.getQuickStatus())

def setVelocity(id):
    candle.controlMd80Enable(id, False)  
    candle.controlMd80Mode(id, pyCandle.VELOCITY_PID)
    candle.controlMd80Enable(id, True) 

def setPosition(id): 
    candle.controlMd80Enable(id, False)  
    candle.controlMd80Mode(id, pyCandle.POSITION_PID)
    candle.controlMd80Enable(id, True)

def setTorque(id):
    candle.controlMd80Enable(id, False)
    candle.controlMd80Mode(id, pyCandle.RAW_TORQUE)
    candle.controlMd80Enable(id, True) 

def velocity(id, index, velocity):
    setVelocity(id=id)
    candle.md80s[MOTORS.index(id)].setTargetVelocity(velocity)

def torque(id, index, torque):  
    setTorque()  
    candle.md80s[MOTORS.index(id)].setTargetTorque(torque)

def stopCandle():
    candle.end()

def main():
    candleObjects = setupCandle()
    while True:
        choice = input()
        if choice == "1":
            position(candleObjects[1][0], 0, math.pi/4)
            stopCandle(candleObjects[0])
            position(candleObjects[1][0], 0, 0)
            stopCandle(candleObjects[0])

        if choice == "2":
            velocity(candleObjects[1][0], 0, -0.5)
            time.sleep(2)
            stopCandle(candleObjects[0])
            velocity(candleObjects[1][0], 0, 0.5)
            time.sleep(2)
            stopCandle(candleObjects[0])
            velocity(candleObjects[1][0], 0, 0)
            stopCandle(candleObjects[0])

        if choice == "3":
            torque(candleObjects[1][0], 0, 1)
            time.sleep(10)
            stopCandle(candleObjects[0])
            torque(candleObjects[1][0], 0, 0)
            stopCandle(candleObjects[0])

if __name__ == "__main__":
    main()
