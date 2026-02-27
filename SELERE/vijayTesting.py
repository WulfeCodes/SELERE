
from classes import Exoskeleton
import tkinter as tk
import can
import struct
import subprocess

from kneeMotor.motorCAN import start_can, tkinter_loop, comm_can_transmit_eid, write_log, can_handler_thread
from kneeMotor.motorControl import current, speed, current_brake, position_speed_acceleration, move_to_desired_angle, set_origin
from ankleMotor.motorControl import velocity, torque, setVelocity, setTorque, setupCandle, position, calibrate
from PIL import Image, ImageTk

import os
from datetime import datetime
import time
import threading
from kneeMotor.motorControl import set_origin
from ankleMotor.motorControl import setupCandle

CHANNEL = 'can0'
BITRATE = BITRATE = 1000000
BUSTYPE = 'socketcan'

def probe_bitrate(channel='can0'):
    ENTER_MODE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
    EXIT_MODE  = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
    
    for bitrate in [1000000, 500000]:
        print(f"\nTrying {bitrate} bps...")
        os.system(f'sudo ip link set {channel} down')
        os.system(f'sudo ip link set {channel} type can bitrate {bitrate}')
        os.system(f'sudo ip link set {channel} up')
        time.sleep(0.1)  # let interface settle
        
        bus = can.interface.Bus(interface='socketcan', channel=channel)
        
        for motor_id in range(0, 129):  # just check low IDs first, faster
            msg = can.Message(arbitration_id=motor_id, data=ENTER_MODE, is_extended_id=False)
            try:
                bus.send(msg)
                reply = bus.recv(timeout=0.05)
                if reply:
                    print(f"GOT REPLY at bitrate={bitrate}, motor_id={motor_id}")
                    print(f"  reply arb_id=0x{reply.arbitration_id:X}, data={reply.data.hex()}")
                    bus.close()
                    return bitrate, motor_id
            except can.CanError as e:
                print(f"  CAN error: {e}")
                break
        
        bus.close()
    
    print("No response at any bitrate.")
    return None, None

def scan_bus():
    print(f"Connecting to {CHANNEL}...")
    bus = can.interface.Bus(channel=CHANNEL, bustype=BUSTYPE, bitrate=BITRATE)
    
    found_ids = []

    print("Scanning for CubeMars actuators (IDs 1-127)...")
    
    # Try every valid ID
    for potential_id in range(1, 128):
        # Construct a 'Get Status' or 'Ping' packet.
        # For CubeMars, sending a 0-command to the position loop is usually safe 
        # provided the motor is not enabled, OR just sending a known query packet.
        # A safe bet is usually the "Get Motor Status" or just an empty packet 
        # to the specific ID if the protocol supports it. 
        
        # CubeMars often replies to a position command even if 0.
        # We will set the command to 0 to be safe.
        
        # EID Structure: [CMD: 5 bits] [ID: 8 bits]
        # We'll use a harmless command like "Get Data" if available, 
        # or a 0-torque command.
        # Let's assume sending a generic packet to the ID.
        
        # NOTE: In many AK firmware versions, they only reply if you send
        # a valid command frame.
        # We will try sending a "Zero Torque" command which is safe.
        # CMD = 1 (Set Current/Torque)
        
        eid = (1 << 8) | potential_id 
        
        # Payload: 0 current (Safe!)
        data = struct.pack('>h', 0) * 2 # 4 bytes of zeros
        
        msg = can.Message(arbitration_id=eid, data=data, is_extended_id=True)
        
        try:
            bus.send(msg)
            
            # Wait briefly for a reply
            reply = bus.recv(timeout=0.01) # 10ms wait
            
            if reply:
                # If we got ANY reply, that ID exists
                # We extract the ID from the reply arbitration_id
                # The reply ID usually matches the send ID logic
                reply_id = reply.arbitration_id & 0xFF
                
                if reply_id == potential_id:
                    print(f"--> FOUND DEVICE AT ID: {potential_id}")
                    found_ids.append(potential_id)
                    
        except can.CanError:
            print("CAN Bus Error")
            break

    print("-" * 30)
    print(f"Scan Complete. Found {len(found_ids)} devices.")
    print(f"IDs: {found_ids}")



def main():
    # 1. Setup Exoskeleton
    currExo = Exoskeleton()
    
    # (Optional) Update the exo objects with the real bus if you want to store it
    # currExo.leftKnee.canbus = can0 

    # Debug prints (Fixed your copy-paste typos in labels)
    print('Debug - Left Knee Bus:', currExo.leftKnee.canbus)
    print('Debug - Right Knee Bus:', currExo.rightKnee.canbus)

    ready = input('Are you ready? type "y" for yes: ')
    
    if ready.lower() == 'y':

        #NOTE pings the bus and sees what IDs are given back (these are the motors)
        #NOTE of those IDs it sets the current positional and velocity PID gains 

        #NOTE this functionality derives from the start_can function within motorCAN.py
        kneeMotors = [currExo.leftKnee, currExo.rightKnee]
        ankleMotors = [currExo.leftAnkle, currExo.rightAnkle]
        components = [kneeMotors, ankleMotors]
        print("Initializing CAN bus...")
        # Create the REAL bus object here

        probe_bitrate()
        input('u done fam?')

        write_log("Setting bitrate for can0...")
        os.system('sudo ip link set can0 type can bitrate '+str(500000))
        write_log("Bringing can0 interface up...")
        os.system('sudo ip link set can0 up')  # Bring the interface up
        write_log("can0 interface is up.")
        write_log('checking for errors in can line')
        result = subprocess.run(['ip', '-details', 'link', 'show', 'can0'], capture_output=True, text=True)
        write_log(f"can0 status:\n{result.stdout}")

        # Create the CAN bus interface
        print("Initializing CAN bus...")
        can0 = can.interface.Bus(interface='socketcan', channel='can0')
        print("CAN bus initialized successfully.")

        # Create thread for receiving messages
        receiver_thread = threading.Thread(target=can_handler_thread, args=(can0, components[0]), daemon=True)
        # Start the receiver thread
        receiver_thread.start()
        print("CAN bus initialized successfully.")

        print('starting ankle CAN and checking for available motors')
        setupCandle()

        print('checking for knee motors..')
        scan_bus()

        # --- FIX 2: ZERO ONCE BEFORE THE LOOP ---
        # We use the ID from the exoskeleton object to be safe

        currAngle = ''
        while True:
            currJoint = input("Enter the desired joint: (choose-> ankle or knee)")
            currDirection = input(f"Enter the desired direction for the {currJoint} (choos-> left or right)")

            if currJoint.lower() == 'ankle': 
                if currDirection.lower() == 'left':
                    currMotor = currExo.leftAnkle

                elif currDirection.lower() == 'right':
                    currMotor = currExo.rightAnkle

                else:
                    print(f'not a valid direction for {currJoint} please choose: left or right')
                    continue

            elif currJoint.lower() == 'knee': 
                if currDirection.lower() == 'left':
                    currMotor = currExo.leftKnee

                elif currDirection.lower() == 'right':
                    currMotor = currExo.rightKnee

                else:
                    print(f'not a valid direction for {currJoint} please choose: left or right') 
                    continue
            else: 
                print('not a valid joint choice, please choose: ankle or knee') 
                continue
            
            currAngle=input("please type in the desired angle in degrees: 0-360 \n type 'q' to quit")
            if currAngle.lower() == 'q':
                break
            currAngle = max(0, min(float(currAngle), 360))            
            # Check for quit BEFORE trying math


            input(f'actuation angle chosen: {currMotor} degrees, type anything to actuate')
            target_pos  = currAngle

            if currJoint.lower() == 'knee':

                try:
                    print("Zeroing Right Knee...")
                    comm_can_transmit_eid(*set_origin(can0, 
                                                    set_origin_mode=0, 
                                                    controller_id=currMotor.id))
                    # --- FIX 3: PASS THE SPECIFIC MOTOR ID ---
                    # We use 'currExo.leftKnee.id' so we talk to the correct joint
                    comm_can_transmit_eid(*move_to_desired_angle(
                        bus=can0, 
                        position=target_pos, 
                        controller_id=currMotor.id
                    ))
                    print(f"Sent command: Move {currMotor.name} to {target_pos}°")
                    
                except ValueError:
                    print("Invalid input! Please enter a number.")

            elif currJoint.lower() == 'ankle':
                print('just checking for ankle indexes')
                try:
                    ip=input('do you want to calibrate? type yes or no')
                    if 'yes' in ip.lower():
                        calibrate(id=currMotor.id)
                    position(id =currMotor.id ,position=target_pos,choice=0)
                    
                except ValueError:
                    print("Invalid input! Please enter a number.")

    # Cleanup
    print("Shutting down.")
    # can0.shutdown() # Good practice if your library version supports it

    #currVal=currExo.leftKnee.extend()
    #extend calls the position_speed_acceleration, but this return tuple is not used,
    #^ that needs to be given to the comm_can_transmit_eid
    # comm_can_transmit_eid(*current_brake())
    # comm_can_transmit_eid(*speed())
    # comm_can_transmit_eid(*current())
    
    ##^^ returns the bus, and two 32 bit numbers (buffer, 8 bit shifted control type | control id)

    #each return a bus, eid, buffer tuple
    # ^these are given in a obj=can.Message() that can be sent with bus.send_message(obj)
    #eid = arbitration id, first 8 bits is destination id, next 8 bits is packet_type

    # input('are you ready? input left knee retraction value')

    # currExo.leftKnee.retract()

    # input('are you ready? input right knee extension value')

    # currExo.rightKnee.extend()

    # input('are you ready? input right ankle retraction value')

    # currExo.rightKnee.retract()


    # ##NOTE rightAnkle extend and react functions dont send any messages

    # input('are you ready? input right ankle extension value')

    # currExo.rightAnkle.extend()

    # input('are you ready? input right ankle retraction value')

    # currExo.rightAnkle.retract()

    # input('are you ready? input left ankle extension value')

    # currExo.leftAnkle.extend()

    # input('are you ready? input left ankle retraction value')

    # currExo.leftAnkle.retract()


    ##^ has three inited modes: currMode is the first of these
        ##^ same for the joints
        ##^ same for states
        #^ likely can only control one joint at a time for this 
        ##
        ##each joint has given movement classes that call .position_speed_acceleration that is the main control class
    
    ##within classes theres an important settings variable

    ##the set_mode() function has possibly incorrect parameters, calls setVelocity/setTorque
    ##the initialization of each joint motor class, isometric sides have same id


if __name__ == "__main__":
    print("This is the main function of vijayTesting.py")
    main()


