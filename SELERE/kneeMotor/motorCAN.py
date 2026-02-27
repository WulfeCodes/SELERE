import os
import can
from datetime import datetime
import time
import threading
from kneeMotor.motorControl import set_origin
from ankleMotor.motorControl import setupCandle

#Motor parameters
BITRATE = 500000
LEFT_KNEE_ID = 0x2900
RIGHT_KNEE_ID = 0x2901

def write_log(log_text, log_dir="logs"):
    #print(log_text)  #for live debugging
    
    # Get the current date for the file name
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{log_dir}/{date_str}.txt"

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Get the current time to include with each entry
    time_str = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{time_str} - {log_text}"

    # Append the log entry to the file, creating it if it doesn't exist
    with open(log_filename, "a") as log_file:
        log_file.write(log_entry + "\n")

def can_handler_thread(bus, jointMotors):
    #Continuously receive CAN messages and write them to log files
    jointIndex = 0
    while True:
        msg = bus.recv()  # Wait for a new message        
        if msg:
            #print(msg.arbitration_id)
            if msg.arbitration_id == LEFT_KNEE_ID: #0x2900 is the ID of the left Knee motor
                jointIndex = 0
            if msg.arbitration_id == RIGHT_KNEE_ID: #0x2900 is the ID of the right Knee motor
                jointIndex = 1
            try:
                jointMotors[jointIndex].position = (msg.data[0]<<8)+msg.data[1]
                jointMotors[jointIndex].speed = (msg.data[2]<<8)+msg.data[3]
                jointMotors[jointIndex].current = (msg.data[4]<<8)+msg.data[5]
                jointMotors[jointIndex].temp = msg.data[6]
                jointMotors[jointIndex].errorCode = msg.data[7]

                if jointMotors[jointIndex].position > 32768:
                    jointMotors[jointIndex].position = jointMotors[jointIndex].position - 65536 
            
            except Exception as e:
                print(f"Error extracting parameter, message: {e}")

            #write_log(msg)


def comm_can_transmit_eid(bus, eid, data):
    # Ensure data length is within CAN limits
    if len(data) > 8:
        data = data[:8]
    
    # Create a CAN message
    message = can.Message(
        arbitration_id=eid,  # Extended ID
        is_extended_id=True, # Use extended ID
        data=data            # Data payload
    )
    
    # Send the message
    try:
        bus.send(message)
    except can.CanError as e:
        print(f"Error sending message: {e}")

def demo_event_loop(canBus):
    while True:
        userInput=input("enter position")
        #Do something

def tkinter_loop(jointMotors, canBus, tkLoop):
    candleObjects = setupCandle()
    candle = candleObjects[0]
    ids = candleObjects[1]
    for kneeMotor in jointMotors[0]:
        kneeMotor.canbus = canBus
        comm_can_transmit_eid(*set_origin(canBus, 0, controller_id=kneeMotor.id))
    for ankleMotor in jointMotors[1]:
        ankleMotor.candle = candle
        if ids:
            ankleMotor.id = ids.pop(0)
        
    tkLoop()

def start_can(components, eventLoop, event):
    try:
        # Set up the CAN interface with the specified bitrate, This must match the Baud Rate parameter set in R-link
        write_log("Setting bitrate for can0...")
        os.system('sudo ip link set can0 type can bitrate '+str(BITRATE))
        write_log("Bringing can0 interface up...")
        os.system('sudo ip link set can0 up')  # Bring the interface up
        write_log("can0 interface is up.")

        # Create the CAN bus interface
        print("Initializing CAN bus...")
        can0 = can.interface.Bus(interface='socketcan', channel='can0')
        print("CAN bus initialized successfully.")

        # Create thread for receiving messages
        receiver_thread = threading.Thread(target=can_handler_thread, args=(can0, components[0]), daemon=True)
        # Start the receiver thread
        receiver_thread.start()
        # CALL MAIN EVENT LOOP BELOW, MUST BE A FUNCTION WITH A CONSTANT LOOP & EXPECTING A PARAMETER 'canbus'
        eventLoop(components, can0, event)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Bring the interface down
        print("Bringing can0 interface down...")
        os.system('sudo ip link set can0 down')
        print("can0 interface is down.")