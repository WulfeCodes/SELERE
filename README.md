This project consists of the actuation of the CubeMars Ak10-9 KV60 V2.0 and Mav Robotics MD-80 motors for knee and ankle actuation respectively.

Communication with the actuators are all done through CAN, the MD-80s specifically utilize the PyCandle library from : https://github.com/mabrobotics/candle. 

Within the SELERE file there exists base code for CAN communication and actuation, these can be found in the ankleMotor, and kneeMotor folders. 

CubeMars Ak10-9 KV60 V2.0: as stated before, this actuator is controlled through CAN, specifically we utilize the RS-485 CAN Hat. 

the classes.py file features the ExoSkeleton class, within this class there are KneeMotor and AnkleMotor members for each joint, 4 in total. As of right now, these are only used for storage of hardware CAN IDs and safe motor selection. 

For Hardware configuration it is best practice for the CAN line to have two 120 Ohm parallel resistors between the H and L CAN lines, this helps mitigate loopback and helps facilitate difference measurement of the two for proper bit readage. 

The RS845 CAN Hat has embedded resistors that can be activated, though a parallel Ohm resistor should still be soldered near the actuator side for robust communication. 

The CubeMars Ak10-9 KV60 V2.0 Knee Motors' communication protocol and actuation functions derive from the /kneeMotor/motorCAN.py and /kneeMotor/motorControl.py respectively. 

For motor parameters of the knee motor, it falls under a bit rate of 500k with arbitration ID labels of different actuation types for the following: 

CAN_PACKET_SET_DUTY = 0
CAN_PACKET_SET_CURRENT = 1
CAN_PACKET_SET_CURRENT_BRAKE = 2
CAN_PACKET_SET_RPM = 3
CAN_PACKET_SET_POS = 4
CAN_PACKET_SET_ORIGIN_HERE = 5
CAN_PACKET_SET_POS_SPD = 6

Specific actuation functions are given in /kneeMotor/motorControl.py 

through current(), speed(), position_speed_acceleration(), current_brake(), move_to_desired_angle(), 

def move_to_desired_angle():

These are all parameterized the same, the argument format typically follows (bus, data, controller_id=CONTROLLER_ID) and each return a tuple of the bus, eid buffer, and data buffer. 

The buffers are expected to be 4 bytes long with the eid being the OR operation of the 8 bit shifted actuation ID type | controller ID you are trying to control: for ex: the construction of an eid buffer for position control: CAN_PACKET_SET_POS = 4 of a controller ID 1 is the 8 byte buffer of: [00000000,00000000,00000100,00000001]

:

eid = ((CAN_PACKET_SET_POS << 8) | controller_id) 


As stated in the documentation the the data buffer should be in big endian style with the absolute position scaled by 10,000:     

buffer = struct.pack('>i', int(position * 10000))

Once the function is called, it is typical in our set up to unpack these and send them as arguments into the comm_can_transmit_eid(bus, eid, data) found in kneeMotor/motorCAN.py

this sends a a message through :    

  message = can.Message(
        arbitration_id=eid,  # Extended ID
        is_extended_id=True, # Use extended ID
        data=data            # Data payload
    )
    
  try:
  bus.send(message)
  
except can.CanError as e:
  print(f"Error sending message: {e}")

WARNING: when operating the CubeMars AK-109 KV60 V2 motor MIT mode and Servo Mode can not be operated within the same power session.  

For example: the probe_bitrate() and scan_bus() functions within vijayTesting.py ping components on the CAN line and print the ID and message received, the data buffer of the outgoing message is in MIT mode, as such if one wishes to check CAN IDs of motors on the CAN line through one of these functions and then actuate them through the Servo Mode, these should not be done within the same power session as it endangers burning the driver board. 

MD80: 

The MD80 Motor as stated before communicates through the 

CAN line initialization: 

for the initialization of the RS845 CAN line the following commands should be called to create a bus on the correct bitrate for communication of the knee motors: 

os.system(f'sudo ip link set {channel} down')
os.system(f'sudo ip link set {channel} type can bitrate {bitrate}')
os.system(f'sudo ip link set {channel} up')
time.sleep(0.1)  # let interface settle

bus = can.interface.Bus(interface='socketcan', channel=channel)

As for the MD80, the creation of a PyCandle object abstracts this away:

candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)



<img width="3000" height="1750" alt="Image" src="https://github.com/user-attachments/assets/3d62464b-2783-4822-9883-b0fb6f001be5" />
