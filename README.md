# SELERE — Exoskeleton Motor Actuation

This project handles actuation of the **CubeMars AK10-9 KV60 V2.0** (knee) and **MAB Robotics MD-80** (ankle) motors via CAN communication on a Raspberry Pi with an RS-485 CAN Hat.

---

## Table of Contents
- [Warnings](#warnings)
- [Project Structure](#project-structure)
- [Hardware Configuration](#hardware-configuration)
- [CAN Line Initialization](#can-line-initialization)
- [Knee Motor — CubeMars AK10-9 KV60 V2.0](#knee-motor--cubemarks-ak10-9-kv60-v20)
- [Ankle Motor — MAB Robotics MD-80](#ankle-motor--mab-robotics-md-80)
- [Software Architecture](#software-architecture)

---

**See warning below before use.**

---

## ⚠️ Warnings

### MIT Mode vs. Servo Mode — Do Not Mix in the Same Power Session

The `probe_bitrate()` and `scan_bus()` functions in `vijayTesting.py` send messages in **MIT mode**. The actuation functions in `kneeMotor/motorControl.py` operate in **Servo Mode**.

> **These two modes must never be used within the same power session on the CubeMars AK10-9 KV60 V2.0.** Doing so risks burning the motor driver board.

If you use `probe_bitrate()` or `scan_bus()` to inspect CAN IDs, **fully power cycle the motor** before running any Servo Mode actuation.


## Project Structure

```
SELERE/
├── kneeMotor/
│   ├── motorCAN.py       # CAN communication primitives
│   └── motorControl.py   # Actuation functions
├── ankleMotor/           # MD-80 base code
├── classes.py            # ExoSkeleton class definition
└── vijayTesting.py       # CAN bus probing utilities
```

The `ExoSkeleton` class in `classes.py` contains `KneeMotor` and `AnkleMotor` members for each joint (4 total), currently used for storing hardware CAN IDs and safe motor selection.

---

## Hardware Configuration

All actuators communicate over CAN. Best practices for the CAN line:

- Place two **120 Ω resistors in parallel** between the CAN H and L lines to mitigate loopback and enable differential bit measurement.
- The RS-485 CAN Hat has embedded termination resistors that can be activated, but an additional resistor should still be **soldered near the actuator end** for robust communication.

---

## CAN Line Initialization

For the RS-485 CAN Hat (used with knee motors), initialize the interface at the correct bitrate before creating the `bus` object:

```python
import os, time
import can

os.system(f'sudo ip link set {channel} down')
os.system(f'sudo ip link set {channel} type can bitrate {bitrate}')
os.system(f'sudo ip link set {channel} up')
time.sleep(0.1)  # Let interface settle

bus = can.interface.Bus(interface='socketcan', channel=channel)
```

For the MD-80, the PyCandle library abstracts this away:

```python
import pyCandle

candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
```

---

## Knee Motor — CubeMars AK10-9 KV60 V2.0

### Communication

- **Protocol:** CAN via RS-485 CAN Hat
- **Bit Rate:** 500k
- **Relevant files:** `kneeMotor/motorCAN.py`, `kneeMotor/motorControl.py`

### Packet Types

| Constant | Value | Description |
|---|---|---|
| `CAN_PACKET_SET_DUTY` | `0` | Set duty cycle |
| `CAN_PACKET_SET_CURRENT` | `1` | Set current |
| `CAN_PACKET_SET_CURRENT_BRAKE` | `2` | Set braking current |
| `CAN_PACKET_SET_RPM` | `3` | Set RPM |
| `CAN_PACKET_SET_POS` | `4` | Set position |
| `CAN_PACKET_SET_ORIGIN_HERE` | `5` | Set current position as origin |
| `CAN_PACKET_SET_POS_SPD` | `6` | Set position with speed |

### Actuation Functions (`kneeMotor/motorControl.py`)

All functions share the same signature pattern:

```python
func(bus, data, controller_id=CONTROLLER_ID)
# Returns: (bus, eid_buffer, data_buffer)
```

Available functions:
- `current()`
- `speed()`
- `position_speed_acceleration()`
- `current_brake()`
- `move_to_desired_angle()`

### Message Construction

**Extended ID (EID):** Constructed by OR-ing the 8-bit-shifted packet type with the controller ID:

```python
eid = (CAN_PACKET_SET_POS << 8) | controller_id
```

Example — position control (`CAN_PACKET_SET_POS = 4`) for controller ID `1`:

```
EID bytes: [0x00, 0x00, 0x04, 0x01]
```

**Data buffer:** Big-endian, position scaled by 10,000:

```python
import struct

buffer = struct.pack('>i', int(position * 10000))
```

### Transmitting a Message

Unpack the return values from a control function and pass them to `comm_can_transmit_eid()` in `kneeMotor/motorCAN.py`:

```python
message = can.Message(
    arbitration_id=eid,
    is_extended_id=True,
    data=data
)

try:
    bus.send(message)
except can.CanError as e:
    print(f"Error sending message: {e}")
```

---

## Ankle Motor — MAB Robotics MD-80

- **Protocol:** CAN via PyCandle
- **Library:** [mabrobotics/candle](https://github.com/mabrobotics/candle)
- **Relevant folder:** `ankleMotor/`

Initialization and communication are abstracted by the PyCandle library (see [CAN Line Initialization](#can-line-initialization) above).

---

## Software Architecture

### `classes.py` — `ExoSkeleton`

Holds `KneeMotor` and `AnkleMotor` instances for all four joints. Used for:
- Storing hardware CAN IDs
- Safe motor selection

### `vijayTesting.py`

Contains `probe_bitrate()` and `scan_bus()` utilities that ping CAN components and print their ID and received message. 
<img width="3000" height="1750" alt="Image" src="https://github.com/user-attachments/assets/3d62464b-2783-4822-9883-b0fb6f001be5" />
