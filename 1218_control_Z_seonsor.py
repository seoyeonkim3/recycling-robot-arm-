import numpy as np
import math
import time
from Arm_Lib import Arm_Device
from ikpy.chain import Chain
import serial

# Create a robotic arm object
Arm = Arm_Device()
time.sleep(0.1)

# Define the robotic arm model
my_chain = Chain.from_urdf_file("1218_dofbot_initialposition.urdf", active_links_mask=[False, True, True, True, True, True])

target_position = [0.0, 0.29, 0.21]  # arbitrary value
target_orientation = [0, 0, -1]

ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Z")

# Camera intrinsic parameters (K)
K = np.array([[946.5420, 0, 350.9673],
              [0, 942.4310, 259.2006],
              [0, 0, 1]])

# Camera-to-world transformation matrix configuration
R_wc = np.array([[1, 0, 0],
                 [0, 0, 1],
                 [0, -1, 0]])

P_corg = np.array([[0], [-8.5 + 7 + 3], [4.5 + 7.5 + 8.5 + 4.5 + 2]])  # Translation vector \( ^W_P_{CORG} \)

# Create the \( ^W_CT \) transformation matrix
T_wc = np.hstack((R_wc, P_corg))  # \( [^W_CR | ^W_P_{CORG}] \)
T_wc = np.vstack((T_wc, np.array([0, 0, 0, 1])))  # Expand to a \( 4 \times 4 \) matrix

# Radian to degree conversion
r2d = 180 / math.pi

def doIK():
    global ik
    old_position = ik.copy()
    ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y", initial_position=old_position)

def move(x, y, z, g, t):
    global target_position
    target_position = [x, y, z]
    doIK()

    r2d = 180 / math.pi
    m_1 = 90 + ik[1].item() * r2d
    m_2 = 180 + ik[2].item() * r2d
    m_3 = ik[3].item() * r2d
    m_4 = ik[4].item() * r2d
    m_5 = 90 + ik[5].item() * r2d
    Arm.Arm_serial_servo_write6(m_1, m_2, m_3, m_4, m_5, g, t)

# Move to the initial position
Arm.Arm_serial_servo_write6(90, 180, 0, 0, 90, 80, 2000)
time.sleep(2)

try:
    # Initialize serial communication with Arduino for Zc
    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    time.sleep(2)  # Allow time for Arduino to reset
    arduino.reset_input_buffer()

    # Load pixel coordinates saved from `1217_camera.py`
    pixel_coords = np.load('pixel_coordinates.npy')
    u, v = pixel_coords

    print("Waiting for Zc value from Arduino...")
    while True:
        raw_data = arduino.readline().decode('utf-8').strip()
        print(f"Raw data from Arduino: {raw_data}")
        try:
            if raw_data.startswith("D:"):
                Zc_mm = float(raw_data.split(":")[1])  # Extract Zc value in millimeters
                break
        except ValueError:
            print("Invalid data. Waiting for valid Zc value...")

    Zc = Zc_mm / 10.0  # Convert Zc to cm
    print(f"Received Zc from Arduino: {Zc:.2f} cm")

    # Calculate camera coordinates (P_c)
    Pc = Zc * np.linalg.inv(K) @ np.array([[u], [v], [1]])
    Pc_h = np.vstack((Pc, np.array([[1]])))  # Convert to homogeneous coordinates

    # Transform to world coordinates (P_w)
    Pw_h = T_wc @ Pc_h
    x, y, z = Pw_h[:3].flatten()  # Extract x, y, z coordinates in cm
    
    print(f"World Coordinates: x={x}, y={y}, z={z}")
    # Move the robotic arm
    x1 = x / 100.0 + 0.05
    y1 = y / 100.0 + 0.30
    z1 = z / 100.0 + 0.70
    
    # Move the robotic arm
    move(x1, y1, z1, 30, 2000)
    time.sleep(5)

    # Check proximity sensor before gripping
    proximity_detected = False
    for _ in range(5):  # Check proximity sensor 5 times to ensure consistency
        raw_data = arduino.readline().decode('utf-8').strip()
        if raw_data.startswith("P:"):
            proximity = int(raw_data.split(":")[1])
            if proximity == 1:
                proximity_detected = True
                break
        time.sleep(0.1)

    if not proximity_detected:
        print("No object detected at target position!")
        print(f"World Coordinates without object: x={x}, y={y}, z={z}")
        # Return to initial position
        Arm.Arm_serial_servo_write6(90, 180, 0, 0, 90, 80, 2000)
        time.sleep(2)
    else:
        print("Object detected! Proceeding to grip.")
        # Grip
        move(x1, y1, z1, 80, 2000)
        time.sleep(5)
        
        # Proceed with further steps (delete position, etc.)
        Arm.Arm_serial_servo_write6(90, 160, 0, 15, 90, 80, 5000)
        time.sleep(5)

        # Delete
        Arm.Arm_serial_servo_write6(90, 160, 0, 15, 270, 80, 5000)
        time.sleep(10)

        # Move to standby position
        Arm.Arm_serial_servo_write6(180, 140, 5, 20, 90, 80, 2000)
        time.sleep(5)

        # Drop action
        Arm.Arm_serial_servo_write6(180, 140, 5, 20, 90, 30, 2000)
        time.sleep(5)

except FileNotFoundError:
    print("Pixel coordinates file not found. Run 1217_camera.py first.")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Program Finished.")
