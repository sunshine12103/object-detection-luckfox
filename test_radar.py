import serial
import time

SERIAL_PORT = "/dev/ttyS2"
BAUDRATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    print(f"✅ Radar connected: {SERIAL_PORT}")
    print("Stand in front of radar...")
    
    for i in range(50):  # Test 5 seconds
        data_all = []
        while ser.in_waiting > 0:
            byte_data = ser.read()
            hex_str = byte_data.hex()
            data_all.append(hex_str)
        
        if len(data_all) >= 7:
            # Heart rate
            if (int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 85) and \
               (int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2):
                heart = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                print(f"❤️ Heart: {heart} bpm")
            
            # Breath rate
            elif (int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 81) and \
                 (int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2):
                breath = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                print(f"🌬️ Breath: {breath} /min")
        
        time.sleep(0.1)
    
    ser.close()
except Exception as e:
    print(f"❌ Error: {e}")
