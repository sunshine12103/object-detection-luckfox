#!/usr/bin/python3
"""
HC-SR04 Test - Adapted from RPi.GPIO style to Luckfox
======================================================
Based on RPi.GPIO code but using direct sysfs GPIO access

GPIO 17: TRIGGER
GPIO 22: ECHO
"""

import time
import os
import sys

# GPIO Configuration (Physical Pin Mapping)
# Physical Pin 11 → GPIO3_A2_d → GPIO 98
# Physical Pin 15 → GPIO0_B5_d → GPIO 13
GPIO_TRIGGER = 98  # Physical Pin 11
GPIO_ECHO = 13     # Physical Pin 15
GPIO_PATH = "/sys/class/gpio"

class SimpleGPIO:
    """Simple GPIO wrapper mimicking RPi.GPIO style"""
    
    OUT = "out"
    IN = "in"
    
    @staticmethod
    def setup(pin, direction):
        """Setup GPIO pin"""
        pin_path = f"{GPIO_PATH}/gpio{pin}"
        
        # Export if needed
        if not os.path.exists(pin_path):
            with open(f"{GPIO_PATH}/export", 'w') as f:
                f.write(str(pin))
            time.sleep(0.1)
        
        # Set direction
        with open(f"{pin_path}/direction", 'w') as f:
            f.write(direction)
    
    @staticmethod
    def output(pin, value):
        """Set GPIO output value"""
        pin_path = f"{GPIO_PATH}/gpio{pin}/value"
        with open(pin_path, 'w') as f:
            f.write("1" if value else "0")
    
    @staticmethod
    def input(pin):
        """Read GPIO input value"""
        pin_path = f"{GPIO_PATH}/gpio{pin}/value"
        with open(pin_path, 'r') as f:
            return f.read().strip() == "1"
    
    @staticmethod
    def cleanup():
        """Cleanup GPIO"""
        for pin in [GPIO_TRIGGER, GPIO_ECHO]:
            try:
                pin_path = f"{GPIO_PATH}/gpio{pin}"
                if os.path.exists(pin_path):
                    with open(f"{GPIO_PATH}/unexport", 'w') as f:
                        f.write(str(pin))
            except:
                pass

# Create GPIO instance
GPIO = SimpleGPIO()

def measure_distance():
    """Measure distance using HC-SR04"""
    
    # Kích hoạt cảm biến bằng cách ta nháy cho nó tí điện rồi ngắt đi luôn
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)  # 10µs pulse
    GPIO.output(GPIO_TRIGGER, False)
    
    # Đánh dấu thời điểm bắt đầu
    start = time.time()
    timeout = start + 0.5  # 500ms timeout
    
    while GPIO.input(GPIO_ECHO) == 0:
        start = time.time()
        if start > timeout:
            return None  # Timeout
    
    # Bắt thời điểm nhận được tín hiệu từ Echo
    stop = time.time()
    timeout = stop + 0.5
    
    while GPIO.input(GPIO_ECHO) == 1:
        stop = time.time()
        if stop > timeout:
            return None  # Timeout
    
    # Thời gian từ lúc gửi tín hiệu
    elapsed = stop - start
    
    # Distance pulse travelled in that time is time
    # multiplied by the speed of sound (cm/s)
    distance = elapsed * 34000
    
    # That was the distance there and back so halve the value
    distance = distance / 2
    
    return distance

# ============== MAIN ==============
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ Need root! Run: sudo python3 test_hcsr04_rpi_style.py")
        sys.exit(1)
    
    print("=" * 50)
    print("Ultrasonic Measurement (RPi.GPIO style)")
    print("=" * 50)
    
    try:
        # Thiết lập GPIO nào để gửi tín hiệu và nhận tín hiệu
        GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Trigger
        GPIO.setup(GPIO_ECHO, GPIO.IN)      # Echo
        
        print("✅ GPIO setup complete")
        print(f"   TRIGGER: GPIO {GPIO_TRIGGER}")
        print(f"   ECHO: GPIO {GPIO_ECHO}")
        
        # Khai báo này ám chỉ việc hiện tại không gửi tín hiệu điện
        # qua GPIO này, kiểu kiểu ngắt điện ấy
        GPIO.output(GPIO_TRIGGER, False)
        
        # Cái này mình cũng không rõ, nhưng họ bảo là để khởi động cảm biến
        print("\n⏳ Initializing sensor...")
        time.sleep(0.5)
        
        print("\n📏 Starting measurements (Press Ctrl+C to stop)...\n")
        
        while True:
            distance = measure_distance()
            
            if distance is not None and 2 <= distance <= 400:
                print(f"Distance: {distance:6.1f} cm ({distance/100:5.3f} m)")
            else:
                print("Out of range or timeout")
            
            time.sleep(0.5)  # Measure every 500ms
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        # Reset GPIO settings
        GPIO.cleanup()
        print("✅ GPIO cleanup complete")
