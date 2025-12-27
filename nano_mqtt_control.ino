/*
 * Arduino Nano - Nhận lệnh từ ESP32 qua UART
 * Modified từ code gốc - SoftwareSerial pin 2, 3
 */
#include <Arduino.h>
#include <SoftwareSerial.h>

#define RX_PIN 2
#define TX_PIN 3
SoftwareSerial Serial2(RX_PIN, TX_PIN);

#define TIEM_CAN_SAU_PIN A3
#define TIEM_CAN_TRAI_PIN A6
#define TIEM_CAN_PHAI_PIN A7

#define TRIG_PIN A1
#define ECHO_PIN A2

#define ENA_BANH_XE_TRAI_TREN   9   
#define IN1_BANH_XE_TRAI_TREN   7  
#define IN2_BANH_XE_TRAI_TREN   8  
#define IN3_BANH_XE_TRAI_DUOI   11    
#define IN4_BANH_XE_TRAI_DUOI   12    
#define ENB_BANH_XE_TRAI_DUOI   10
#define ENA_BANH_XE_PHAI_TREN   5
#define IN1_BANH_XE_PHAI_TREN   A0
#define IN2_BANH_XE_PHAI_TREN   4
#define IN3_BANH_XE_PHAI_DUOI   A5
#define IN4_BANH_XE_PHAI_DUOI   A4
#define ENB_BANH_XE_PHAI_DUOI   6

#define MOTOR_SPEED 70
#define TURN_SPEED_SLOW 35   // Quay chậm (arc) - lệch ít
#define TURN_SPEED_FAST 0    // Quay nhanh (spin) - lệch nhiều

int currentMovement = 0;

long readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH);
  long distance = duration * 0.034 / 2;
  return distance;
}

void stopAllMotors() {
  analogWrite(ENA_BANH_XE_TRAI_TREN, 0);
  analogWrite(ENB_BANH_XE_TRAI_DUOI, 0);
  analogWrite(ENA_BANH_XE_PHAI_TREN, 0);
  analogWrite(ENB_BANH_XE_PHAI_DUOI, 0);
  
  digitalWrite(IN1_BANH_XE_TRAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, LOW);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, LOW);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, LOW);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, LOW);
}

void moveForward() {
  analogWrite(ENA_BANH_XE_TRAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, LOW);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, LOW);
}

void moveBackward() {
  analogWrite(ENA_BANH_XE_TRAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, HIGH);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, HIGH);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, HIGH);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, HIGH);
}

void turnRightSlow() {
  // Quay phải chậm (arc turn) - dùng khi lệch ít
  // Trái nhanh, phải chậm
  analogWrite(ENA_BANH_XE_TRAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, LOW);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, TURN_SPEED_SLOW);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, TURN_SPEED_SLOW);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, LOW);
}

void turnRightFast() {
  // Quay phải nhanh (spin in place) - dùng khi lệch nhiều
  // Trái tiến, phải lùi
  analogWrite(ENA_BANH_XE_TRAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, LOW);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, HIGH);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, HIGH);
}

void turnLeftSlow() {
  // Quay trái chậm (arc turn)
  // Phải nhanh, trái chậm
  analogWrite(ENA_BANH_XE_TRAI_TREN, TURN_SPEED_SLOW);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, TURN_SPEED_SLOW);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, LOW);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, LOW);
}

void turnLeftFast() {
  // Quay trái nhanh (spin in place)
  // Phải tiến, trái lùi
  analogWrite(ENA_BANH_XE_TRAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_TRAI_TREN, LOW);
  digitalWrite(IN2_BANH_XE_TRAI_TREN, HIGH);
  
  analogWrite(ENB_BANH_XE_TRAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_TRAI_DUOI, LOW);
  digitalWrite(IN4_BANH_XE_TRAI_DUOI, HIGH);
  
  analogWrite(ENA_BANH_XE_PHAI_TREN, MOTOR_SPEED);
  digitalWrite(IN1_BANH_XE_PHAI_TREN, HIGH);
  digitalWrite(IN2_BANH_XE_PHAI_TREN, LOW);
  
  analogWrite(ENB_BANH_XE_PHAI_DUOI, MOTOR_SPEED);
  digitalWrite(IN3_BANH_XE_PHAI_DUOI, HIGH);
  digitalWrite(IN4_BANH_XE_PHAI_DUOI, LOW);
}

void turnRight() {
  turnRightSlow();  // Deprecated - dùng slow
}

void turnLeft() {
  turnLeftSlow();  // Deprecated - dùng slow
}

void setMovement(int newMovement) {
  if (currentMovement == newMovement) return; 
  
  currentMovement = newMovement;
  
  switch (newMovement) {
    case 1: // Forward
      moveForward();
      Serial.println("FORWARD");
      break;
    case 2: // Backward
      moveBackward();
      Serial.println("BACKWARD");
      break;
    case 3: // Turn Left Slow
      turnLeftSlow();
      Serial.println("TURN LEFT SLOW");
      break;
    case 4: // Turn Right Slow
      turnRightSlow();
      Serial.println("TURN RIGHT SLOW");
      break;
    case 5: // Turn Left Fast
      turnLeftFast();
      Serial.println("TURN LEFT FAST");
      break;
    case 6: // Turn Right Fast
      turnRightFast();
      Serial.println("TURN RIGHT FAST");
      break;
    default: // Stop
      stopAllMotors();
      Serial.println("STOP");
      currentMovement = 0;
      break;
  }
}

void setup() {
  Serial.begin(115200);  // Serial monitor debug
  Serial2.begin(9600);   // SoftwareSerial từ ESP32 (pin 2, 3)
  
  pinMode(TIEM_CAN_SAU_PIN, INPUT);
  pinMode(TIEM_CAN_TRAI_PIN, INPUT);
  pinMode(TIEM_CAN_PHAI_PIN, INPUT);
  
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  pinMode(ENA_BANH_XE_TRAI_TREN, OUTPUT);
  pinMode(IN1_BANH_XE_TRAI_TREN, OUTPUT);
  pinMode(IN2_BANH_XE_TRAI_TREN, OUTPUT);
  pinMode(IN3_BANH_XE_TRAI_DUOI, OUTPUT);
  pinMode(IN4_BANH_XE_TRAI_DUOI, OUTPUT);
  pinMode(ENB_BANH_XE_TRAI_DUOI, OUTPUT);
  pinMode(ENA_BANH_XE_PHAI_TREN, OUTPUT);
  pinMode(IN1_BANH_XE_PHAI_TREN, OUTPUT);
  pinMode(IN2_BANH_XE_PHAI_TREN, OUTPUT);
  pinMode(IN3_BANH_XE_PHAI_DUOI, OUTPUT);
  pinMode(IN4_BANH_XE_PHAI_DUOI, OUTPUT);
  pinMode(ENB_BANH_XE_PHAI_DUOI, OUTPUT);
  
  stopAllMotors();
  
  Serial.println("=== AGV MECANUM WHEEL - MQTT MODE ===");
  Serial.println("Waiting for commands from ESP32...");
}

void loop() {
  // Nhận lệnh từ ESP32 qua SoftwareSerial
  if (Serial2.available() > 0) {
    String input = Serial2.readStringUntil('\n');
    int command = input.toInt();
    
    Serial.print("Received command: ");
    Serial.println(command);
    
    setMovement(command);
  }
}
