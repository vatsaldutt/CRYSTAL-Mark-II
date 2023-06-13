import serial
from time import sleep

# ArduinoData = serial.Serial('/dev/cu.usbmodem14601', 9600)
try:
    ArduinoData = serial.Serial('/dev/cu.usbserial-1420', 9600)

    def claw_open():
        sleep(3)
        ArduinoData.write(b'2')

    def claw_close():
        sleep(3)
        ArduinoData.write(b'1')

    def forward(speed, distance):
        ArduinoData.write(b'3')

    def backward(speed, distance):
        ArduinoData.write(b'4')

    def stop_car():
        sleep(3)
        ArduinoData.write(b'0')

    def wrist_up():
        sleep(3)
        ArduinoData.write(b'5')

    def wrist_down():
        sleep(3)
        ArduinoData.write(b'6')

    def elbow_up():
        sleep(3)
        ArduinoData.write(b'7')
    def elbow_down():
        sleep(3)
        ArduinoData.write(b'8')

    def shoulder_up():
        sleep(3)
        ArduinoData.write(b'9')
    def shoulder_down():
        sleep(3)
        ArduinoData.write(b'10')

    def shoulder_middle_up():
        sleep(3)
        ArduinoData.write(b'11')
        
    def shoulder_middle():
        sleep(3)
        ArduinoData.write(b'12')

    def shoulder_middle_down():
        sleep(3)
        ArduinoData.write(b'13')

    def elbow_middle_up():
        sleep(3)
        ArduinoData.write(b'14')

    def elbow_middle():
        sleep(3)
        ArduinoData.write(b'15')

    def elbow_middle_down():
        sleep(3)
        ArduinoData.write(b'16')

    def wrist_middle_up():
        sleep(3)
        ArduinoData.write(b'17')

    def wrist_middle():
        sleep(3)
        ArduinoData.write(b'18')

    def wrist_middle_down():
        sleep(3)
        ArduinoData.write(b'19')

    def initialize_robot():
        sleep(3)
        elbow_middle_up()
        sleep(2)
        shoulder_middle_up()
        sleep(2)
        wrist_middle_down()
        sleep(2)
        claw_open()

    def health_check():
        sleep(3)
        claw_close()
        sleep(1)
        claw_open()
        sleep(1)
        wrist_up()
        sleep(1)
        wrist_middle_down()
        sleep(1)
        shoulder_up()
        sleep(1.5)
        elbow_middle_down()
        sleep(1.5)
        elbow_middle_up()
        sleep(1.5)
        shoulder_middle_up()
        sleep(10)

except:
    print("ARDUINO PORT NOT CONFIGURED")