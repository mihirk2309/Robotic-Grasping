import serial
import time

arduino = serial.Serial('/dev/ttyUSB0',9600)
arduino.timeout=1


def write_data(x):
    print("This value will be sent: " , bytes(x))
    arduino.write(bytes(x))

def read_data():
    #time.sleep(0.05)
    x = arduino.readline() 
    return x


arr = [1.9256326537,2.82686234,3.826387123,4.23,5.2]

#while True:
# num1 = input("Enter the number 1: ")
# num2 = input("Enter the number 2: ")
# num3 = input("Enter the number 3: ")
while True:
    if(input("Enter 1: ") ):
        for i in arr:
            write_data(i)
            write_data(' ')

# write_data(num2)
# write_data(num3)
print("Value from arduino:",read_data())


