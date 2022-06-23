#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

/* My function definitions*/
void setServoPulse(uint8_t n, double pulse);
void move_servo(int num, int speed_servo);
int angleToPulse(int ang);
void servo_calibrate(void);
void move_servo_speed(int servo_num,int servo_angle, int servo_speed);
void move_servo_home();
void move_servo_gripper(float gripper_width_mm);
float read_Serialterminal(void);

void arm_move(float angle_1, float angle_2, float angle_3, float angle_4, float angle_5, float gripper_width_mm);

void get_IK_solution(float angle[]);




//#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
//#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates




// our defines
uint16_t SERVOMIN[6] = { 90, 140,  150, 150, 90, 250 };
uint16_t SERVOMAX[6] = {440, 450, 350, 350, 450,  450 };
uint16_t ANGLEMAX[6] = {180, 170, 116, 113, 180, 180 }; // last gripper 180 not used

// otor- 4 (370 - 90) 

//waist rotation
#define SERVO_1   0
//shoulder elevation
#define SERVO_2   1c
//elbow elevation
#define SERVO_3   2
//wrist elevation
#define SERVO_4   3
//wrist rotation
#define SERVO_5   4
//gripper open
#define SERVO_6   5


int incomingByte = 0;
int choice = 0;
float sol_angle[6];
float temp[6];
float angle_values[20][6];
float final_values[20][6];
int no_of_points = 20;
int flag = 0;
int i = 0;



///////////////////////////////////////////////////////////////////

// our servo # counter
uint8_t servonum = 0;


void setup() {
  Serial.begin(9600);
  //Serial.println("8 channel Servo test!");

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

//  delay(10);
//  get_IK_solution(sol_angle);
//  angle_values[0][0] = sol_angle[0];
//  angle_values[0][1] = sol_angle[1];
//  angle_values[0][2] = sol_angle[2];
//  angle_values[0][3] = sol_angle[3];
//  angle_values[0][4] = sol_angle[4];
//  
//  arm_move(sol_angle[0],sol_angle[1],sol_angle[2],sol_angle[3],sol_angle[4], 30); //last is gripper  
    
    //move_servo_home();   // move servo to our defined home.(PREVIOUS)
    //arm_move(0,110,160,130,90,30);   // home
   
}




#define SERVO_MAX_SPEED 10
int angle =180;
int mot_num=0;
int pulse_val = 0;

String mot_num_str, pulse_val_str;
int val_curr[6];
int val_prev[6] = {320,320,220,190,200,350};  // initial values
int speed_ser =15; //15: medium ; 5: slow  ; 30 max
int servo_angle_offset[6] =  { 25 , -8, -26, -25, 0, 0}; //115-90 ( calib angle - orignal angle )



      /*
       // Servo Motor Parameters, Refernces
       // int orig_angle[6] =  { 90, 90 , 90, 90, 90};
       // int calib_angle[6] = { 115 , 77, 48, 50, 75};  // angles at 90 degree
       //gripper max x =450 (servo pwm) ,  48 mm
       //gripper min x= 250  (servo pwm) , 10 mm
      
      */
      
         //uint16_t ANGLEMAX[6] = {180, 170, 116, 113, 103, 180 };   
         // Test_motion_00();
         //home = {90,110,120,130}




void loop(){

//move_servo_gripper(20);

  get_IK_solution(sol_angle);
  
  Serial.println(sol_angle[0]);   // if you multiply by 1000; we get decimal accuracy
  Serial.println(sol_angle[1]); 
  Serial.println(sol_angle[2]);
  Serial.println(sol_angle[3]);
  Serial.println(sol_angle[4]);
  Serial.println(sol_angle[5]);
  Serial.println("Next");
  
  arm_move(sol_angle[0],sol_angle[1],sol_angle[2],sol_angle[3],sol_angle[4]-90, sol_angle[5]); //last is gripper    




  

/*
        Serial.print("In loop-");
    
        temp[0] = sol_angle[0];
        temp[1] = sol_angle[1];
        temp[2] = sol_angle[2];
        temp[3] = sol_angle[3];
        temp[4] = sol_angle[4];

        angle_values[0][0] = sol_angle[0];
        angle_values[0][1] = sol_angle[1];
        angle_values[0][2] = sol_angle[2];
        angle_values[0][3] = sol_angle[3];
        angle_values[0][4] = sol_angle[4];

        
        Serial.println(temp[0]);   // if you multiply by 1000; we get decimal accuracy
        Serial.println(temp[1]); 
        Serial.println(temp[2]);
        Serial.println(temp[3]);
        Serial.println(temp[4]);
        
        delay(1000);
        get_IK_solution(sol_angle);
        Serial.println(sol_angle[0]);   // if you multiply by 1000; we get decimal accuracy
        Serial.println(sol_angle[1]); 
        Serial.println(sol_angle[2]);
        Serial.println(sol_angle[3]);
        Serial.println(sol_angle[4]);
        i = 1;

      while(temp[0]!=sol_angle[0] || temp[1]!=sol_angle[1] || temp[2]!=sol_angle[2] || temp[3]!=sol_angle[3] || temp[4]!=sol_angle[4]){
          angle_values[i][0] = sol_angle[0];
          angle_values[i][1] = sol_angle[1];
          angle_values[i][2] = sol_angle[2];
          angle_values[i][3] = sol_angle[3];
          angle_values[i][4] = sol_angle[4];
          
          i++;
//          Serial.println(angle_values[i][0]);
//          Serial.println(angle_values[i][1]);
//          Serial.println(angle_values[i][2]);
//          Serial.println(angle_values[i][3]);
//          Serial.println(angle_values[i][4]);
          flag = 1;
          Serial.print("In While-");
          temp[0] = sol_angle[0];
          temp[1] = sol_angle[1];
          temp[2] = sol_angle[2];
          temp[3] = sol_angle[3];
          temp[4] = sol_angle[4];
          get_IK_solution(sol_angle);
    }

    
    if(flag==1){
      Serial.print("In if-");
   
      for(int j=0; j<=no_of_points;j++){
          Serial.println(angle_values[j][0]);
          Serial.println(angle_values[j][1]);
          Serial.println(angle_values[j][2]);
          Serial.println(angle_values[j][3]);
          Serial.println(angle_values[j][4]);
        arm_move(angle_values[j][0],angle_values[j][1],angle_values[j][2],angle_values[j][3],angle_values[j][4], 30); //last is gripper  
        delay(300);
        Serial.println("In for2-");
      }
    flag = 0;
    }

*/



  /*
  float value = read_Serialterminal();
  move_servo_speed(4, value);
  */

  
  //float value = read_Serialterminal();
  //  move_servo_gripper(value);
  
  //int out = map(value, 0  , 180  , -90,90);
  //servo_calibrate(4);
  //servo_calibrate_raw(4);



  

}
