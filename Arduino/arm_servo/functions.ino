




void print_function(){
  Serial.println("Hello");
}



/*
 * angleToPulse(int ang)
 * gets angle in degree and returns the pulse width
 * also prints the value on seial monitor
 * written by Ahmad Nejrabi for Robojax, Robojax.com
 */
int angleToPulse(int ang){
   int pulse = map(ang,0, 180, 100,600);// map angle of 0 to 180 to Servo min and Servo max 
   Serial.print("Angle: ");Serial.print(ang);
   Serial.print(" pulse: ");Serial.println(pulse);
   return pulse;
}




// You can use this function if you'd like to set the pulse length in seconds
// e.g. setServoPulse(0, 0.001) is a ~1 millisecond pulse width. It's not precise!
void setServoPulse(uint8_t n, double pulse) {
  double pulselength;
  
  pulselength = 1000000;   // 1,000,000 us per second
  pulselength /= SERVO_FREQ;   // Analog servos run at ~60 Hz updates
  Serial.print(pulselength); Serial.println(" us per period"); 
  pulselength /= 4096;  // 12 bits of resolution
  Serial.print(pulselength); Serial.println(" us per bit"); 
  pulse *= 1000000;  // convert input seconds to us
  pulse /= pulselength;
  Serial.println(pulse);
  pwm.setPWM(n, 0, pulse);
}


float servo_angletopulse(int servo_num,float servo_angle){
  
  float servo_position;
  if(servo_num == 5) // gripper
    servo_position = map(servo_angle, 10 ,  48 , SERVOMIN[servo_num], SERVOMAX[servo_num]);// map angle of 0 to 180 to Servo min and Servo max 
  else
    servo_position = map(servo_angle+servo_angle_offset[servo_num], 0, ANGLEMAX[servo_num], SERVOMIN[servo_num], SERVOMAX[servo_num]);// map angle of 0 to 180 to Servo min and Servo max 

  // limit values
  if( servo_position <  SERVOMIN[servo_num] ) {
     
     Serial.print(servo_position);
     Serial.print("  Value less than ServoMIN: ");
    Serial.println(servo_num);
    servo_position = SERVOMIN[servo_num];
  }
  else if( servo_position > SERVOMAX[servo_num] )
  {     
        Serial.print(servo_position);
        Serial.print("  Value exceeded ServoMAX: ");
        Serial.println(servo_num);
        servo_position = SERVOMAX[servo_num];
  }
      
    
  return servo_position;
}

/*
 * servo_speed = 1-30  //default  = 10
 */
void move_servo_speed(int servo_num,int servo_angle){

  //recallibrates
  if(servo_num <5)
     servo_angle = map(servo_angle,  90, -90, 0, 180); 


  int servo_speed = 5 ;
  int servo_position = servo_angletopulse(servo_num,servo_angle);
    
    if(servo_position> val_prev[servo_num]){
      for( val_curr[servo_num] = val_prev[servo_num]; val_curr[servo_num]<=servo_position; val_curr[servo_num] +=3){
                pwm.setPWM(servo_num, 0, val_curr[servo_num] );
                delay(30-servo_speed);
          }
    }
    else{
          for( val_curr[servo_num] = val_prev[servo_num]; val_curr[servo_num]>=servo_position; val_curr[servo_num] -=3){
                pwm.setPWM(servo_num, 0, val_curr[servo_num] );
                delay(30-servo_speed);
          }
    }
    
        val_prev[servo_num] = val_curr[servo_num];
}




void move_servo_home(void){
    int speed_ser = 5;
    move_servo_speed(0,90);
    delay(1000);
    
    move_servo_speed(1,110);
    move_servo_speed(2,40);
    move_servo_speed(3,130);
    move_servo_speed(4,90);
    move_servo_gripper(30);
    delay(3000);
    Serial.println("Home achieved");

}




// THis is two position working code
void Test_motion_00(void){

    int speed_ser =15; //15: medium ; 5: slow  ; 30 max
    move_servo_speed(0,120,speed_ser);
    delay(2000);
    
    move_servo_speed(1,30);
    move_servo_speed(2,30);
    move_servo_speed(3,30);
    move_servo_speed(4,30);
    delay(4000);
    
    move_servo_speed(0,20);
    delay(2000);
    
    move_servo_speed(1,80);
    move_servo_speed(2,80);
    move_servo_speed(3,80);
    move_servo_speed(4,80);
    delay(4000);

}

void Test_motion_01(void){

  // code 4
/*
pwm.setPWM(0, 0, 220);
delay(2000);

pwm.setPWM(1, 0, 240);
pwm.setPWM(2, 0, 180);
pwm.setPWM(3, 0, 220);
pwm.setPWM(4, 0, 300);
delay(4000);

pwm.setPWM(0, 0, 320);
delay(2000);

pwm.setPWM(1, 0, 300);
pwm.setPWM(2, 0, 250);
pwm.setPWM(3, 0, 150);
pwm.setPWM(4, 0, 180);
delay(4000);

*/






}


void Test_motion_02(void){
      //code 2
      /*
        Serial.print("Enter the Motor Number:  ");
        while (Serial.available()==0){}
        mot_num_str = Serial.readString();
        mot_num = mot_num_str.toInt();
        Serial.println(mot_num);
      
        Serial.print("Enter the Pulse value:  ");
        while (Serial.available()==0){}
        pulse_val_str = Serial.readString();
        pulse_val = pulse_val_str.toInt();
        Serial.println(pulse_val);  
        Serial.print("\n");
        pwm.setPWM(mot_num, 0, pulse_val);
          
       delay(2000);
      
      
      
        // code 3
        //
      //  Serial.print("Enter the Motor Number:  ");
      //  val_str = Serial.readString();
      //  val = val_str.toInt();
      //  
      //  Serial.println(val);
      //    pwm.setPWM(2, 0, val);
      //  
      */
      
      
      
      /*
          for( int angle =0; angle<=180; angle +=10){
            for(int i=0; i<16; i++)
              {      
                  pwm.setPWM(i, 0, angleToPulse(angle) );
              }
          }
      
      */
      
      // pwm.setPWM(2, 0, angleToPulse(20) );
      // delay(2000);
      // pwm.setPWM(2, 0, angleToPulse(90) );
      // delay(2000);
      // pwm.setPWM(2, 0, angleToPulse(160) );
      // delay(2000);
      //  pwm.setPWM(2, 0, angleToPulse(90) );
      // delay(2000);




}


void servo_calibrate(int servo_num){

  int val = 0;
  String val_str;
   //snippet to calibrate individual motor from serial monitor
  Serial.print("Enter the Value:  ");
  
  while (Serial.available()==0){}
  val_str = Serial.readString();
  val = val_str.toInt();


  //Servo remap task
   // val = map(val,  90, -90, 0, 180);  //servo 0

    //val = map(val,  90, -90, 0, 180);   // servo 0
  Serial.println(val);
    if(servo_num <5)
        val = map(val,  90, -90, 0, 180); 
  

  
  
  
    move_servo_speed(servo_num ,val ); //servo_speed = 5

}


void servo_calibrate_gripper(){
  int value = read_Serialterminal();
  move_servo_gripper(value);
}


void servo_calibrate_raw(int servo_num){
     //snippet to calibrate individual motor from serial monitor
  Serial.print("Enter the Value:  ");
  
  while (Serial.available()==0){}
  String  val_str = Serial.readString();
  int val = val_str.toInt();
  
    Serial.println(val);
    //move_servo_speed(servo_num ,val ); //servo_speed = 5
       pwm.setPWM(servo_num, 0, val);

}




void move_servo_gripper(float gripper_width_mm){

  int servo_num = 5;
  int servo_position = servo_angletopulse(servo_num,gripper_width_mm);
  

  
  if(servo_position> val_prev[servo_num]){
    for( val_curr[servo_num] = val_prev[servo_num]; val_curr[servo_num]<=servo_position; val_curr[servo_num] +=2){
              pwm.setPWM(servo_num, 0, val_curr[servo_num] );
              delay(10);
        }
  }
  else{
        for( val_curr[servo_num] = val_prev[servo_num]; val_curr[servo_num]>=servo_position; val_curr[servo_num] -=2){
              pwm.setPWM(servo_num, 0, val_curr[servo_num] );
              delay(10);
        }
  }
  
      val_prev[servo_num] = val_curr[servo_num];


}


float read_Serialterminal(void){
    
  Serial.print("Enter the Value:  ");
  while (Serial.available()==0){}
  String val_str = Serial.readString();
  float val = val_str.toFloat();
  
    Serial.println(val);

    return val;
  
}
