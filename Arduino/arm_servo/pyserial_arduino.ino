

void get_IK_solution(float angle[]){

  char temp[40];  
  
  while(!Serial.available());
  String str = Serial.readString();
  //Serial.println(str);   //for debug

   for(int i = 0, j = 0, angle_count = 0 ; str[i]!=NULL ; i++){
        temp[j] = str[i];
        j++;      
        
        if(str[i] == ' '){
           String s = String(temp);
           //Serial.println(s);
           angle[angle_count] = s.toFloat();
           j = 0;
    
           int counter=0;
           while(temp[counter]!='\0')
              {temp[counter] = '\0';
               counter++;
              }
            angle_count++;
          }
  
    }

}
