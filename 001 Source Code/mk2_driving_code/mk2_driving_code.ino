#include <Arduino.h>
#include "ServoEasing.hpp"
#define USE_PCA9685_SERVO_EXPANDER
#define recvCH1 38  // THROTTLE(오른쪽 스틱 상하) 연결 
#define recvCH2 40  // AILE(오른쪽 스틱 좌우) 연결
#define recvCH3 42  // ELEV(왼쪽 스틱 상하) 연결
#define recvCH4 44  // RUDO(왼쪽 스틱 좌우) 연결 
#define recvCH5 46  // S1 switch와 연결(변신모드용)
//#define recvCH6 13 // S3 switch와 연결(변신모드전 위치조절용)
int Servoinput[12] ;
//솔레노이드 액추에이터
int solenoid = 30;
//인터럽트 핀 연결 변수
const int encoderPinA = 18; //엔코더 인터럽트 핀 연결
const int encoderPinB = 19; //엔코더 인터럽트 핀 연결
const int encoderPinC = 20; //엔코더 인터럽트 핀 연결
const int encoderPinD = 21; //엔코더 인터럽트 핀 연결
long encoderPos_R ;
long encoderPos_L ;
const float ratio = 360. / 100. / 22.;
//모드변경확인
//false는 휠모드, true는 walking mode.
boolean current_modetrigger = 0;

#define ACT 4   // Linear Actuator enable signal output pin

//모터 쉴드 변수선언
int E1 = 5; // 모터 1 속도 제어
int M1 = 4; // 모터 1 방향 제어
int E2 = 6; // 모터 2 속도 제어
int M2 = 7; // 모터 2 방향 제어

unsigned long motor_speed; // unsigned long : 양수값만 받음

//수신기의 6개 채널 변수선언
long valueCH1;
long valueCH2;
long valueCH3;
long valueCH4;
long valueCH5;
long valueCH6;

//변신모드 준비 변수
int aim_value = 0;

int control_R;
int control_L;
int Pcontrol;
int a;
int b;
long time_j;


int *g ;
//캘리브레이션 변수
long maxValue = 0;
long minValue = 0;
long sensorMin = 20000;
long sensorMax = 0;
long time_s;

long valueCH_1;
long valueCH2_L;
long valueCH2_R;
long valueCH3_B;
long valueCH3_F;
long valueCH_4;
long valueCH_5;
long valueCH_6;

//서보모터구동위한 구조체,명령어들 선언
ServoEasing ServoFR1(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoFR2(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoFR3(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBR1(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBR2(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBR3(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoFL1(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoFL2(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoFL3(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBL1(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBL2(PCA9685_DEFAULT_ADDRESS, &Wire);
ServoEasing ServoBL3(PCA9685_DEFAULT_ADDRESS, &Wire);
const int Millis = 750;//서보모터가 목표 각도까지 움직이는 게 걸리는 시간
const int SERVO_PIN[12] = {0, 1, 2,
                           4, 5, 6,
                           8, 9, 10,
                           12, 13, 14
                          };
typedef struct quadrobot {
  //각 조인트들 표시(원래 에러값을 나눠서 더하려고 했으나 코드가 너무 길어질거같아 그냥 합쳐서 쓰는게 나을거같음.)
  int FR_1, FR_2, FR_3; //각 조인트값 정의
  int BR_1, BR_2, BR_3;
  int BL_1, BL_2, BL_3;
  int FL_1, FL_2, FL_3;

  int FR_1_initial = 10; // 0 //rolling 시작할때각도
  int FR_2_initial = 170; //180
  int FR_3_initial = 5; //0

  int BR_1_initial = 175; //180
  int BR_2_initial = 180; //180
  int BR_3_initial = 0; //0

  int BL_1_initial = 1; //0
  int BL_2_initial = 0; //0
  int BL_3_initial = 92;

  int FL_1_initial = 179; //180
  int FL_2_initial = 3; //0
  int FL_3_initial = 91;
  //현재 조인트값 정의
  int FR_1_current = FR_1_initial;
  int FR_2_current = FR_2_initial;
  int FR_3_current = FR_3_initial;

  int BR_1_current = BR_1_initial;
  int BR_2_current = BR_2_initial;
  int BR_3_current = BR_3_initial;

  int BL_1_current = BL_1_initial;
  int BL_2_current = BL_2_initial;
  int BL_3_current = BL_3_initial;

  int FL_1_current = FL_1_initial;
  int FL_2_current = FL_2_initial;
  int FL_3_current = FL_3_initial;

  //추가될 값 정의
  int (*FR_1_angle) = 0;
  int *FR_2_angle = 0;
  int *FR_3_angle = 0;

  int *BR_1_angle = 0;
  int *BR_2_angle = 0;
  int *BR_3_angle = 0;

  int *BL_1_angle = 0;
  int *BL_2_angle = 0;
  int *BL_3_angle = 0;

  int *FL_1_angle = 0;
  int *FL_2_angle = 0;
  int *FL_3_angle = 0;
  // 0으로 변환

  // 스텝간 움직인 수를 체크하기 위함
  int count = 0;
  //전체적인 모드에 관한 조인트 각도 선언
  int leg_up = 30; // 다리를 들어올리는 각도 정의
} quadrobot;

quadrobot Joint;



//펄스값 읽는 인터럽트 함수
void doencoderPinA() {

  encoderPos_R += (digitalRead(encoderPinA) == digitalRead(encoderPinB)) ? -1 : 1;

}
void doencoderPinB() {
  encoderPos_R += (digitalRead(encoderPinA) == digitalRead(encoderPinB)) ? 1 : -1;

}
void doencoderPinC() {

  encoderPos_L += (digitalRead(encoderPinC) == digitalRead(encoderPinD)) ? -1 : 1;

}
void doencoderPinD() {
  encoderPos_L += (digitalRead(encoderPinC) == digitalRead(encoderPinD)) ? 1 : -1;

}

int anglecontrol() {
  int error_R = aim_value - (encoderPos_R % 1404);
  int control_R = Pcontrol * error_R;
  int error_L = aim_value - (encoderPos_L % 1404);
  int control_L = Pcontrol * error_L;

  if (error_R != 0) {

    if (error_R > 0) {
      while (1) {
        //Serial.println(encoderPos);
        digitalWrite(M1, 0);
        analogWrite(E1, min(255, 50 + control_R));
        if ( encoderPos_R % 1404 < 0) {

          analogWrite(E1, 0);

          break;
        }
      }
    }
    else if (error_R < 0) {
      while (1) {
        //Serial.println(encoderPos);
        digitalWrite(M1, 1);
        analogWrite(E1, min(255, 50 + control_R));
        if ( encoderPos_R % 1404 > 0) {

          analogWrite(E1, 0);

          break;
        }
      }
    }
    if (error_L != 0) {

      if (error_L > 0) {
        while (1) {
          //Serial.println(encoderPos);
          digitalWrite(M2, 1);
          analogWrite(E2, min(255, 50 + control_L));
          if ( encoderPos_L % 1404 < 0) {

            analogWrite(E2, 0);

            break;
          }
        }
      }
      else if (error_L < 0) {
        while (1) {
          //Serial.println(encoderPos);
          digitalWrite(M2, 0);
          analogWrite(E2, min(255, 50 + control_L));
          if ( encoderPos_L % 1404 > 0) {

            analogWrite(E2, 0);

            break;
          }
        }
      }
    }
    return a = aim_value - (encoderPos_R % 1404) ;
    return b = aim_value - (encoderPos_L % 1404) ;
  }
}
//------------------------------------------------------setup-------------------------
void setup() {
  //모드확인
  current_modetrigger = 0;
  //솔레노이드 선언
  pinMode(solenoid, OUTPUT);
  //처음은 고정(low가 고정,high가 고정해제)
  digitalWrite(solenoid, LOW);
  //서보모터관련 요소들
  //  ServoFR1.attach(SERVO_PIN[0], Joint.FR_1_initial);
  //  ServoFR2.attach(SERVO_PIN[1], Joint.FR_2_initial);
  //  ServoFR3.attach(SERVO_PIN[2], Joint.FR_3_initial);
  //  ServoBR1.attach(SERVO_PIN[3], Joint.BR_1_initial);
  //  ServoBR2.attach(SERVO_PIN[4], Joint.BR_2_initial);
  //  ServoBR3.attach(SERVO_PIN[5], Joint.BR_3_initial);
  //  ServoFL1.attach(SERVO_PIN[6], Joint.BL_1_initial);
  //  ServoFL2.attach(SERVO_PIN[7], Joint.BL_2_initial);
  //  ServoFL3.attach(SERVO_PIN[8], Joint.BL_3_initial);
  //  ServoBL1.attach(SERVO_PIN[9], Joint.FL_1_initial);
  //  ServoBL2.attach(SERVO_PIN[10], Joint.FL_2_initial);
  //  ServoBL3.attach(SERVO_PIN[11], Joint.FL_3_initial);
  int zero = 0;// 변수주소로 넣어줘야 잘 구동됨.
  Joint.FR_1_angle = &zero;
  Joint.FR_2_angle = &zero;
  Joint.FR_3_angle = &zero;

  Joint.BR_1_angle = &zero;
  Joint.BR_2_angle = &zero;
  Joint.BR_3_angle = &zero;

  Joint.BL_1_angle = &zero;
  Joint.BL_2_angle = &zero;
  Joint.BL_3_angle = &zero;

  Joint.FL_1_angle = &zero;
  Joint.FL_2_angle = &zero;
  Joint.FL_3_angle = &zero;
  Serial.begin(9600);

  delay(1000);
  if (pulseIn(recvCH1, HIGH) >= 1700) {
    Serial.println(" ");
    Serial.println("스로틀을 최대로 올리세요");
    Serial.println("A");
    while (millis() < 5000) {

      maxValue = analogRead(valueCH1 = pulseIn(recvCH1, HIGH));
      Serial.println("B");
      if (valueCH1 > sensorMax) {
        sensorMax = valueCH1;
      }
    }
    Serial.println("스로틀을 최대로 내리세요");
    Serial.println("C");
    delay(2000);
    Serial.println("D");
    while (millis() < 10000) {

      minValue = analogRead(valueCH1 = pulseIn(recvCH1, HIGH));
      Serial.println("E");
      if (valueCH1 < sensorMin) {
        sensorMin = valueCH1;
      }

    }

    Serial.println("F");
  }
  else if (pulseIn(recvCH1, HIGH) < 1700) {
    Serial.println("쓰로틀을 최대로 올리고 다시 실행하세요.");
  }

  pinMode(recvCH1, INPUT);
  pinMode(recvCH2, INPUT);
  pinMode(recvCH3, INPUT);
  pinMode(recvCH4, INPUT);
  pinMode(recvCH5, INPUT_PULLUP);//모드변경이라 풀업해줌
  //pinMode(recvCH6, INPUT);
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);

  pinMode(encoderPinA, INPUT_PULLUP);
  attachInterrupt(5, doencoderPinA, CHANGE);

  pinMode(encoderPinB, INPUT_PULLUP);
  attachInterrupt(4, doencoderPinB, CHANGE);

  pinMode(encoderPinC, INPUT_PULLUP);
  attachInterrupt(3, doencoderPinC, CHANGE);

  pinMode(encoderPinD, INPUT_PULLUP);
  attachInterrupt(2, doencoderPinD, CHANGE);
}


//----------------------------------------loop--------------loop---------loop----------
void loop() {
  //컨트롤러 데이터 수신
  Serial.print(encoderPos_L);
  Serial.print("/");
  Serial.println(encoderPos_R);
  long error = encoderPos_L - encoderPos_R;
  long control = error * Pcontrol;
  Serial.println(error);
  noInterrupts(); //인터럽트에 영향을 주지 않는 명령어들
  //Serial.print(encoderPos_R);
  //Serial.print("/");
  //Serial.print(encoderPos_L);
  //Serial.println("/");
  // HIGH일 때 duration의 길이를 정수값으로 변환
  valueCH1 = pulseIn(recvCH1, HIGH); // 1090~1880까지 연속적으로 변화
  valueCH2 = pulseIn(recvCH2, HIGH); // 1090~1880까지 연속적으로 변화
  valueCH3 = pulseIn(recvCH3, HIGH); // 1090~1880까지 연속적으로 변화
  valueCH4 = pulseIn(recvCH4, HIGH); // 1090~1880까지 연속적으로 변화
  valueCH5 = pulseIn(recvCH5, HIGH); // 1090, 1880 두가지 값으로 변화
  //valueCH6 = pulseIn(recvCH6, HIGH); // 1090, 1880 두가지 값으로 변화
  //valueCH1 = constrain(map(valueCH1, sensorMin, sensorMax, 0, 255),0,255);
  valueCH2_L = constrain(map(valueCH2, (sensorMin + sensorMax) / 2, sensorMax, 0, 255), 0, 255);
  valueCH2_R = constrain(map(valueCH2, sensorMin, (sensorMin + sensorMax) / 2, 255, 0), 0, 255);
  valueCH3_F = constrain(map(valueCH3, (sensorMin + sensorMax) / 2, sensorMax, 0, 255), 0, 255);
  valueCH3_B = constrain(map(valueCH3, sensorMin, (sensorMin + sensorMax) / 2, 255, 0), 0, 255);
  valueCH4 = constrain(map(valueCH4, sensorMin, (sensorMin + sensorMax) / 2, 0, 1), 0, 1);
  valueCH5 = constrain(map(valueCH5, sensorMin, (sensorMin + sensorMax) / 2, 0, 1), 0, 1);
  interrupts();
  if (valueCH5 != current_modetrigger) {
    if (valueCH5 == 0) {
      //보행 -> 휠
      digitalWrite(solenoid, HIGH);
    }
    else if (valueCH5 == 1) {
      digitalWrite(solenoid, LOW);
      
      Servoinput[1] = 170 - 15; //FR_2
      Servoinput[4] = 163 - 15; //BR_2
      Servoinput[7] = 3 + 15; //FL_2
      Servoinput[10] = 3 + 15; //BL_2
      allservo();

      Servoinput[4] = 163 - 135; //BR_2
      Servoinput[10] = 3 + 135; //BL_2
      Servoinput[5] = 18 + 20; //BR_3
      Servoinput[11] = 22 - 20; //BL_3
      Servoinput[1] = 170 - 165; //FR_2
      Servoinput[7] = 3 + 165; //FL_2
      Servoinput[2] = 2 + 90; //FR_3
      Servoinput[8] = 93 - 90; //FL_3
      allservo();

      Servoinput[4] = 163 - 180; //BR_2
      allservo();

      Servoinput[3] = 160 - 45; //BR_1
      Servoinput[4] = 163 - 135; //BR_2
      Servoinput[5] = 18 + 1; //BR_3
      allservo();

      Servoinput[10] = 3 + 180; //BL_2
      allservo();

      Servoinput[9] = 2 + 45; //BL_1
      Servoinput[10] = 3 + 135; //BL_2
      Servoinput[11] = 22 - 1; //BL_3
      allservo();

      Servoinput[1] = 170 - 125; //FR_2
      allservo();

      Servoinput[0] = 24 + 45; //FR_1
      Servoinput[1] = 170 - 165; //FR_2
      allservo();

      Servoinput[7] = 3 + 120; //FL_2
      allservo();

      Servoinput[6] = 166 - 45; //FL_1
      Servoinput[7] = 3 + 165; //FL_2
      allservo();

      //휠 -> 보행
    }
    current_modetrigger = valueCH5;
  }
  if (valueCH5 == 0) {// wheelmode
    // ELEV로 전진/후진 조절
    if (valueCH3_F > 50) {
      digitalWrite(M1, 1);
      digitalWrite(M2, 0);
      analogWrite(E1, valueCH3_F);
      analogWrite(E2, valueCH3_F);
    }

    else if (valueCH3_B > 30) {
      digitalWrite(M1, 0);
      digitalWrite(M2, 1);
      analogWrite(E1, valueCH3_B);
      analogWrite(E2, valueCH3_B);

    }
    else if (valueCH3_B <= 30 && valueCH3_F <= 30 ) {
      analogWrite(E1, 0);
      analogWrite(E2, 0);
    }
    // AILE로 좌우 방향 조절
    if (valueCH2_L > 30) {
      digitalWrite(M1, 0);
      digitalWrite(M2, 1);
      analogWrite(E1, valueCH2_L);
      analogWrite(E2, valueCH2_L);
    }
    else if (valueCH2_R > 30) {
      digitalWrite(M1, 1);
      digitalWrite(M2, 0);
      analogWrite(E1, valueCH2_R);
      analogWrite(E2, valueCH2_R);
    }

  }
  else if (valueCH5 == 1) {
    if (valueCH3_F > 50) {
      forward();
    }
    else if (valueCH3_F < -50) {
      backward();
    }
  }
}

void update_currentjoint() {
  //  Serial.println("updating");
  //실제 사용하는 각도값
  Joint.FR_1 = Joint.FR_1_current + *Joint.FR_1_angle; // 0 //rolling 시작할때각도
  Joint.FR_2 = Joint.FR_2_current - *Joint.FR_2_angle; //180
  Joint.FR_3 = Joint.FR_3_current + *Joint.FR_3_angle; //0

  Joint.BR_1 = Joint.BR_1_current - *Joint.BR_1_angle; //180
  Joint.BR_2 = Joint.BR_2_current - *Joint.BR_2_angle; //180
  Joint.BR_3 = Joint.BR_3_current + *Joint.BR_3_angle; //0

  Joint.BL_1 = Joint.BL_1_current + *Joint.BL_1_angle; //0
  Joint.BL_2 = Joint.BL_2_current + *Joint.BL_2_angle; //0
  Joint.BL_3 = Joint.BL_3_current + *Joint.BL_3_angle;

  Joint.FL_1 = Joint.FL_1_current - *Joint.FL_1_angle; //180
  Joint.FL_2 = Joint.FL_2_current + *Joint.FL_2_angle; //0
  Joint.FL_3 = Joint.FL_3_current + *Joint.FL_3_angle;

  Joint.FR_1_current = Joint.FR_1;
  Joint.FR_2_current = Joint.FR_2;
  Joint.FR_3_current = Joint.FR_3;

  Joint.BR_1_current = Joint.BR_1;
  Joint.BR_2_current = Joint.BR_2;
  Joint.BR_3_current = Joint.BR_3;

  Joint.BL_1_current = Joint.BL_1;
  Joint.BL_2_current = Joint.BL_2;
  Joint.BL_3_current = Joint.BL_3;

  Joint.FL_1_current = Joint.FL_1;
  Joint.FL_2_current = Joint.FL_2;
  Joint.FL_3_current = Joint.FL_3;
  return Joint;
}

void forward() {
  if (Joint.count = 0) {
    moveLeg("BR", 30, 0, 0);
    Joint.count = 1;
  }
  else if (Joint.count = 1) {
    moveLeg("FR", -30, 0, 0);
    Joint.count = 2;
  }
  else if (Joint.count = 2) {
    moveBody(25, -25, -25, 25);
    Joint.count = 3;
  }
  else if (Joint.count = 3) {
    moveLeg("BL", 30, 0, 0);
    Joint.count = 4;
  }
  else if (Joint.count = 4) {
    moveLeg("FL", -30, 0, 0);
    Joint.count = 5;
  }
  else if (Joint.count = 5) {
    moveBody(25, -25, -25, 25);
    Joint.count = 0;
  }

  return Joint;
}

void backward() {
  if (Joint.count = 5) {
    moveBody(25, -25, -25, 25);
    Joint.count = 4;
  }
  else if (Joint.count = 4) {
    moveLeg("FL", -30, 0, 0);
    Joint.count = 3;
  }
  else if (Joint.count = 3) {
    moveLeg("BL", 30, 0, 0);
    Joint.count = 2;
  }
  else if (Joint.count = 2) {
    moveBody(50, 150, 0, 0);
    Joint.count = 1;
  }
  else if (Joint.count = 1) {
    moveLeg("FR", -30, 0, 0);
    Joint.count = 0;
  }
  else if (Joint.count = 0) {
    moveLeg("BR", 30, 0, 0);
    Joint.count = 5;
  }
}


void moveLeg(String a, int b, int c, int d ) {// 여기 변수정의 나중에 잿슨이랑 통신해서 값 주고받을 땐 문자열로 받아서 구동하도록 수정해야함.
  int leg;
  if (a == "BR") {
    //오른쪽 뒷다리
    //각 조인트에 각도 입력
    Joint.BR_1_angle = &b;
    Joint.BR_2_angle = &c;
    Joint.BR_3_angle = &d;
    update_currentjoint();
    // 다리를 이동시키기 전 다리를 들어올리는 문구
    int stepBR2 = Joint.BR_2 - Joint.leg_up;
    int stepBR3 = Joint.BR_3 + Joint.leg_up;
    ServoBR2.startEaseToD(stepBR2, Millis);
    ServoBR3.startEaseToD(stepBR3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    // 1번 조인트를 움직여 다리를 이동시키는 문구
    ServoBR1.startEaseToD(Joint.BR_1, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }//다리 내리는 코드
    ServoBR2.startEaseToD(Joint.BR_2, Millis);
    ServoBR3.startEaseToD(Joint.BR_3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    Serial.println("BR");
    //    Serial.println(*Joint.FR_1_angle);
    //    Serial.println(*Joint.FR_2_angle);
    //    Serial.println(*Joint.FR_3_angle);
    //    Serial.println("---------------");

  }
  else if (a == "FR") {
    //오른쪽 앞다리
    //각 조인트에 각도 입력
    Joint.FR_1_angle = &b;
    Joint.FR_2_angle = &c;
    Joint.FR_3_angle = &d;
    update_currentjoint();
    // 다리를 이동시키기 전 다리를 들어올리는 문구
    int step1_1 = Joint.FR_2 - Joint.leg_up;
    int step1_2 = Joint.FR_3 + Joint.leg_up;
    ServoFR2.startEaseToD(step1_1, Millis);
    ServoFR3.startEaseToD(step1_2, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    // 1번 조인트를 움직여 다리를 이동시키는 문구
    ServoFR1.startEaseToD(Joint.FR_1, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }//다리 내리는 코드
    ServoFR2.startEaseToD(Joint.FR_2, Millis);
    ServoFR3.startEaseToD(Joint.FR_3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    Serial.println("FR");
    //    Serial.println(*Joint.FR_1_angle);
    //    Serial.println(*Joint.FR_2_angle);
    //    Serial.println(*Joint.FR_3_angle);
    //    Serial.println("---------------");

  }
  else if (a == "BL") {
    //왼쪽 뒷다리
    //각 조인트에 각도 입력
    Joint.BL_1_angle = &b;
    Joint.BL_2_angle = &c;
    Joint.BL_3_angle = &d;
    update_currentjoint();
    // 다리를 이동시키기 전 다리를 들어올리는 문구
    int stepBL2 = Joint.BL_2 - Joint.leg_up;
    int stepBL3 = Joint.BL_3 + Joint.leg_up;
    ServoBL2.startEaseToD(stepBL2, Millis);
    ServoBL3.startEaseToD(stepBL3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    // 1번 조인트를 움직여 다리를 이동시키는 문구
    ServoBL1.startEaseToD(Joint.BL_1, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }//다리 내리는 코드
    ServoBL2.startEaseToD(Joint.BL_2, Millis);
    ServoBL3.startEaseToD(Joint.BL_3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    Serial.println("movLeg done");
    Serial.println(*Joint.FR_1_angle);
    Serial.println(*Joint.FR_2_angle);
    Serial.println(*Joint.FR_3_angle);
    Serial.println("---------------");

  }
  else if (a == "FL") {
    //왼쪽 앞다리
    //각 조인트에 각도 입력
    Joint.FL_1_angle = &b;
    Joint.FL_2_angle = &c;
    Joint.FL_3_angle = &d;
    update_currentjoint();
    // 다리를 이동시키기 전 다리를 들어올리는 문구
    int stepFL2 = Joint.FL_2 - Joint.leg_up;
    int stepFL3 = Joint.FL_3 + Joint.leg_up;
    ServoFL2.startEaseToD(stepFL2, Millis);
    ServoFL3.startEaseToD(stepFL3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    // 1번 조인트를 움직여 다리를 이동시키는 문구
    ServoFL1.startEaseToD(Joint.FL_1, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }//다리 내리는 코드
    ServoFL2.startEaseToD(Joint.FL_2, Millis);
    ServoFL3.startEaseToD(Joint.FL_3, Millis);
    while (ServoEasing::areInterruptsActive()) {
      ; // no delays here to avoid break between forth and back movement
    }
    //    Serial.println("movLeg done");
    //    Serial.println(*Joint.FR_1_angle);
    //    Serial.println(*Joint.FR_2_angle);
    //    Serial.println(*Joint.FR_3_angle);
    //    Serial.println("---------------");

  }
  //String leg = a;
  //  Serial.println("movelegstart");
  //  Serial.println(leg);
  //  delay(1000);

  return Joint;
}

void moveBody(int a, int b, int c, int d) {
  Joint.FR_1_angle = &a;
  Joint.BR_1_angle = &b;
  Joint.BL_1_angle = &c;
  Joint.FL_1_angle = &d;
  ServoFR1.startEaseToD(Joint.FR_1_angle, Millis);
  ServoBR1.startEaseToD(Joint.BR_1_angle, Millis);
  ServoFL1.startEaseToD(Joint.BL_1_angle, Millis);
  ServoBL1.startEaseToD(Joint.FL_1_angle, Millis);
  while (ServoEasing::areInterruptsActive()) {
    ; // no delays here to avoid break between forth and back movement
  }
}

void allservo() {
  ServoFR1.startEaseToD(Servoinput[0], Millis);
  ServoFR2.startEaseToD(Servoinput[1], Millis);
  ServoFR3.startEaseToD(Servoinput[2], Millis);
  ServoBR1.startEaseToD(Servoinput[3], Millis);
  ServoBR2.startEaseToD(Servoinput[4], Millis);
  ServoBR3.startEaseToD(Servoinput[5], Millis);
  ServoBL1.startEaseToD(Servoinput[6], Millis);
  ServoBL2.startEaseToD(Servoinput[7], Millis);
  ServoBL3.startEaseToD(Servoinput[8], Millis);
  ServoFL1.startEaseToD(Servoinput[9], Millis);
  ServoFL2.startEaseToD(Servoinput[10], Millis);
  ServoFL3.startEaseToD(Servoinput[11], Millis);
  while (ServoEasing::areInterruptsActive()) {
    ; // no delays here to avoid break between forth and back movement
  }
}
