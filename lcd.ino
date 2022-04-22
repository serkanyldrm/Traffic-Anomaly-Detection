#include <SoftwareSerial.h>
#include <Wire.h> // Library for I2C communication
#include <LiquidCrystal_I2C.h> // Library for LCD
LiquidCrystal_I2C lcd = LiquidCrystal_I2C(0x27, 16, 2);

SoftwareSerial bt_iletisim(7,6);

void setup() {
  lcd.init();
  lcd.backlight();
  pinMode(13, OUTPUT); 
   pinMode(12, OUTPUT);
    Serial.begin(9600);
  bt_iletisim.begin(9600);
}

void loop() {
  if (bt_iletisim.available())
  {
    char data = bt_iletisim.read();
    Serial.println(data);
    if(data == 'H') 
    {
      digitalWrite(13, HIGH);
      digitalWrite(12, HIGH);  
      lcd.setCursor(4, 0);
      lcd.print("Dikkat");
      lcd.setCursor(0, 1);
      lcd.print("Hayvan Var");
      delay(300);
      while(data == 'H'){
        data = bt_iletisim.read();
        delay(10);
      }
    }
    if(data == 'S') 
    {
      digitalWrite(13, HIGH);
      digitalWrite(12, HIGH);  
      lcd.setCursor(4, 0);
      lcd.print("Dikkat");
      lcd.setCursor(0, 1);
      lcd.print("Yolda Duraklama var");
            delay(300);
      while(data == 'S'){
        data = bt_iletisim.read();
        delay(10);
      }
    }
        if(data == 'P') 
    {
      digitalWrite(13, HIGH);
      digitalWrite(12, HIGH);  
      lcd.setCursor(4, 0);
      lcd.print("Dikkat");
      lcd.setCursor(0, 1);
      lcd.print("İnsan Var");
            delay(300);
      while(data == 'P'){
        data = bt_iletisim.read();
        delay(10);
      }
    }
    lcd.clear();
    digitalWrite(13, LOW); 
    digitalWrite(12, LOW);
  }
}
