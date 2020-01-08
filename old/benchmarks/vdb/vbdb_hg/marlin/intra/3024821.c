#include <stdio.h>

int main(void){
unsigned long millis = 123456789l;
#if defined (THERMAL_RUNAWAY_PROTECTION_PERIOD) && THERMAL_RUNAWAY_PROTECTION_PERIOD > 0
  int state = 0;
  unsigned long *timer = 123456782l;
  float temperature = 145.0f;
  float target_temperature = 200.0f;
  int heater_id = 1;
  int period_seconds = 10;
  int hysteresis_degc = 1;
  
  
if (temperature >= (target_temperature - hysteresis_degc))
      {
        *timer = 123456782;
      } 
      else if ( (millis - *timer) > (period_seconds) * 1000)
      {
        printf("%s","Thermal Runaway, system stopped! Heater_ID: ");
        printf("%d",(int)heater_id);
        printf("%s","THERMAL RUNAWAY");
      }
 
#endif
return 0;
}