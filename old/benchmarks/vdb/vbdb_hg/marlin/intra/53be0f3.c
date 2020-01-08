#include <util/delay.h> // http://download.savannah.gnu.org/releases/avr-libc/binary-releases/ for the util/delay library
#include <stdbool.h>
#include <stdio.h>

#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2

uint8_t axis = 0;

int main(void) 
{
#ifdef BABYSTEPPING
#define BABYSTEP_AXIS(axis, AXIS, INVERT) { \
        /* enable_## axis(); */ \
        uint8_t old_pin = AXIS; \
        /* more functions here */ \
        _delay_us(1U); \ 
        /* more functions here */ \
      }

    switch(axis) {

      case X_AXIS:
        printf("%s", "enabling x axis");
        break;

      case Y_AXIS:
        printf("%s", "enabling y axis");
        break;
     
      case Z_AXIS: {

        #ifndef DELTA

        printf("%s", "enabling y axis");
        
        #else // DELTA

          printf("%s", "enabling x axis");
          printf("%s", "enabling y axis");
          printf("%s", "enabling z axis");
          //setup new step
          //functions to move the x,y,z axis
        _delay_us(1U);
          //functions to move the x,y,z axis
        
        #endif
        break;
      } 
     
      default: break;
    }

#endif //BABYSTEPPING

    return 0;
}