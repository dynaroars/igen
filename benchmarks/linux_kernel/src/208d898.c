
#include <stdlib.h>

#define local_irq_enable()
#define local_irq_disable()
#define local_bh_enable()
#define local_bh_disable()

int main(int argc, char** argv)
{
//  udp_poll();
  unsigned int mask = 0;

  local_bh_disable();
  while (rand() % 2) {
    unsigned int csum = 0;
    while (rand() % 2) {
      if (rand() % 2) {
        #ifdef CONFIG_HIGHMEM
	local_bh_enable(); // ERROR
        #endif
      }
    }
  }
  local_bh_enable();
  
  return 0;
}

