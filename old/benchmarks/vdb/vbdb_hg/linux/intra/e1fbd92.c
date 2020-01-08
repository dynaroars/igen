
#include <stdio.h>

#ifdef CONFIG_LBDAF
typedef unsigned long long sector_t;
#else
typedef unsigned long sector_t;
#endif

int main() {
  #ifdef CONFIG_AMIGA_Z2RAM
//  do_z2_request();
  printf("bad access: block=%lu\n", (sector_t) 0); // ERROR
  #endif
  return 0;
}
