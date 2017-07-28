
#include <stdio.h>

int main(int argc, char** argv) {
#ifdef CONFIG_RCU_CPU_STALL_INFO
//  print_cpu_stall_info(0);
  char fast_no_hz[72];
  int cpu = 0;
  
  #ifdef CONFIG_RCU_FAST_NO_HZ
  sprintf(fast_no_hz, "foobar");
  #else
  
  #endif
  printf("\t%d: %s\n", cpu, fast_no_hz); // ERROR
#else
  printf(" %d", cpu);
#endif
  return 0;
}
