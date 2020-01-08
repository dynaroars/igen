
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
#ifdef CONFIG_SMP
  int smp_found_config;
  smp_found_config = false;
//  native_smp_prepare_cpus();
  static int called = 0;

  called++;
  if (called > 1) 
    do { fprintf(stderr, "panic!\n"); exit(1); } while(0);
  
  if (!smp_found_config) {
      called = 0;

      called++;
      if (called > 1) 
        do { fprintf(stderr, "panic!\n"); exit(1); } while(0);
  }
#endif
  return 0;
}

