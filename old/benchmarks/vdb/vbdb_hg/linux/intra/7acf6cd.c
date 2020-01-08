
#ifdef UNK_TRUE
#define unk() 1
#else
#include <stdlib.h>
#define unk() (rand() % 2)
#endif

int some_int = 1;

int main()
{
#ifdef CONFIG_UNIX98_PTYS
//  ptmx_open();
  int *driver_data;

  if (unk()) {
    goto err_release;
  }

  driver_data = &some_int;
  
  return 0;

err_release:
#ifdef CONFIG_DEVPTS_MULTIPLE_INSTANCES
  if (*driver_data)
    ;
#endif
  return -1;
#endif
}

