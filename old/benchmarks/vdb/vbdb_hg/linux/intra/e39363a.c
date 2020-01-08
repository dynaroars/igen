
#ifdef UNK_TRUE
#define unk() 1
#else
#include <stdlib.h>
#define unk() (rand() % 2)
#endif

int main()
{
#ifdef CONFIG_NETPOLL
//  netpoll_setup();
  int err;

#ifdef CONFIG_IPV6
  err = -1;
  if (unk())
    goto put;
#else
  goto put;
#endif

put:
  return err; // ERROR
#endif
  return 0;
}

