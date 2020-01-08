#include <unistd.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>

int main(int argc, char** argv)
{
  char *buffer;
//  allocate_buffer();
#if !defined(CONFIG_KGDB) && !defined(CONFIG_XMON) && !defined(CONFIG_BDI_SWITCH)
  int pagesize;

  pagesize = sysconf(_SC_PAGE_SIZE);
  if (pagesize == -1)
    do { perror("sysconf"); exit(EXIT_FAILURE); } while (0);

  buffer = memalign(pagesize, 4*pagesize);
  if (buffer == NULL)
    do { perror("memalign"); exit(EXIT_FAILURE); } while (0);

  if (mprotect(buffer, 4*pagesize, PROT_READ) == -1)
    do { perror("mprotect"); exit(EXIT_FAILURE); } while (0);
#else
  buffer = malloc(4092);
#endif

#ifdef CONFIG_KPROBES
  *buffer = 'a'; // ERROR
#endif
   return 0;
}
