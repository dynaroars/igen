
#include <errno.h>
#include <stdlib.h>

#ifdef CONFIG_PROC_PAGE_MONITOR

#define PM_ENTRY_BYTES sizeof(long)
#define PM_END_OF_BUFFER 1
#define NUM_ENTRIES 10

long *pm_buffer;
unsigned long pm_pos, pm_len;
#endif

int main(int argc, char** argv) {
#ifdef CONFIG_PROC_PAGE_MONITOR
//  pagemap_read();
  pm_len = PM_ENTRY_BYTES * NUM_ENTRIES;
  pm_buffer = malloc(pm_len);

  if (!pm_buffer)
    return -ENOMEM;

  pm_pos = 0;
  int err = 0;

  while (1) {
    //err = add_to_pagemap();
    pm_buffer[pm_pos++] = 0; // ERROR
    if (pm_pos >= pm_len)
      err = PM_END_OF_BUFFER;
    if (err)
      return err;
  }
#endif
  return 0;
}
