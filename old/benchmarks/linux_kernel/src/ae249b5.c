
#include <stdbool.h>

#ifdef CONFIG_DISCONTIGMEM
#include <assert.h>
#include <string.h>
#endif

#define PFNNID_MAP_MAX  512     /* support 512GB */
unsigned char pfnnid_map[PFNNID_MAP_MAX];
unsigned long max_pfn = PFNNID_MAP_MAX;

int main()
{
//  setup_bootmem();
#ifdef CONFIG_DISCONTIGMEM
  memset(pfnnid_map, 0xff, sizeof(pfnnid_map));
#endif
#ifdef CONFIG_PROC_PAGE_MONITOR
//  kpageflags_read();
  unsigned int pfn = 0;

  for (;pfn<max_pfn;pfn++) {
    #ifdef CONFIG_DISCONTIGMEM
    unsigned char r;

    assert(pfn < PFNNID_MAP_MAX);
    r = pfnnid_map[pfn];
    assert(r != 0xff); // ERROR

    int nid = (int)r;
    return (nid >= 0);
    #endif
        
    if (pfn_valid(pfn))
      ;
  }
#endif
  return 0;
}

