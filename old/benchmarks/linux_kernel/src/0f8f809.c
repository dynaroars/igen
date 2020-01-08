
#ifndef CONFIG_FORCE_MAX_ZONEORDER
#define MAX_ORDER 11
#else
#define MAX_ORDER CONFIG_FORCE_MAX_ZONEORDER
#endif

#if defined(CONFIG_PPC_256K_PAGES)
#define PAGE_SHIFT		18
#elif defined(CONFIG_PPC_64K_PAGES)
#define PAGE_SHIFT		16
#elif defined(CONFIG_PPC_16K_PAGES)
#define PAGE_SHIFT		14
#else
#define PAGE_SHIFT		12
#endif

#ifdef CONFIG_SLAB
#define KMALLOC_SHIFT_HIGH	((MAX_ORDER + PAGE_SHIFT - 1) <= 25 ? \
				(MAX_ORDER + PAGE_SHIFT - 1) : 25)
#else
#define KMALLOC_SHIFT_HIGH	(PAGE_SHIFT + 1)
#endif


#ifndef CONFIG_SLOB
int kmalloc_caches[KMALLOC_SHIFT_HIGH + 1];
#endif

int main(int argc, char** argv)
{
#ifndef CONFIG_SLOB
//  kmem_cache_init_late();
  #ifdef CONFIG_LOCKDEP
  int i;

  for (i = 1; i < PAGE_SHIFT + MAX_ORDER; i++) {
    int* cache = &kmalloc_caches[i]; // ERROR

    if (!cache)
      continue;

    int n = *cache;
  }
  #else
  
  #endif
#endif
  return 0;
}

