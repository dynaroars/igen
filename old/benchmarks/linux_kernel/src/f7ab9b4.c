
#ifdef CONFIG_TMPFS
#define CONFIG_SHMEM
#endif

#include <stdlib.h>

typedef int filler_t(int);

int some_fun(int n)
{
  return n;
}

#ifdef CONFIG_TMPFS
filler_t *f = &some_fun;
#else
filler_t *f = NULL;
#endif

int main()
{
#ifdef CONFIG_DRM_I915
//  i915_driver_load();
  filler_t *readpage;
//  return i915_load_modeset_init(&readpage);
//  i915_gem_alloc_object(readpage);
  #ifdef CONFIG_SHMEM
  *readpage = f;
  #else
  *readpage = &some_fun;
  #endif
  int i = 0;
  if (i != 0)
    ;
//  return i915_gem_object_pin(readpage);
  filler_t *filler = **readpage;
  filler(0); // ERROR
#endif
  return 0;
}

