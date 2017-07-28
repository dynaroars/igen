
#ifdef CONFIG_VORTEX
#include <stdlib.h>

int some_int = 1;


#ifdef CONFIG_PCI
#define DEVICE_PCI(dev) ((dev % 2) ? &some_int : NULL)
#else
#define DEVICE_PCI(dev) NULL
#endif

#define VORTEX_PCI(vp) ((vp) ? DEVICE_PCI(vp) : NULL)

int main()
{
#ifdef CONFIG_VORTEX
//  acpi_set_WOL(1); 
  if (1) {
    if (*VORTEX_PCI(1) < 0) // ERROR
      return;
  }
#endif
  return 0;
}

