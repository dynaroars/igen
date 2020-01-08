
#include <assert.h>
#include <stdlib.h>

#ifdef CONFIG_ARCH_OMAP3
static int* _omap3_sram_configure_core_dpll;
#endif

int main(int argc, char** argv)
{
//  omap3pandora_init();
  if (rand() % 2) {
#if defined(CONFIG_ARCH_OMAP3) && defined(CONFIG_PM)
    int some_int = 1;
    _omap3_sram_configure_core_dpll = &some_int;
#endif
  }

  if (rand() % 2) {
#ifdef CONFIG_ARCH_OMAP3
    int v;
    assert(_omap3_sram_configure_core_dpll);
    v = *_omap3_sram_configure_core_dpll; // ERROR
#endif
  }
  return 0;
}

