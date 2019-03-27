
#if defined(CONFIG_X86) && defined(CONFIG_ACPI_WMI)
#include <string.h>

static char block_object_id[3] = "XX";
#endif

int main()
{
#if defined(CONFIG_X86) && defined(CONFIG_ACPI_WMI)
//  acer_wmi_init();
  char method[4];

  strcpy(method, "WQ");
  strncat(method, block_object_id, 2); // ERROR
#endif
  return 0;
}

