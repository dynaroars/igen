
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
#if defined(CONFIG_IPV6) || defined(CONFIG_VLAN_8021Q)
//  ocrdma_inet6addr_event();
//  vlan_dev_real_dev();
  #ifdef CONFIG_VLAN_8021Q
    return NULL;
  #else
    assert(false); // ERROR
    return NULL;
  #endif
#endif
  return 0;
}

