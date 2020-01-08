
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
//  netif_receive_skb();
  if (rand() % 2) {
    #ifdef CONFIG_VLAN_8021Q
    // do something
    return true;
    #else
    assert(false); // ERROR
    return false;
    #endif
  }
  return 0;
}
