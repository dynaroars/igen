
#include <stdbool.h>
#include <stdlib.h>

#ifndef SCTP_DEBUG
#ifdef CONFIG_SCTP_DBG_MSG
#define SCTP_DEBUG      1
#else
#define SCTP_DEBUG      0
#endif /* CONFIG_SCTP_DBG */
#endif /* SCTP_DEBUG */

#if defined(CONFIG_IP_SCTP) && defined(CONFIG_IPV6)
static int some_int = 1;
#endif

int main()
{
#if defined(CONFIG_IP_SCTP) && defined(CONFIG_IPV6)
//  sctp_v6_get_dst();
  int *dst = NULL;
  dst = &some_int;
  if (!(dst == (void *)-1)) {
    dst = NULL;
  }
  if (!(dst == (void *)-1)) {
    char *rt =(char *) dst;
//    SCTP_DEBUG_PRINTK(rt); // ERROR
    #if SCTP_DEBUG
    ((*rt)++);
    #else
    #endif
  }
#endif
  return 0;
}

