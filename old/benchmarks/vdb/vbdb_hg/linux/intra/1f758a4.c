
#include <errno.h>
#include <stdlib.h>

#define ___GFP_WAIT             0x10u
#define ___GFP_IO               0x40u
#define ___GFP_FS               0x80u
#define __GFP_WAIT      ((gfp_t)___GFP_WAIT)    /* Can wait and reschedule? */
#define __GFP_IO        ((gfp_t)___GFP_IO)      /* Can start physical IO? */
#define __GFP_FS        ((gfp_t)___GFP_FS)      /* Can call down to low-level FS? */
#define GFP_KERNEL      (__GFP_WAIT | __GFP_IO | __GFP_FS)

#define ___GFP_DMA              0x01u
#define __GFP_DMA       ((gfp_t)___GFP_DMA)
#define GFP_DMA         __GFP_DMA

typedef int gfp_t;

static int some_int = 0;

int main(int argc, char** argv)
{
#ifdef CONFIG_EP93XX_ETH
//  ep93xx_open();
  #ifndef CONFIG_ZONE_DMA
  if ((GFP_KERNEL | GFP_DMA) & __GFP_DMA)
    return NULL;
  #endif
  int *descs = &some_int;
  if (descs == NULL)
    return -ENOMEM; // ERROR
#endif
  return 0;
}
