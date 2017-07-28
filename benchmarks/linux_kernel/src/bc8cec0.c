
#include <errno.h>
#include <stdlib.h>

#define WBUF_PAGESIZE 32

unsigned char *wbuf; /* Write-behind buffer for NAND flash */
#ifdef CONFIG_JFFS2_FS_WBUF_VERIFY
unsigned char *wbuf_verify; /* read-back buffer for verification */
#endif

int main()
{
//  jffs2_nor_wbuf_flash_setup();
  wbuf = malloc(WBUF_PAGESIZE);
  if (!wbuf)
    return -ENOMEM;
//  jffs2_flash_write();
  #ifdef CONFIG_JFFS2_FS_WBUF_VERIFY
  return(int) *wbuf_verify; // ERROR
  #else
  do { } while (0);
  #endif
  return 0;
}

