
#include <stdio.h>

int main(int argc, char** argv)
{
//  ext2fs_open();
  char *gdp;
    
#ifdef EXT2FS_ENABLE_SWAPFS
  gdp = "group_desc";
  for (int j = 0; j < 10; j++)
  {
    printf("%c\n", (*gdp)++);
  }
#endif
  return 0;
}