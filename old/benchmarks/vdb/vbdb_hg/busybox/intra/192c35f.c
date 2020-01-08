
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv)
{
//  doSyslogd();
  char *msg = "syslogd started";
  static char line[512];
  
#ifdef CONFIG_FEATURE_REMOTE_LOG
  int remotefd = 1;
  
  if (remotefd) {
    memset(&line, 0, sizeof(line)); /* read the array elements, save them */
    printf(line, sizeof(line), "<%s>", msg);
  }
#endif
  return 0;
}