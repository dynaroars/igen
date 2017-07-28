
#include <errno.h>
#include <stdio.h>

#define UEVENT_BUFFER_SIZE		2048	/* buffer for the variables */

static char buf[UEVENT_BUFFER_SIZE];
static int buflen = 0;

int main(int argc, char** argv)
{
//  input_add_uevent_modalias_var();
  int r;
#if defined(CONFIG_HOTPLUG)
  int len = sprintf(&buf[buflen], "MODALIAS=");
  buflen += len + 1;
  r = 0;
#else
  r = 0;
#endif
  if (r)
    return -ENOMEM;

  buf[buflen - 1] = 'x'; // ERROR: buflen == 0, so buf[-1] is written.
  return 0;
}
