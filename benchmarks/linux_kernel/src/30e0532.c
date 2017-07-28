
#include <stdlib.h>

int main(int argc, char** argv) {
//  reiserfs_create();
  int retval;
  int length;

  int error;
//  error = security_old_inode_init_security(length);
#ifdef CONFIG_SECURITY
  if (rand() % 2) {
    length = 0;
    error = 0;
  }
  else
    error = -1;
#else
  error = 0;
#endif

  if (error) {
    length = 0;
    retval = error;
  }

  if (length) { // ERROR
    // do something
  }

  if (retval < 0) {
    return retval;
  }

  return 0;
}

