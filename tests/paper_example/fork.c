#include <stdio.h>

#ifdef CONFIG_B
extern int probe();
#endif

int fork() {
  printf("fork\n");
  #ifdef CONFIG_B
  probe();
  #endif
}

int main() {
  fork();
}
