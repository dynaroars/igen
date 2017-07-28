
#include <assert.h>
#include <stdbool.h>

#ifdef UNK_TRUE
#define unk() 1
#else
#include <stdlib.h>
#define unk() (rand() % 2)
#endif

int main(int argc, char** argv)
{
//  __run_timers();
  int preempt_counter = 0;
  int preempt_count = preempt_counter;
//  inet_twdr_hangman();
  #ifdef CONFIG_TCP_MD5SIG
  if (unk())
    #ifdef CONFIG_PREEMPT
      preempt_counter--
    #else
    #endif
  #endif

  if (preempt_count != preempt_counter) {
    assert(false); // ERROR
  }
  return 0;
}

