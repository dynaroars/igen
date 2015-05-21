#include <stdio.h>
int main(int argc, char **argv){
  int listen = atoi(argv[1]);
  int timeout = atoi(argv[2]);
  int ssl = atoi(argv[3]);
  int local = atoi(argv[4]);
  int anon = atoi(argv[5]);
  int log = atoi(argv[6]);
  int chunk = atoi(argv[7]);
  int one_process_mode = 1;
  //  printf("%d,%d,%d,%d,%d,%d\n",listen,timeout,ssl,local,anon,log);
  if (listen){
    if (timeout)
      printf("L1\n");
    else
      printf("L2\n");
  }
  else
    printf("L3\n");
  
  printf("L4\n");
  if (one_process_mode)
    if (local || ssl){
      printf("L5\n");
	return 0;
    }
  
  printf("L6\n");
  if (local==0 && anon==0)
    return 0;

  //local || anon  
  printf("L7\n");

  if (chunk == 2048 || chunk == 4096){
    if (log)
      printf("L8\n");
    else
      printf("L9\n");
  }

  return 0;

}

        

