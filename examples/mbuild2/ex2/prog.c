
#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv){

  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  int z = atoi(argv[3]);

  if (x==5){
    printf("L0\n"); //x == 5
    if(y!=700){
      printf("L1o\n"); //x == 5 & y == 7 | 70
    }
  }
  
  return 0;
}
  


