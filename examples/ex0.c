#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv){

  // options: s,t,u,v, x,y,z
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  
  if (x&&y){
    printf("L0\n"); //x & y
  }
  else{
    printf("L1\n"); // !x|!y

  }

  return 0;
}
  


