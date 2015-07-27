#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv){

  // options: s,t,u,v, x,y,z

  int s = atoi(argv[1]);
  int t = atoi(argv[2]);
  int u = atoi(argv[3]);
  int v = atoi(argv[4]);  

  int x = atoi(argv[5]);
  int y = atoi(argv[6]);
  int z = atoi(argv[7]);
  
  int max_z = 3;
  
  if (x&&y){
    if (!(0 < z && z < max_z)){
      printf("L1\n"); //x & y & (z=0|3|4)
    }
  }
  else{
    printf("L2\n"); // !x|!y
    printf("L3\n"); // !x|!y    
  }

  printf("L4\n"); // true
  if(s||t){
    if(u&&v){
      printf("L5\n");  // (s|t) & (u&v)
    }
  }

  return 0;
}
  


/*
full 320
1. (0) true: (1) L4
2. (2) (x=0 | y=0): (2) L2,L3
3. (3) (x=1 & y=1 & z=0,3,4): (1) L1
4. (4) (u=1 & v=1) & (s=1 | t=1): (1) L5
 (0, 1, 1), (2, 1, 2), (3, 1, 1), (4, 1, 1)

 */
