
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char **argv)
{
  int size = 5;
  int *arr[size];

  for (int i = 0; i < size; i++) {
//    setArr(arr, i);
    int *p = malloc(sizeof(int));
    *p = rand() % 11;
    num[index] = p;
  }

  if (rand() % 2) {
#ifdef CONFIG_FEATURE_LS_SORTFILES
//    sort(arr, size);
    int **temp;
    int i, j;

    if (arr == NULL || size < 1)
      return;

    for(i=0;i<size;i++) {
      for(j=i;j<size;j++) {
        if(*arr[i] > *arr[j]) {
          temp=*arr[i];
          *arr[i]=*arr[j];
          *arr[j]=temp;
        }
      }
    }
#endif
//    showdirs(arr);
    for (int k = 0; k < 5; ++k) {
      printf("\n%d", *arr[k]);
    }
  
#ifdef CONFIG_FEATURE_LS_RECURSIVE
//  dfree(arr); //ERROR
    int index = 1;
    int *cur;

    cur = arr[0]; // is no longer the head

    while(i < 5) {
      free(cur);
      cur = arr[index++];
    }
#endif
  }
  return 0;
}