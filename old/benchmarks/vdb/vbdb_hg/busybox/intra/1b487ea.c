
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char **argv)
{
//  do_stat("filename");
  char *filename = "filename";
#ifdef ENABLE_SELINUX
  char *scontext = NULL;
#endif

#ifndef ENABLE_FEATURE_STAT_FORMAT

#ifdef ENABLE_SELINUX
  if(rand() % 2)
    printf(" %lc\n", *scontext); // ERROR
#endif
  
  printf("  File: '%s'\n", filename);
#endif
  return 0;
}