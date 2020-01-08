
#include <stdio.h>
#include <stdlib.h>

#ifdef ENABLE_FEATURE_HTTPD_GZIP
int content_gzip;
#endif

int main(int argc, char** argv)
{
//  mini_httpd();
#ifdef ENABLE_FEATURE_HTTPD_GZIP
  if (rand() % 2)
    content_gzip = 1;
#endif
  
#ifdef ENABLE_FEATURE_HTTPD_BASIC_AUTH
  if (content_gzip)
    printf("Content-Encoding: gzip\r\n");
#endif
  return 0;
}
