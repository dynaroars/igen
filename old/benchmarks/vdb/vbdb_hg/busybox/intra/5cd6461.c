
#include <stdio.h>
#include <stdlib.h>

#define HTTP_UNAUTHORIZED      401 /* auth needed, respond with auth hdr */
#define HTTP_NOT_IMPLEMENTED   501 /* used for unrecognized requests */
#define HTTP_FORBIDDEN         403

#ifdef CONFIG_LFS
#define cont_l_fmt "%lld"
#else
#define cont_l_fmt "%ld"
#endif

#ifdef CONFIG_FEATURE_HTTPD_CGI
static const char request_GET[] = "GET";
#endif

int main(int argc, char** argv)
{
  const char request[] = "POST";
//  handleIncoming(request);
#ifdef CONFIG_LFS
  long long total = 10LL; 
#else
  long total = 10L;
#endif
  
#ifdef CONFIG_FEATURE_HTTPD_BASIC_AUTH
  int random = rand() % 2;
  int http_unauthorized = random;

  if (http_unauthorized) {
    printf("%ld\r\n", total); //ERROR
  }
#endif

#ifdef CONFIG_FEATURE_HTTPD_CGI
  if (request != request_GET)
    printf("%ld\r\n", total); //ERROR
#endif
  return 0;
}