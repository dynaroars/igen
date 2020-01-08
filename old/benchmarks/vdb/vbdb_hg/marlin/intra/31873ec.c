#include <stdio.h>

int main(int argc, char **argv)
{
#ifdef ULTRA_LCD
  float x = 512.5f;
  int16_t xx=x*10;
  conv[0]=(xx>=0)?'+':'-';
  xx=abs(xx);
  conv[1]=(xx/10000)%10+'0';
  conv[2]=(xx/1000)%10+'0';
  conv[3]=(xx/100)%10+'0';
  conv[4]=(xx/10)%10+'0';
  conv[5]='.';
  conv[6]=(xx)%10+'0';
  conv[7]=0;
  printf("%l", conv);
#endif
  return 0;
}
