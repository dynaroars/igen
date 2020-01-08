#include <string.h>

int main(int argc, char** argv)
{
static char *strchr_pointer = "eiffel~1.gco*57";
char *starpos = NULL;
#ifdef SDSUPPORT
	starpos = (strchr(strchr_pointer + 4,'*'));
	if(starpos!=NULL)
		*(starpos-1)='\0'; //ERROR
#endif
return 0;
}

