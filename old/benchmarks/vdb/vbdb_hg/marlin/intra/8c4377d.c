#ifndef MARLIN_H
const char echomagic[] ="echo:";
#endif

int main(int argc, char** argv)
{
#ifndef MARLIN_H
#define SERIAL_ECHO_START (printf("%s\n", echomagic));
#endif

float pid_output = 1;
int e = 0;
#ifdef PIDTEMP
	#ifdef PID_DEBUG
	SERIAL_ECHO_START(" PIDDEBUG "); // ERROR
	#endif
#else
pid_output = 0;
#endif
	return 0;
}
