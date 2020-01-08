#ifdef ENABLE_AUTO_BED_LEVELING
	#define Z_PROBE_OFFSET_FROM_EXTRUDER 7
#endif 

int main(int argc, char **argv)
{

float zprobe_zoffset;

zprobe_zoffset = -Z_PROBE_OFFSET_FROM_EXTRUDER; // ERROR
  
return 0;
}