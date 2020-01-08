#ifdef MESH_BED_LEVELING
	#define MESH_NUM_X_POINTS 7
	#define MESH_NUM_Y_POINTS 7
#endif 

int main(int argc, char **argv)
{

#ifndef DISABLE_M503

void Config_PrintSettings(bool forReplay){

#ifdef MESH_BED_LEVELING
    printf("%s%d"," X", MESH_NUM_X_POINTS); // ERROR
    printf("%s%d"," Y", MESH_NUM_Y_POINTS); // ERROR
#endif

}

#endif
 
  return 0;
}