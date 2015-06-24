int main(int argc, char **argv){
  int listen = atoi(argv[1]);
  int timeout = atoi(argv[2]);
  int ssl = atoi(argv[3]);
  int local = atoi(argv[4]);
  int anon = atoi(argv[5]);
  int log = atoi(argv[6]);
  int chunk = atoi(argv[7]);
  
  int one_process_mode = 1;
  int vsftp_data_buf_siz = 65536;

  if (listen){
    if (timeout)
      printf("l1\n");
    else
      printf("l2\n");
  }
  else
    printf("l3\n");
  
  printf("l4\n");
  if (one_process_mode)
    if (local || ssl){
	printf("l5\n");
	return 0;
    }
  
  printf("l6\n");
  if (local==0 && anon==0)
    return 0;
  
  printf("l7\n");

  if (chunk < vsftp_data_buf_siz && chunk > 0){
    if (log)
      printf("l8\n");
    else
      printf("l9\n");
  }

  return 0;

}

        

