
int main(int argc, char** argv) {
//  trace_dump_stack(0); // ERROR
  #ifdef CONFIG_TRACING
  // do something
  return;
  #else
  #endif
  return 0;
}
