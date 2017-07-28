
#include <errno.h>
#include <stdlib.h>
#include <string.h>

struct module {
  void *refptr;
  void *module_core;
};

int main(int argc, char** argv) {
//  load_module(64 * 1024);
  void *hdr;
  struct module *mod;
  long err = 0;
  void *ptr = NULL;
	
  if ((hdr = malloc(len)) == NULL)
    return -ENOMEM;
	
  mod = (void *)hdr;

#if defined(CONFIG_MODULE_UNLOAD) && defined(CONFIG_SMP)
  mod->refptr = malloc(32);
  if (!mod->refptr) {
    err = -ENOMEM;
    goto free_mod;
  }
#endif

  ptr = malloc(512);
  if (!ptr) {
    err = -ENOMEM;
    goto free_percpu;
  }
  mod->module_core = ptr;
  
  while (1) {
    void *dest = mod->module_core;
    memcpy(dest, mod, sizeof(struct module));
    mod = mod->module_core; 
    break;
  }

  if (rand() % 2)
    goto free_unload;

  return 0;

  free_unload:
  free_core:
    free(mod->module_core);
  free_percpu:
#if defined(CONFIG_MODULE_UNLOAD) && defined(CONFIG_SMP)
    free(mod->refptr); // ERROR
#endif
  free_mod:
  free_hdr:
    free(hdr);

  return 0;
}

