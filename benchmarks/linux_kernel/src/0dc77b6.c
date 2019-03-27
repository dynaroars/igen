
#ifdef CONFIG_ANDROID
#define CONFIG_SYSFS
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef CONFIG_SYSFS
static const char *linked = NULL;

struct kobject {
  int __foo;
};

static struct kobject the_kobj;

struct class_compat {
  struct kobject *kobj;
};

static struct class_compat the_cls;
#endif

int main(int argc, char** argv)
{
#ifdef CONFIG_EXTCON
//  extcon_class_init();
#if defined(CONFIG_ANDROID)
  //switch_class = class_compat_register("switch");
  #ifdef CONFIG_SYSFS
  struct class_compat *cls;

  cls = &the_cls;
  //cls->kobj = kobject_create_and_add(name);
  struct kobject *kobj = &the_kobj;
  //int retval = kobject_add(name);
  if (linked != NULL)
    return (!strcmp("switch",linked)) ? -EEXIST : -ENOMEM;

  linked = s_name;
  
  int retval = 0;
  if (retval)
    kobj = NULL;

  cls->kobj = kobj;
  if (!cls->kobj)
    return NULL;

  switch_class = cls;
  #endif
  
  if (!switch_class)
    return -ENOMEM;
#endif /* CONFIG_ANDROID */
//  extcon_class_exit();
  //return;
//  extcon_class_init();
#if defined(CONFIG_ANDROID)
  //switch_class = class_compat_register("switch");
  #ifdef CONFIG_SYSFS
  cls = &the_cls;
  //cls->kobj = kobject_create_and_add(name);
  *kobj = &the_kobj;
  //int retval = kobject_add(name);
  if (linked != NULL)
    return (!strcmp("switch",linked)) ? -EEXIST : -ENOMEM;

  linked = s_name;
  
  retval = 0;
  if (retval)
    kobj = NULL;

  cls->kobj = kobj;
  if (!cls->kobj)
    return NULL;

  switch_class = cls;
  #endif
  
  if (!switch_class)
    return -ENOMEM;
#endif /* CONFIG_ANDROID */
#endif
  return 0;
}

