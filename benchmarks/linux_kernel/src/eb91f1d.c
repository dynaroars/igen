
#if defined(CONFIG_LOCKDEP) && defined(CONFIG_TRACE_IRQFLAGS) && defined(CONFIG_PROVE_LOCKING)
#include <assert.h>
#endif
#include <stdbool.h>

#define ___GFP_WAIT             0x10u
#define ___GFP_IO               0x40u
#define ___GFP_FS               0x80u
#define __GFP_WAIT      ((gfp_t)___GFP_WAIT)    /* Can wait and reschedule? */
#define __GFP_IO        ((gfp_t)___GFP_IO)      /* Can start physical IO? */
#define __GFP_FS        ((gfp_t)___GFP_FS)      /* Can call down to low-level FS? */
#define GFP_KERNEL      (__GFP_WAIT | __GFP_IO | __GFP_FS)

typedef int gfp_t;

bool irqs_disabled = false;

int main() 
{
//  local_irq_disable();
  irqs_disabled = true;
//  mm_init();
#ifdef CONFIG_SLAB
//  kmalloc_node(GFP_KERNEL);
  #ifdef CONFIG_NUMA
    #ifdef CONFIG_KMEMTRACE
      #if defined(CONFIG_LOCKDEP) && defined(CONFIG_TRACE_IRQFLAGS) && defined(CONFIG_PROVE_LOCKING)
      if (!(GFP_KERNEL & __GFP_WAIT))
        return;

      if (!(GFP_KERNEL & __GFP_FS))
        return;

      assert(!irqs_disabled);
      #else
      #endif
    #else
    
    #endif
  #endif

  #if !defined(CONFIG_NUMA)
  return;
  #endif
#else
  
#endif
  return 0;
}

