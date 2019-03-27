
#ifdef CONFIG_X86_32
#undef CONFIG_X86_64
#endif

#include <stdio.h>

  /* #define NR_CPUS 64 */
typedef unsigned long cpumask_t;

cpumask_t cpu_none_mask = 0;
cpumask_t cpu_online_mask = 0;

#ifdef CONFIG_NUMA
#define NODES_SHIFT     8
#define MAX_NUMNODES    (1 << NODES_SHIFT)

int nr_node_ids;

cpumask_t* node_to_cpumask_map[MAX_NUMNODES];
#else
#define nr_node_ids             1
#endif

int main(int argc, char** argv)
{
#ifdef CONFIG_NUMA
  nr_node_ids = rand() % (MAX_NUMNODES-1);
#endif
//  setup_node_to_cpumask_map();
  unsigned int node;

  /* allocate the map */
  for (node = 0; node < nr_node_ids; node++)
    node_to_cpumask_map[node] = &cpu_online_mask;
#ifdef CONFIG_PCI
//  int node = pcibios_scan_root();
  #ifdef CONFIG_NUMA
  #define BUS_NR 256
  #ifdef CONFIG_X86_64
  static int mp_bus_to_node[BUS_NR] = {
    [0 ... BUS_NR - 1] = -1
  };
  
  int node = -1;

  if (busnum < 0 || busnum > (BUS_NR - 1))
    return node;

  node = mp_bus_to_node[busnum];
  
  #else /* CONFIG_X86_32 */
  static unsigned char mp_bus_to_node[BUS_NR] = {
    [0 ... BUS_NR - 1] = -1
  };

  if (busnum < 0 || busnum > (BUS_NR - 1))
    return 0;
  node = mp_bus_to_node[busnum];
  #endif
  #endif

//  dev_attr_show(node);
  const cpumask_t *mask;
  int len = 1;

#ifdef CONFIG_NUMA
  mask = (node == -1) ? &cpu_online_mask
                      : 
                      //cpumask_of_node(node);
                      #ifdef CONFIG_DEBUG_PER_CPU_MAPS
                      if (node >= nr_node_ids) {
                        fprintf(stderr, "cpumask_of_node(%d): node > nr_node_ids(%d)\n",node, nr_node_ids);
                        cpu_none_mask;
                      }
                      if (node_to_cpumask_map[node] == NULL) {
                        fprintf(stderr, "cpumask_of_node(%d): no node_to_cpumask_map!\n", node);
                        cpu_online_mask;
                      }
	              node_to_cpumask_map[node];
                      #else
                      node_to_cpumask_map[node];
                      #endif
#else
  mask = &cpu_online_mask;
#endif
  printf("mask: %ld\n", *mask);
#endif
  return 0;
}

