
#ifdef CONFIG_BLK_CGROUP
#define CFQ_WEIGHT_DEFAULT	500
#endif

int main()
{
#ifdef CONFIG_IOSCHED_CFQ
//  cfq_scale_slice();
  long d = 1;
  d = d * CFQ_WEIGHT_DEFAULT; // ERROR
#endif
  return 0;
}

