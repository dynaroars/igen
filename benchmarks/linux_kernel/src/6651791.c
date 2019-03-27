
#ifdef CONFIG_SND_FSI_AK4642
#define CONFIG_SND_SOC_AK4642
#endif

#ifdef CONFIG_SND_FSI_DA7210
#define CONFIG_SND_SOC_DA7210
#endif

#ifdef CONFIG_I2C
static int i2c_master_send = 0;
#endif

int main(int argc, char** argv)
{
#ifdef CONFIG_SND_SOC_DA7210
//  da7210_init();
  int codec_hw_write = i2c_master_send; // ERROR
#endif
#ifdef CONFIG_SND_SOC_AK4642
//  ak4642_init();
  int codec_hw_write2 = i2c_master_send; // ERROR
#endif
  return 0;
}

