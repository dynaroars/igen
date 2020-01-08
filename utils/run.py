import sys
import os
import subprocess

utils_dir = os.path.dirname(os.path.realpath(__file__))
collect_script = os.path.join(utils_dir, "collect.sh")
compare_program = os.path.join(utils_dir, "compare_configs.py")

def gen_config_line(varname, val):
  if val == "y" or val == "m":
    return "CONFIG_%s=%s" % (varname, val)
  elif val == "n":
    return "# CONFIG_%s is not set" % (varname)
  else:
    sys.stderr.write("%s must be set to y, m, or n.  saw %s.\n" % (varname, val))
    exit(1)

def gen_config_lines(configs):
  """Generate a .config file from a dictionary of varname to value mapping, where varnames do not have the CONFIG_ prefix and values are y, m, or n"""
  config_lines = [ gen_config_line(k, v) for k, v in configs.items() ]
  return config_lines

def gen_config_file(configs, filename):
  config_lines = gen_config_lines(configs)
  with open(filename, "w") as f:
    for config_line in config_lines:
      f.write(config_line)
      f.write("\n")

def check_config_file(before, after):
  returncode = subprocess.call(["python", compare_program, "kconfig.kmax", before, after], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return returncode == 0

if __name__ == "__main__":
  project = sys.argv[1]
  vars = sys.argv[2]
  vals = sys.argv[3]
  if not os.path.isdir(project):
    sys.stderr.write("project directory %s not found.\n" % (project))
    exit(1)
  configs = dict(zip(vars.split(','),vals.split(',')))
  os.chdir(project)
  subprocess.call(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  gen_config_file(configs, "generated.config")
  subprocess.call(["cp", "generated.config", ".config"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  sys.stderr.write("waiting to configure with \"make oldconfig\".  should see \"done configuring\" or else it is waiting for user input, meaning the configuration does not have variables being set.\n")
  subprocess.call(["make", "oldconfig"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  sys.stderr.write("done configuring\n")
  if (check_config_file("generated.config", ".config")):
    collectout = subprocess.check_output([collect_script, "make"], stderr=subprocess.DEVNULL)
    collectfiles = collectout.decode("utf-8").split('\n')
    filteredfiles = [ x for x in collectfiles if x.endswith(".o") and not x.startswith('/tmp') and not x.endswith('built-in.o') ]
    print(",".join(filteredfiles))
  else:
    sys.stderr.write("invalid configuration\n")
    print()
