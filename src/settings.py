import os.path
import config_common as CC

logger_level = 3

me_file = CC.getpath(__file__)
me_dir = os.path.dirname(me_file)

tmp_dir = "/var/tmp"

examples_dir = CC.getpath(os.path.join(me_dir, "../examples/igen"))
otter_dir = "~/igen_exps/otter"
otter_progs = {"vsftpd", "ngircd"}

#Note: whenever changing directory, need to recompile coreutils for gcov to work
coreutils_main_dir = "~/igen_exps/coreutils"  
coreutils_doms_dir = CC.getpath(os.path.join(me_dir,"../benchmarks/doms"))
