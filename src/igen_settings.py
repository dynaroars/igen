import os.path
import vu_common as CM

me_file = CM.getpath(__file__)
me_dir = os.path.dirname(me_file)

tmp_dir = "/var/tmp"

examples_dir = CM.getpath(os.path.join(me_dir, "../examples/igen"))
otter_dir = "~/igen_exps/otter"
otter_progs = {"vsftpd", "ngircd"}

#Note: whenever changing directory, need to recompile coreutils for gcov to work
coreutils_main_dir = "~/igen_exps/mycoreutils"  
coreutils_doms_dir = CM.getpath(os.path.join(me_dir,"../benchmarks/doms"))
