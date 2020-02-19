import os.path
import vcommon as CM
from pathlib import Path

tmp_dir = Path("/var/tmp")
logger_level = 3

me_file = CM.getpath(__file__)
me_dir = os.path.dirname(me_file)


examples_dir = CM.getpath(os.path.join(me_dir, "../examples/igen"))
otter_dir = Path("~/igen_exps/otter")
otter_progs = {"vsftpd", "ngircd"}

# Note: whenever changing directory, need to recompile coreutils for gcov to work
coreutils_main_dir = Path("~/igen_exps/coreutils")
coreutils_doms_dir = CM.getpath(os.path.join(me_dir, "../benchmarks/doms"))

allow_known_errors = False
show_cov = True
analyze_outps = False

doMP = True
