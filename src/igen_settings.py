import os.path
import vu_common as CM

me_file = CM.getpath(__file__)
me_dir = os.path.dirname(me_file)

tmp_dir = "/var/tmp"

examples_dir = CM.getpath(os.path.join(me_dir, "../examples/igen"))
otter_dir = "~/igen_exps/otter"
otter_progs = {"vsftpd", "ngircd"}

#Note: whenever changing directory, need to recompile coreutils for gcov to work
coreutils_main_dir = "~/igen_exps/coreutils"  
coreutils_doms_dir = CM.getpath(os.path.join(me_dir,"../benchmarks/doms"))



#pylint stuff
pylint_dir = "~/igen_exps/pylint"
pylint_tests_dir = CM.getpath(os.path.join(pylint_dir, "tests"))
pylint_additional_tests_dir = CM.getpath(
    os.path.join(me_dir, "../benchmarks/additional_tests/pylint"))

#grin stuff
grin_dir = CM.getpath("~/igen_exps/grin")
grin_tests_dir = os.path.join(grin_dir, "tests")
grin_additional_tests_dir = os.path.join(
    me_dir, "../benchmarks/additional_tests/grin")
    



