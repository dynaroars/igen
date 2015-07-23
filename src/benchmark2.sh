#CoreUtils
prog="uname"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 20 > ~/Dropbox1/config_results/${prog}_bm20 2>&1

prog="sort"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 20 > ~/Dropbox1/config_results/${prog}_bm20 2>&1

prog="ls"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 20 > ~/Dropbox1/config_results/${prog}_bm20 2>&1
