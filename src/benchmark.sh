#Otter
prog="vsftpd"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="ngircd"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

#CoreUtils
prog="uname"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="mv"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="cat"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="ln"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full 2>&1
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="date"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="sort"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1

prog="ls"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10 2>&1
