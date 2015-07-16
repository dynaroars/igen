#Otter
prog="vsftpd"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ngircd"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

#CoreUtils
prog="uname"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="mv"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="cat"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ln"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="date"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="sort"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ls"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10
