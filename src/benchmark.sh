#Otter
prog="vsftpd"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_gt > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ngircd"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_gt > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

#CoreUtils
prog="uname"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="mv"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="cat"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ln"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --do_full > ~/Dropbox1/config_results/${prog}_full
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="date"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="sort"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10

prog="ls"
echo ${prog}
python igen.py ${prog}  --logger_level 4 --seed 0 --benchmark 10 > ~/Dropbox1/config_results/${prog}_bm10
