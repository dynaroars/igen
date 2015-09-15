

prog="vsftpd"
echo ${prog}
python2.7 igen.py ${prog}  --logger_level 4 --benchmark 20 > ~/Dropbox1/config_results/${prog}_bm20_2 2>&1
