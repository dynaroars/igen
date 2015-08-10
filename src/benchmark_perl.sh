#!/bin/sh

resultsDir="/fs/buzz/ukoc/Dropbox/git/config/benchmarks/ppt/results";

#python2.7 igen.py "ln" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/ln_ppt.txt 2>&1
#python2.7 igen.py "ln" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/ln_ppt_full.txt 2>&1

#python2.7 igen.py "date" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/date_ppt.txt 2>&1

#exit

#python2.7 igen.py "uname" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/uname_ppt.txt 2>&1
#python2.7 igen.py "uname" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/uname_ppt_full.txt 2>&1

#python2.7 igen.py "id" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/id_ppt.txt 2>&1
#python2.7 igen.py "id" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/id_ppt_full.txt 2>&1

#python2.7 igen.py "cat" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/cat_ppt.txt 2>&1
#python2.7 igen.py "cat" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/cat_ppt_full.txt 2>&1

#python2.7 igen.py "sort" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/sort_ppt.txt 2>&1
#python2.7 igen.py "sort" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/sort_ppt_full.txt 2>&1

python2.7 igen.py "join" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/join_ppt.txt 2>&1
python2.7 igen.py "join" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/join_ppt_full.txt 2>&1

#python2.7 igen.py "touch" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/touch_ppt.txt 2>&1
#python2.7 igen.py "touch" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/touch_ppt_full.txt 2>&1  

python2.7 igen.py "ls" --do_perl --benchmark 20 --logger_level 4 --seed 0 > $resultsDir/ls_ppt.txt 2>&1
