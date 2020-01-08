#!/bin/sh

resultsDir="/fs/buzz/ukoc/Dropbox/git/config/benchmarks/ppt/results";
bm=20;
if [ ! -z "$1" ]; then
    bm=$1;
fi

declare -a bm_arr=()
#"uname" "id" "cat" "date" "join" "sort" "ln" "ls"
for i in "${bm_arr[@]}"
do
    python2.7 igen.py "$i" --do_perl --benchmark ${bm} --logger_level 4 --seed 0 > $resultsDir/${i}_ppt_bm${bm}.txt 2>&1
done


declare -a full_arr=("date") 
#"uname" "id" "cat" "join" "sort" "ln"
for i in "${full_arr[@]}"
do
    python2.7 igen.py "$i" --do_perl --do_full --logger_level 4 --seed 0 > $resultsDir/${i}_ppt_full.txt 2>&1
done
