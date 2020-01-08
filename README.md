Run example 
```
$  ./igen2.py -domain ../examples/mbuild2/ex1/

inferred results (5):
1. (0) true (conj): (1) L3
2. (2) (x=6 & y=6) (conj): (1) L0
3. (2) (x=7 | y=7) (disj): (2) L2,L2a
4. (2) (u=3 & v=5) (conj): (2) L4,L5
5. (3) (x=6 & y=6 & z=10,13,14) (conj): (1) L1
** done 1 runs, seed 1578460469.13, time 1.1949596405029297, results '/var/tmp/igen_1_normal_ky7sae2c'
./igen2.py -domain ../examples/mbuild2/ex1/  2.45s user 0.57s system 219% cpu 1.378 total

```
Note: 
- use `-log 4` to see lots of debug info
- use `-nomp` to run sequential, no multiprocessing (might be easier to debug). default uses multiple processing
   






Test it out on examples/ (from this directory):

    ./collect.sh make -C examples/
    
On busybox or toybox

    cd /path/to/_box/
    /path/to/multibuild/collect.sh |& tee busybox_files.txt
