# Introduction

In this document we demonstrate how to setup and use iGen.

# Setup
iGen uses Python and uses SMT solving to check results. The tool has been tested using:

* Debian Linux 7 (Wheezy)
* Python 2.7.x
* Microsoft [Z3 SMT solver](https://github.com/Z3Prover/z3/releases) 4.4.1

Note that after building Z3 (using its own build instructions), make sure Z3 is setup correctly so that you can do `import z3` in a Python interpreter. In addition, have something like this in `~/.bash_profile`

```
export COMMON_PYTHON=/PATH/TO/common_python
export IGEN=/PATH/TO/igen
export CONFIG=$IGEN/config
export PYTHONPATH=$COMMON_PYTHON/:$CONFIG:/Z3_DIR/build/
```

# Demonstration: A simple example
We demonstrate iGen using a simple example `$IGEN/examples/igen/ex.c`:

```
int main(int argc, char **argv){
  // options: s,t,u,v,x,y,z.
  // Option z can take values 0-4 while others are bools
  int s = atoi(argv[1]);
  int t = atoi(argv[2]);
  int u = atoi(argv[3]);
  int v = atoi(argv[4]);

  int x = atoi(argv[5]);
  int y = atoi(argv[6]);
  int z = atoi(argv[7]);

  int max_z = 3;

  if (x&&y){
    printf("L0\n"); //x & y                                                                                                        
    if (!(0 < z && z < max_z)){
      printf("L1\n"); //x & y & (z=0|3|4)                                                                                          
    }
  }
  else{
    printf("L2\n"); // !x|!y                                                                                                       
    printf("L2a\n"); // !x|!y                                                                                                      
  }

  printf("L3\n"); // true                                                                                                          
  if(u&&v){
    printf("L4\n"); //u&v                                                                                                          
    if(s||t){
      printf("L5\n");  // (s|t) & (u&v)                           
    }
  }
  return 0;
}
```
Next, compile the program 

```
$ cd $IGEN/examples/igen
#compile `ex.c`
$ gcc ex.c -o ex.Linux.exe
```
## Running iGen using its iterative (default) algorithm
We use iGen to automatically generate the interactions annotated next to different program locations, e.g., `x & y & (z=0|3|4)` at `L1`, as follows:

```
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script ex.run -logger_level 2 -seed 0
```


Thus iGen requires a `dom_file` that contains the domains of the interested options e.g., `z` has 4 possible values, and a `run_script` to obtain program coverage, e.g., running `ex` on `s=1 t=1 ... z=1` covers lines `L1, L3, L4, L5`.  In addition, option `-O` disables debugging code for faster performance.  Option `-logger_level N`, where `N = 0 .. 4`, controls the verbosity of iGen. To see other options, use `-h`.

```
# the above produces
...
inferred results (6):
1. (0) true (conj): (1) L3
2. (2) (u=1 & v=1) (conj): (1) L4
3. (2) (x=1 & y=1) (conj): (1) L0
4. (2) (x=0 | y=0) (disj): (2) L2,L2a
5. (3) (x=1 & y=1 & z=0,3,4) (conj): (1) L1
6. (4) (s=1 | t=1) & (u=1 & v=1) (mix): (1) L5
... 
* done 1 runs, seed 0.0, time 2.23679900169, results in '/var/tmp/igen_1_normal_noname_hPVkLt'
```

For this run, iGen produces 6 interactions, e.g., conjunctive interaction  `x & y & (z=0|3|4)` at `L1` and the mixed interaction `(s=1 | t=1) & (u=1 & v=1)` at 'L4'.
We also see that the run takes `2.2s` and uses seed `0.0` (which can be used to reproduce the run).
Finally, iGen saves all data in the directory  `/var/tmp/igen_1_normal_noname_hPVkLt` so that it can rerun the experiment (more on this in the *Analyze Results* section below).

## Additional run options

We can run iGen exhaustively by creating *all* possible configurations using the `-do_full` option (thus only applicable to programs having small numbers of configurations):
```
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script ex.run -do_full
...
inferred results (6):
** done 1 runs, seed 0.0, time 9.13450908661, results '/var/tmp/igen_1_full_noname_ZF5kxs'
```

We can also tell iGen to create configurations *randomly* using the `-rand_n` option.
```
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script ex.run -rand_n 10
inferred results (2):
...
** done 1 runs, seed 0.0, time 0.358823060989, results '/var/tmp/igen_1_normal_noname_1p8CZz'
```
Note that iGen can only achieve 2 interactions from the 10 randomly generated configurations.

iGen uses a heuristcs to generate results, thus we can run iGen several times using the `-benchmark` option:
```
# run iGen `3` times
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script ex.run -logger_level 2 -seed 0 -benchmark 3
...
** done 3 runs, seed 0.0, time 5.59616589546, results '/var/tmp/igen_3_normal_noname_16OTNe'
```
Here, iGen saves all data of these runs in the directory `/var/tmp/igen_3_normal_noname_16OTNe` for later analysis.

Finally, use the `-help` command to find out about other run options.

# Analyze iGen's Results

The directory `/var/tmp/igen_1_normal_noname_hPVkLt` generated above run consists of data generated during the run. We can analyze the data in this directory to rerun or `replay` the experiment:


```
$ python -O $IGEN/src/igen.py /var/tmp/igen_1_normal_noname_hPVkLt
inferred results (6):
...
```

Similarly, we can analyze results of multiple runs using the directory `/var/tmp/igen_3_normal_noname_16OTNe` given above:

```
$ python -O $IGEN/src/igen.py /var/tmp/igen_3_normal_noname_16OTNe
inferred results (6):
...
inferred results (6):
...
inferred results (6):
...
19:54:1:analysis:Info:*** Analysis over 3 runs ***
19:54:1:analysis:Info:iter 9.0 (0.5), ints 6.0 (0.0), time 2.12061905861 (0.0790084600449), xtime 0.792189121246 (0.0222933292389), configs 47.0 (2.0), covs 7.0 (0.0), nminconfigs 0.0 (0.0), nmincovs 0.0 (0.0)
19:54:1:analysis:Info:Int types: conjs 4.0 (0.0), disjs 1.0 (0.0), mixed 1.0 (0.0)
19:54:1:analysis:Info:Int strens: (0, 1.0 (0.0), 1.0 (0.0)), (2, 3.0 (0.0), 4.0 (0.0)), (3, 1.0 (0.0), 1.0 (0.0)), (4, 1.0 (0.0), 1.0 (0.0))

```

Here, iGen analyzes these `3` runs and computes results such as: the median of the number of generated configurations  is `47` (with SIQR `2.0`), the median of the number of conjunctive interactions generated is `4` (with SQIR `0.0`), etc.  For detailed descriptions on these, see the *Evaluation* section in the paper.

## Additional Analyses
We can compare iGen's iterative algorithm to the exhaustive run: 
```
python -O $IGEN/src/igen.py /var/tmp/igen_1_normal_noname_hPVkLt/ -cmp_dir /var/tmp/igen_1_full_noname_ZF5kxs/run0_AVgFYA/ -evolution
...
fscores (iter, fscore, configs): (1, 0.5624046406655102, 6) -> (2, 0.6984126984126985, 16) -> (3, 0.6984126984126985, 21) -> (4, 0.6984126984126985, 24) -> (5, 0.7067669172932332, 27) -> (6, 0.7142857142857143, 32) -> (7, 0.7142857142857143, 34) -> (8, 0.7142857142857143, 35) -> (9, 0.7142857142857143, 37) -> (10, 0.7142857142857143, 39)
...
configs (47/320) cov (7/7) fscore 1.0
```
`Configs (47/320)` shows that we generated `47` configurations (exhaustive run creates `320` configurations), and `cov (7/7)` shows that we covered `7` lines (exhaustive run also covers `7` lines).
Finally, the computed `f-score` shows the similarity between interations generated by iterative algorithm and those from the exhaustive run (closer to `1.0` indicates ver similiar and closer to `0.0` means very different).

We can also compare iGen's random algorithm to the exhaustive run:
```
python -O $IGEN/src/igen.py /var/tmp/igen_1_normal_noname_1p8CZz/ -cmp_dir /var/tmp/igen_1_full_noname_ZF5kxs/run0_AVgFYA/ -evolution
...
configs (10/320), cov (4/7), fscore 0.0909090909091
```
As expected, we achieve very low `f-score` indicating the results of the random algorithm (generated from `10` randomly created configurations) are very different than the results of the exhaustive run.


We can use interactions to compute a minimal set of configurations with high coverage
```
python -O $IGEN/src/igen.py `/var/tmp/igen_1_normal_noname_hPVkLt  -do_minconfig -dom_file ex.dom -run_script ex.run
....
minset: 2 configs cover 7/7 sids (time 0.0823450088501s)
...
1. s=1 t=0 u=1 v=1 x=1 y=1 z=0: (5) L0,L1,L3,L4,L5
2. s=1 t=0 u=0 v=1 x=0 y=1 z=0: (3) L2,L2a,L3
```

Finally, use the `-help` command to find out about other analysis options.

# Other Wiki's
To setup and run the experiments (Coreutils, Apache Httpd, etc) used in the FSE'16 paper, go to the follow wiki's:

* [experiment setups](exps_setups): instructions to setup and run experiments in the paper
* [experiment runs](exps_runs): instructions to reproduce the results in the paper
