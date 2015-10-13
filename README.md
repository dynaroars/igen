# README #

iGen (Interaction Generator) is a dynamic analysis tool that discovers interactions among configurable options from traces. iGen employs a stochastic search method to iteratively search for a small and "good" set of configurations to find desired interactions.  Igen performs trace partitioning on both positive and negative traces to generate expressive interactions, e.g., combination of both conjunctive and disjunctive formulae. Preliminary results show that Intgen is highly efficient and effective on a set of benchmarks consisting of highly-configurable software, e.g., apache httpd, mysql.

## Setup ##

The source code of iGen is released under the BSD license and can be downloaded using the command
"hg clone https://nguyenthanhvuh@bitbucket.org/nguyenthanhvuh/igen/" and "hg clone https://nguyenthanhvuh@bitbucket.org/nguyenthanhvuh/common_python/"

iGen uses Python. Some operations, e.g., verifying candidate interactions, require a recent SMT solver.  

iGen has been tested using the following setup:

* Debian Linux 7 (Wheezy)
* Python 2.7.x
* Microsoft Z3 SMT solver 4.x

Setup Z3 using its own build instruction. Make sure Z3 is setup correctly so that you can do "import z3" in a Python interpreter.

Then in ~/.bash_profile, have something like this 

```
#!script

export IGEN=$IGEN_DIR/git/igen/
export CONFIG=$IGEN_DIR/config
export PYTHONPATH=$/COMMON_PYTHON_DIR/:/Z3_DIR/build/:$CONFIG
export PATH
```

### Running iGen ###
We can now try to run iGen to generate interactions using a simple example `$IGEN_DIR/example/ex.c`.  

```
#!c

int main(int argc, char **argv){

  // options: s,t,u,v, x,y,z                                                                                                       

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

Here we want to use iGen to automatically generate the interactions annotated next to different program locations, e.g., `x & y & (z=0|3|4)` at `L4`.  For the impatient, we can invoke iGen as follows
```
#!shell

$ cd igen/examples
$ gcc ex.c -o ex.Linux.exe  #compile `ex.c`
$ python -O ../src/igen.py --dom_file ex.dom -run_script run_script "prog" --seed 0  #call iGen 

# which produces the results
...

1. (0) true (conj): (1) L3
2. (2) (u=1 & v=1) (conj): (1) L4
3. (2) (x=1 & y=1) (conj): (1) L0
4. (2) (x=0 | y=0) (disj): (2) L2,L2a
5. (3) (x=1 & y=1 & z=0,3,4) (conj): (1) L1
6. (4) (s=1 | t=1) & (u=1 & v=1) (mix): (1) L5

```

### Experiments ###
We describe steps to reproduce some more complex experiments 

*GNU Coreutils*: We use `gcc` and `gcov` to obtain coverage information for coreutil commands. First, download coreutils-8-23 from http://ftp.gnu.org/gnu/coreutils/coreutils-8.23.tar.xz.  Then unpack and cd to the coreutils-8.23 dir.  Next we'll compile these programs as follows.

*NOTE*: if you move this directory to a different location,  it's best to recompile everything again.

    1. mkdir obj-gcov/
    2. cd obj-gcov
    3. ../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage"
    #make sure no error
    4. make  #make sure  no error
    5. cd src  #obj-gcov/src dir contains binary programs
    6. rm -rf *.gcda

    Now let's tests to see that it works
    7. run the test suite,  e.g.,  ./echo** #this creates gcov data file echo.gcda -- *make sure that it does*, if not then it doesn't work !
    8. cd ../../src  (src dir containing src)
    9. gcov echo.c -o ../obj-gcov/src  #reads the echo.gcda in obj-gcov and generates human readable format

    For example,
    $ gcov echo.c -o ../obj-gcov/src/
    File '../src/echo.c'
    Lines executed:22.02% of 109
    Creating 'echo.c.gcov'

    File '../src/system.h'
    Lines executed:0.00% of 10
    Creating 'system.h.gcov'

    iGen uses the generated echo.c.gcov and system.h.gcov files for coverage.
    10. Finally, edit config_setup.py in src (where igen.py is) so that coreutils_dir points to the coreutils-8.23 dir