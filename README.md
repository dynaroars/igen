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

export DROPBOX=$HOME/Dropbox
export CONFIG=$DROPBOX/git/igen/config
export PYTHONPATH=$DROPBOX/git/common_python:/fs/buzz/ukoc/z3/z3/build/:$CONFIG
export PATH
```

### Experiments ###

*GNU Coreutils*: we use *gcov* to obtain coverage information for coreutils commands. Download and unpack coreutils-8.23.tar.bj2 (other versions probably should work too), then cd to coreutils-8.23 dir.  

*NOTE*: if you move this directory to a different location,  it's best to recompile everything again using the below instruction.  

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

    Now we can analyze echo.c.gcov file and system.h.gcov, butiGen will take care of this task
    10. Finally, edit config_setup.py in src (where igen.py is) so that coreutils_dir points to the coreutils-8.23 dir