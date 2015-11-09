# README #

iGen (Interaction Generator) is a dynamic analysis tool that discovers interactions among configurable options from traces. iGen employs a stochastic method to iteratively search for a small and "good" set of configurations to find desired interactions.  iGen performs trace partitioning on both positive and negative traces to generate expressive interactions, e.g., combination of both conjunctive and disjunctive formulae. Preliminary results show that the tool is highly efficient and effective on a set of benchmarks consisting of highly-configurable software, e.g., apache httpd, mysql.

## SETUP ##

The source code of iGen is released under the BSD license and can be downloaded using the commands

```
hg clone https://nguyenthanhvuh@bitbucket.org/nguyenthanhvuh/igen/ 
hg clone https://nguyenthanhvuh@bitbucket.org/nguyenthanhvuh/common_python/
```

iGen uses Python and uses SMT solving to check some results.  The tool has been tested using:

* Debian Linux 7 (Wheezy)
* Python 2.7.x
* Microsoft Z3 SMT solver 4.x

Setup Z3 using its own build instruction. Make sure Z3 is setup correctly so that you can do `import z3` in a Python interpreter.

Then have something like this in `~/.bash_profile`

```
export $IGEN=/PATH/TO/IGEN
export CONFIG=$IGEN/config
export PYTHONPATH=$COMMON_PYTHON_DIR/:$CONFIG:/Z3_DIR/build/
export PATH
```

## RUN ##
We can now run iGen to generate interactions using a simple example `$IGEN/example/ex.c`.  
```
#!c

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

We use iGen to automatically generate the interactions annotated next to different program locations, e.g., `x & y & (z=0|3|4)` at `L4`.
 
```
#!shell
$ cd $igen/examples
$ gcc ex.c -o ex.Linux.exe  #compile `ex.c`
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script run_script -seed 0  #call iGen 

# which produces the results
...
1. (0) true (conj): (1) L3
2. (2) (u=1 & v=1) (conj): (1) L4
3. (2) (x=1 & y=1) (conj): (1) L0
4. (2) (x=0 | y=0) (disj): (2) L2,L2a
5. (3) (x=1 & y=1 & z=0,3,4) (conj): (1) L1
6. (4) (s=1 | t=1) & (u=1 & v=1) (mix): (1) L5

```

Thus in its most basic form iGen requires a `dom_file` that contains the domains of the interested options e.g., `z` has 4 possible values, and a `run_script` to obtain program coverage, e.g., running `ex` on `s=1 t=1 ... z=1` covers lines `L1, L3, L4, L5`.


### Options ###
iGen accepts several options to generate interactions.  By default, the tool uses the CEGIR process that finds interactions and iterative refines them.  Other options include:

```
#!shell

# analyze *all* possible configurations (thus only applicable to programs having small numbers of configurations)
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script run_script -do_full 

# analyze only `N` randomly generated configurations
$ python -O $IGEN/src/igen.py -dom_file ex.dom -run_script run_script -rand_n N
```

## ADVANCED USAGE ##
This section lists more advanced usages of iGen.


## Experiments ##
We describe steps to reproduce some more complex experiments 

### GNU Coreutils ###

We use `gcc` and `gcov` to obtain coverage information for `coreutil` commands. First, download `coreutils-8-23` from http://ftp.gnu.org/gnu/coreutils/coreutils-8.23.tar.xz. Then
```
$ mkdir mycoreutils; cd mycoreutils; tar xf /PATH/TO/coreutils-8.23.tar.xz; ln -sf coreutils-8.23 coreutils; cd coreutils
```

Next compile these programs as follows
```
#!shell
    $ mkdir obj-gcov/; cd obj-gcov
    $ ../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage" #make sure no error
    $ make  #make sure  no error
    $ cd src  #obj-gcov/src dir contains binary programs
    $rm -rf *.gcda

    #Let's tests to see that it works
    $ uname #this creates gcov data file uname.gcda -- *make sure that it does*, if not then it doesn't work !
    $ cd ../../src  (src dir containing src)
    $ gcov uname.c -o ../obj-gcov/src  #reads the echo.gcda in obj-gcov and generates human readable format
    File '../src/uname.c'
    Lines executed:37.50% of 88
    Creating 'uname.c.gcov'

    File '../src/system.h'
    Lines executed:0.00% of 10
    Creating 'system.h.gcov'

    #you should see some lines like above saying "lines executed ... X% of Y
    #iGen uses the generated uname.c.gcov and system.h.gcov files for coverage.    
```

Finally, edit `$IGEN/src/igen_settings.py` so that coreutils_dir points to `mycoreutils` directory. 
*NOTE*: if you move the above directories (e.g., `mycoreutils`) to different locations,  it's best to recompile everything again.


If everything is done correctly, we can now run iGen for `coreutils` commands (supported commands include `uname, cat, cp, date, hostname, id, join, ln, ls, mv, sort`)

```
# Generate interactions for the `uname` command 
# notice that iGen already contains the necessary runscripts and dom files for `coreutils`
$ python -O $IGEN/src/igen.py uname
...


```




iGen can analyzes the resulting interactions to learn more about program properties.


### Perl Power Tools ###

Perl Power Tools (PPT), are the perl implementation of some coreutil commands.
In the experiments we used PPT version 0.14. You can obtain it from [here](http://search.cpan.org/dist/ppt). We get the coverage insformation, we used [Devel::Cover](http://search.cpan.org/~pjcj/Devel-Cover-1.21/lib/Devel/Cover.pm). Install `Devel::Cover` as follows:

#!shell
$ sudo cpan install Devel::Cover
```

Now you can test `Devel::Cover` as follows:
#!shell
$ cd into/ppt/directory
$ perl -MDevel::Cover bin/date
Devel::Cover 1.21: Collecting coverage data for branch, condition, statement, subroutine and time.
    Pod coverage is unavailable.  Please install Pod::Coverage from CPAN.
Selecting packages matching:
Ignoring packages matching:
    /Devel/Cover[./]
Ignoring packages in:
    /etc/perl
    /usr/local/lib/perl/5.18.2
    /usr/local/share/perl/5.18.2
    /usr/lib/perl5
    /usr/share/perl5
    /usr/lib/perl/5.18.2
    /usr/share/perl/5.18.2
Mon Nov  9 13:57:01 PST 2015
Devel::Cover: Writing coverage database to /home/ugur/Downloads/ppt-0.14/cover_db/runs/1447106221.3912.02250
----------------------------------- ------ ------ ------ ------ ------ ------
File                                  stmt   bran   cond    sub   time  total
----------------------------------- ------ ------ ------ ------ ------ ------
bin/date                              78.7   50.0   33.3   88.8  100.0   65.5
Total                                 78.7   50.0   33.3   88.8  100.0   65.5
----------------------------------- ------ ------ ------ ------ ------ ------


```



### Other Apps ###



*min configs*
```
#!shell
$ python -O $IGEN/igen.py  --replay uname_cegir/run0_n3vGQD/  --do_min_configs uname
```
