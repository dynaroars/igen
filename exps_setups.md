# Setup FSE experiments
This section shows how to setup and run the programs used in the FSE'16 paper *iGen: Dynamic Interaction Inference for Configurable Software*. The instructions below have been tested on a Debian machine.

For demonstration, we will setup these experiments in the  directory `~\igen_exps`.

## GNU Coreutils

### Setup

We use `gcc` and `gcov` to obtain coverage information for `coreutil` commands. First, download [`coreutils-8-23`](http://ftp.gnu.org/gnu/coreutils/coreutils-8.23.tar.xz). Then
```
$ cd ~/igen_exps/mycoreutils; tar xf /PATH/TO/coreutils-8.23.tar.xz; ln -sf coreutils-8.23 coreutils; cd coreutils; tar xf $IGEN/scripts/coreutils_testfiles.tar.gz
```
Next compile these programs
```
#!shell
$ mkdir obj-gcov/; cd obj-gcov
$ ../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage" #make sure no error
# if want to get cfg then use `../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage -fdump-tree-cfg-lineno"`
$ make  #make sure  no error
$ cd src  #obj-gcov/src dir contains binary programs
$ rm -rf *.gcda

# Test to see that it works
$ ./uname #this creates gcov data file uname.gcda -- *make sure that it does*, if not then it doesn't work !
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

Finally, edit `$IGEN/src/igen_settings.py` so that `coreutils_dir` points to `~/igen_exps/mycoreutils` directory. 
*NOTE*: if you move the above directories (e.g., `mycoreutils`) to different locations,  it's best to recompile everything again.

### Run
If everything is done correctly, we can now run iGen for `coreutils` commands (supported commands include `uname, cat, cp, date, hostname, id, join, ln, ls, mv, sort`)

```
# Generate interactions for the `uname` command 
# notice that iGen already contains the necessary runscripts and dom files for `coreutils`
$ python -O $IGEN/src/igen.py uname
...
```

## Perl Powertools
We also experiment with the [Perl Power Tools implementation](http://search.cpan.org/dist/ppt) (ppt) of several `coreutils` programs. 

### Setup
First, download [`ppt 0.14`](http://search.cpan.org/CPAN/authors/id/C/CW/CWEST/ppt-0.14.tar.gz), then 

```
$ cd ~/igen_exps; mkdir ppt; cd ppt; mkdir archive; tar xf PATH/TO/ppt-0.14.tar.gz; cd ppt-0.14
```


To obtain coverage, we use `Devel::Cover` and `HTML::TableExtract` modules, which can be installed on a Debian-based system using
```
$sudo apt-get install libdevel-cover-perl libhtml-tableextract-perl 
```

Now, do some tests to make sure everything works

```
#!shell
$ perl -MDevel::Cover bin/date
Devel::Cover 1.21: Collecting coverage data for branch, condition, statement, subroutine and time.
...
Devel::Cover: Writing coverage database to ...
----------------------------------- ------ ------ ------ ------ ------ ------
File                                  stmt   bran   cond    sub   time  total
----------------------------------- ------ ------ ------ ------ ------ ------
bin/date                              78.7   50.0   33.3   88.8  100.0   65.5
Total                                 78.7   50.0   33.3   88.8  100.0   65.5
----------------------------------- ------ ------ ------ ------ ------ ------

```

If you see a coverage report similar to above one, `ppt` and `Devel::Cover` work fine.

Finally, edit `$IGEN/scripts/pptCoverageHelper.pl` so that `$SUT_DIR` points to the `ppt` directory. Then, test `pptCoverageHelper.pl` script as follows:

```
#!shell
$ cd $IGEN/scripts; ./pptCoverageHelper.pl "@@@date " 
```
It will output the name of coverage file if everything works fine.

Finally, domain files for PPT programs are located under benchmarks/doms and igen already know about them.
There is no need to change these domain files.

### Run
Now to run igen on `PPT` programs, you need to use `-do_perl` option. For example `PPT uname` can be run as follows:

```
$ python -O $IGEN/src/igen.py uname -do_perl
```

## Ack and Cloc ##

## Setup
[`Ack`](http://search.cpan.org/dist/ack/ack) is a grep-like search tool for developers and [`cloc`](http://cloc.sourceforge.net/) is line of code counter. They both are written in `Perl`. To get their coverage, use the instructions above for `Powertools` to install `Devel::Cover`.

## Run
To run `iGen` on either of them you need to use `-run_script` and `-dom_file` options (as in httpd).
Run scripts for them are located under scripts directory, and the domain files are under benchmarks/doms directory.

For `ack`:
```
#!shell
$ python -O $IGEN/igen.py "ack" -run_script /path/to/run_script -dom_file /path/to/domain_file
```

And for `cloc`:
```
$ python -O $IGEN/igen.py "cloc" -run_script /path/to/run_script -dom_file /path/to/domain_file
```
Todo: replace above /path/to/ with real examples

## Ocaml Apps

## Haskell Apps

## Python Apps


## VSFTPD and NGIRCD

## Apache HTTPD
### Setup
Apache Httpd is a webserver. In our experiments we used [`httpd 2.2.29`](http://archive.apache.org/dist/httpd/). 

### Run
To run igen for httpd, `-run_script` and `-dom_file` options are needed to be used as follows:

```
#!shell
$ python -O $IGEN/src/igen.py "httpd" -run_script /path/to/run_script -dom_file /path/to/domain_file
```

`run_script` for httpd is under `scripts` directory: `run_httpd.pl` and domain file is under` benchmarks/doms/dom_httpd` directory: `config_space_model.dom`.

In `run_httpd.pl`, the variable `$SUT_DIR` should be updated accordingly.

